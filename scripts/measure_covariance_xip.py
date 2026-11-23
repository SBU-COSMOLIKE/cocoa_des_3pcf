"""Compare a bounded map measurement with Rafael's released xi-plus rows.

The joint covariance has 200 xi-plus, 200 xi-minus and 80 Map3 rows. This
check measures only source-bin slot four, selecting the four xi-plus bins
from 39.622 to 99.527 arcminutes. These are the released bin edges. Equal pixel
weights, NSIDE 512, and the unresolved shear signs prevent claiming the
published catalogue estimator is exactly reproduced.

TreeCorr supplies the shear pair measurement (Mike Jarvis and contributors):
https://rmjarvis.github.io/TreeCorr/_build/html/gg.html
Both g2 signs are measured on the first map; all parents then use the
explicit east/north assumption (flip_g2=True). Agreement is not used to
choose a convention. Only cut zero and draw zero of each selected run are used.
No precision correction is trained on the released covariance: it may share
parent skies with these files and is itself simulation-estimated.

Run from Cocoa after installing TreeCorr 5.1.2 and LSSTDESC.Coord 1.3.0:
  python projects/des_3pcf/scripts/measure_covariance_xip.py --runs 32

The first eight runs also test cross-correlations between realization labels
and between run labels. At most two maps are held at once, and only small
text vectors are saved in chains/.
"""

import argparse
import os
from pathlib import Path
import time

# Leave room for the independent, single-threaded aperture-map stream.
# Three OpenMP workers leave one core for that stream; BLAS stays one.
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy_healpix import HEALPix
import treecorr
from threadpoolctl import threadpool_limits


def main():
    """Measure exact angular bins and check the tree against direct pairs.

    TreeCorr's spherical basis points west and north. Flipping g2 therefore
    matches the east/north assumption used in the coarse aperture pilot.
    The alternative sign on the first map records the unresolved convention.
    Both approximation tolerances are zero. A 2,048-pixel subset is also
    checked with brute-force pairs, independently of tree grouping.

    Returns:
      nothing; prints progress and the output directory.
    Raises:
      ValueError if the requested count, map shape or released bins disagree.
      AssertionError if exact tree pairs disagree with direct enumeration.
    Side effects:
      Writes xi-plus vectors, angular rows, tree/direct-pair checks and provenance
      under chains/covariance_xip_comparison. Never changes input arrays.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=32)
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError(
            f"Requested {args.runs} runs; choose at least one for a measurement "
            "and more than one for a covariance."
        )
    project = Path(__file__).resolve().parents[1]
    folder = project / "chains/rafael/maps_cov_512_v2"
    data = project / "data" / "upstream_y3_observed" / "data"
    output = project / "chains" / "covariance_xip_comparison"
    indices = np.load(file=folder / "indices_512.npy", allow_pickle=False)
    healpix = HEALPix(nside=512, order="ring")
    longitude, latitude = healpix.healpix_to_lonlat(healpix_index=indices)
    ra = longitude.to_value(unit=u.deg)
    dec = latitude.to_value(unit=u.deg)
    rows = fits.getdata(
        filename=data / "dv" / "real_2pt_desy3_unblind_g2flip_matchElisa_2025Mar10.fits",
        extname="xip",
    )
    selected = (rows["BIN1"] == 4) & (rows["BIN2"] == 4)
    selected &= rows["ANGLEMIN"] >= 39.
    selected &= rows["ANGLEMAX"] <= 100.
    published_rows = np.flatnonzero(a=selected)
    angular_bins = rows["ANGBIN"][selected]
    if not np.array_equal(angular_bins, np.arange(12, 16)):
        raise ValueError("Released source-bin-four angular rows changed; inspect FITS.")
    paths = sorted(folder.glob("ShearMaps_DESY3_run_*_ncut0_nreal0.npy"))[:args.runs]
    if len(paths) != args.runs:
        raise ValueError(f"Requested {args.runs} primary maps; found {len(paths)}.")
    output.mkdir(exist_ok=True, parents=True)
    np.savetxt(
        fname=output / "rows.txt",
        X=np.column_stack((published_rows, rows["ANGLEMIN"][selected],
                           rows["ANGLEMAX"][selected], rows["ANG"][selected])),
        header="joint_row theta_min_arcmin theta_max_arcmin published_mean_theta",
    )
    start = time.monotonic()
    with threadpool_limits(limits=1, user_api="blas"):
        with (output / "xip.txt").open("w") as stream:
            stream.write("# run flip_g2 followed by four xi-plus entries\n")
            for number, path in enumerate(paths):
                run = int(path.name.split("_run_")[1].split("_")[0])
                shear = np.load(file=path, mmap_mode="r", allow_pickle=False)
                if shear.shape != (4, 2, len(indices)):
                    raise ValueError(f"Unexpected map shape in {path.name}.")
                flips = (True,)
                if number == 0:
                    flips = (False, True)
                for flip in flips:
                    catalog = treecorr.Catalog(
                        ra=ra,
                        dec=dec,
                        ra_units="deg",
                        dec_units="deg",
                        g1=np.array(shear[3, 0]),
                        g2=np.array(shear[3, 1]),
                        flip_g2=flip,
                    )
                    correlation = treecorr.GGCorrelation(
                        min_sep=float(rows["ANGLEMIN"][selected][0]),
                        max_sep=float(rows["ANGLEMAX"][selected][-1]),
                        nbins=4,
                        sep_units="arcmin",
                        bin_slop=0.,
                        angle_slop=0.,
                        metric="Arc",
                        verbose=0,
                    )
                    correlation.process(cat1=catalog, num_threads=3)
                    np.testing.assert_allclose(
                        correlation.left_edges,
                        rows["ANGLEMIN"][selected],
                        rtol=1.e-12,
                    )
                    vector = correlation.xip
                    stream.write(f"{run} {int(flip)} "
                                 + " ".join(f"{x:.17e}" for x in vector) + "\n")
                    if number == 0:
                        sample = np.linspace(0, len(indices) - 1, 2048, dtype=int)
                        small_catalog = treecorr.Catalog(
                            ra=ra[sample],
                            dec=dec[sample],
                            ra_units="deg",
                            dec_units="deg",
                            g1=np.array(shear[3, 0, sample]),
                            g2=np.array(shear[3, 1, sample]),
                            flip_g2=flip,
                        )
                        exact = treecorr.GGCorrelation(config=correlation.config)
                        exact.process(cat1=small_catalog, num_threads=3)
                        direct = treecorr.GGCorrelation(
                            config=correlation.config,
                            brute=True,
                        )
                        direct.process(cat1=small_catalog, num_threads=3)
                        np.testing.assert_allclose(exact.xip, direct.xip, atol=1.e-16)
                        np.testing.assert_allclose(exact.npairs, direct.npairs, atol=0.)
                        np.savetxt(
                            fname=output / f"tree_check_flip{int(flip)}.txt",
                            X=np.column_stack((exact.xip, direct.xip, exact.npairs)),
                            header="tree_xip brute_xip pair_count; 2048-pixel subset",
                        )
                        del small_catalog, exact, direct
                    del catalog, correlation
                del shear
                stream.flush()
                print(f"Measured {number + 1}/{len(paths)} parents", flush=True)
        # If nreal changes only independent shape noise, correlating two draws
        # of one sky keeps the common sky signal but averages away that noise.
        # Different run labels provide a control whose ensemble signal should
        # vanish if the runs are independent. One pair is noisy: repeat eight
        # pairs, with disjoint run labels in the different-run control.
        pair_count = min(8, len(paths) // 2)
        with (output / "cross_xip.txt").open("w") as stream:
            stream.write("# run partner_run partner_draw followed by four cross-xip\n")
            for number in range(pair_count):
                path = paths[number]
                run = int(path.name.split("_run_")[1].split("_")[0])
                shear = np.load(file=path, mmap_mode="r", allow_pickle=False)
                catalog = treecorr.Catalog(
                    ra=ra,
                    dec=dec,
                    ra_units="deg",
                    dec_units="deg",
                    g1=np.array(shear[3, 0]),
                    g2=np.array(shear[3, 1]),
                    flip_g2=True,
                )
                del shear
                partners = (
                    path.with_name(path.name.replace("_nreal0.npy", "_nreal1.npy")),
                    paths[number + pair_count],
                )
                for partner_path in partners:
                    partner_run = int(partner_path.name.split("_run_")[1].split("_")[0])
                    partner_draw = int(partner_path.stem.split("_nreal")[1])
                    partner = np.load(
                        file=partner_path,
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                    partner_catalog = treecorr.Catalog(
                        ra=ra,
                        dec=dec,
                        ra_units="deg",
                        dec_units="deg",
                        g1=np.array(partner[3, 0]),
                        g2=np.array(partner[3, 1]),
                        flip_g2=True,
                    )
                    correlation = treecorr.GGCorrelation(
                        min_sep=float(rows["ANGLEMIN"][selected][0]),
                        max_sep=float(rows["ANGLEMAX"][selected][-1]),
                        nbins=4,
                        sep_units="arcmin",
                        bin_slop=0.,
                        angle_slop=0.,
                        metric="Arc",
                        verbose=0,
                    )
                    correlation.process(
                        cat1=catalog,
                        cat2=partner_catalog,
                        num_threads=3,
                    )
                    stream.write(f"{run} {partner_run} {partner_draw} "
                                 + " ".join(f"{x:.17e}" for x in correlation.xip) + "\n")
                    if number == 0 and partner_draw == 1:
                        # Positive control: two distinct catalogue objects
                        # containing the SAME shear field must recover the
                        # auto-correlation. This checks the cross-route's
                        # normalization, not only whether two maps differ.
                        copied_catalog = treecorr.Catalog(
                            ra=ra,
                            dec=dec,
                            ra_units="deg",
                            dec_units="deg",
                            g1=catalog.g1.copy(),
                            g2=catalog.g2.copy(),
                        )
                        self_cross = treecorr.GGCorrelation(config=correlation.config)
                        self_cross.process(
                            cat1=catalog,
                            cat2=copied_catalog,
                            num_threads=3,
                        )
                        primary = np.loadtxt(fname=output / "xip.txt")
                        selected_primary = (primary[:, 0] == run) & (primary[:, 1] == 1)
                        auto = primary[selected_primary, 2:][0]
                        np.testing.assert_allclose(
                            self_cross.xip, auto, rtol=0., atol=1.e-16,
                        )
                        np.savetxt(
                            fname=output / "self_cross_check.txt",
                            X=np.column_stack((auto, self_cross.xip)),
                            header="auto_xip cross_of_two_copies_xip; first primary map",
                        )
                        del copied_catalog, self_cross
                        # The different-map cross route also gets its own
                        # direct-pair check. The g2 values below are already
                        # in TreeCorr's basis, so they must not be flipped again.
                        subset = np.linspace(0, len(indices) - 1, 2048, dtype=int)
                        small_first = treecorr.Catalog(
                            ra=ra[subset],
                            dec=dec[subset],
                            ra_units="deg",
                            dec_units="deg",
                            g1=catalog.g1[subset],
                            g2=catalog.g2[subset],
                        )
                        small_second = treecorr.Catalog(
                            ra=ra[subset],
                            dec=dec[subset],
                            ra_units="deg",
                            dec_units="deg",
                            g1=partner_catalog.g1[subset],
                            g2=partner_catalog.g2[subset],
                        )
                        cross_tree = treecorr.GGCorrelation(config=correlation.config)
                        cross_tree.process(
                            cat1=small_first,
                            cat2=small_second,
                            num_threads=3,
                        )
                        cross_direct = treecorr.GGCorrelation(
                            config=correlation.config,
                            brute=True,
                        )
                        cross_direct.process(
                            cat1=small_first,
                            cat2=small_second,
                            num_threads=3,
                        )
                        np.testing.assert_allclose(
                            cross_tree.xip, cross_direct.xip, rtol=0., atol=1.e-16,
                        )
                        np.testing.assert_array_equal(
                            cross_tree.npairs, cross_direct.npairs,
                        )
                        np.savetxt(
                            fname=output / "cross_tree_check.txt",
                            X=np.column_stack((cross_tree.xip, cross_direct.xip,
                                               cross_tree.npairs)),
                            header="tree_cross_xip brute_cross_xip pair_count; "
                                   "2048 pixels",
                        )
                        del small_first, small_second, cross_tree, cross_direct
                    del partner, partner_catalog, correlation
                del catalog
                stream.flush()
                print(f"Measured cross-correlation pair {number + 1}/{pair_count}",
                      flush=True)
    record = (
        f"Parents: {len(paths)}\nCut: 0\nDraw: 0\nSource-bin slot: 4\n"
        f"TreeCorr version: {treecorr.__version__}\nOpenMP: 3\nBLAS: 1\n"
        "Weights: equal per retained pixel; producer weights unavailable\n"
        "Tree approximations: bin_slop=0 angle_slop=0; 2048-pixel tree/brute check\n"
        "Metric: Arc\nNSIDE: 512\nPixel ordering: RING inferred from footprint\n"
        "Signs: first map both signs; main sample flip_g2=True (assumed east/north)\n"
        "Bins: released xi-plus rows 192..195, 39.622..99.527 arcmin\n"
        "No catalogue-noise covariance from TreeCorr is used\n"
        "Released matrix may share parent simulations; this is a consistency check\n"
        f"Cross tests: {pair_count} same-run draw pairs "
        "and disjoint different-run pairs\n"
        f"Elapsed seconds: {time.monotonic() - start:.3f}\n"
    )
    (output / "measurement.txt").write_text(record)
    print(record, flush=True)


if __name__ == "__main__":
    main()
