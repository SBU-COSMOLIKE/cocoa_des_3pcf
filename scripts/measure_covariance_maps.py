"""Measure a small aperture-moment diagnostic from Rafael's DES shear maps.

This is a pilot for covariance estimation, not the published DES estimator.
It integrates tangential shear around 64 fixed, nearly unmasked centers with
30 and 60 arcminute Gaussian aperture filters. It then averages the squared
and cubed aperture values over centers, separately in each of four source
bins. Those 16 measurements retain noise self-products and use equal pixel
weights; they are not the catalogue xi/finite-triangle Map3 measurements.

The RING pixel ordering is supported by the footprint inspection. The code
ASSUMES equatorial components with g1 along east and g2 at +45 degrees
from east toward north. This shear-basis assumption needs producer confirmation.
The run/cut/noise interpretation also remains conditional on that confirmation.

Only one map is open at a time. No map copies or per-map figures are written.
Outputs are a small text table of moments, a table of primary aperture
values for the Gaussian-reference exercise, and input/geometry records.
Run from Cocoa, with astropy-healpix installed in the active environment:

  python projects/des_3pcf/scripts/measure_covariance_maps.py --max-runs 48

The default reads cut 0 and draws 0..3 of at most 48 runs. --all selects every
complete, settled map currently present, including the other cuts and draws.
It does not establish that the producer's entire release has arrived.
After a completed pass, --all --resume appends newly arrived files to the
same small output tables without rereading previously measured maps.
"""

import argparse
import os
from pathlib import Path
import re
import time

# Preserve the caller's OpenMP allocation; use four only when none is set.
# This driver streams maps serially; sparse multiplication needs no thread pool.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

import numpy as np
from astropy import units as u
from astropy_healpix import HEALPix
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]


def aperture_weights(indices, nside=512):
    """Build the fixed tangential-shear weights without reading any map.

    Equal-area HEALPix pixels approximate the angular integral. The Gaussian
    Q filter is truncated at four aperture radii, retaining 99.698% of its
    flat-sky radial weight. Disk counts reject footprint edges first; a
    second check requires 99% of the truncated filter weight at BOTH radii.
    This also detects small mask holes near a filter's peak. The exact
    retained weight is saved; missing pixels are never filled with zero.

    Arguments:
      indices = one-dimensional HEALPix RING pixel IDs at the given NSIDE.
      nside = map resolution; the angular apertures stay fixed.
    Returns:
      first, second = sparse [128,npixel] matrices multiplying g1 and g2.
      geometry = [64,5] center pixel ID, RA, Dec, two retained Q weights.
    Raises:
      ValueError if there are fewer than 64 sufficiently enclosed centers.
    """
    healpix = HEALPix(nside=nside, order="ring")
    longitude, latitude = healpix.healpix_to_lonlat(healpix_index=indices)
    ra = longitude.to_value(unit=u.rad)
    dec = latitude.to_value(unit=u.rad)
    xyz = np.column_stack((np.cos(dec) * np.cos(ra),
                           np.cos(dec) * np.sin(ra), np.sin(dec)))
    tree = cKDTree(data=xyz)
    # 60 arcminutes is the larger aperture radius; four radii span 4 degrees.
    radii = np.deg2rad([0.5, 1.0])
    outer_angle = 4. * radii[-1]
    chord = 2. * np.sin(outer_angle / 2.)
    pixel_area = 4. * np.pi / (12 * nside**2)
    expected_count = 2. * np.pi * (1. - np.cos(outer_angle)) / pixel_area
    # Pick centers from geometry alone, before any covariance is inspected.
    # The broad candidates follow the retained pixels' deterministic order.
    candidates = np.linspace(0, len(indices) - 1, 2048, dtype=int)
    enclosed = []
    for index in candidates:
        count = tree.query_ball_point(x=xyz[index], r=chord, return_length=True)
        if count >= 0.99 * expected_count:
            nearby = tree.query_ball_point(x=xyz[index], r=chord)
            cosine = xyz[nearby] @ xyz[index]
            angles = np.arccos(np.clip(cosine, -1., 1.))
            covered = True
            for radius in radii:
                x = angles / radius
                weight = pixel_area * x**2 * np.exp(-0.5 * x**2)
                weight /= 4. * np.pi * radius**2
                weight[x > 4.] = 0.
                # Integral of x^3 exp(-x^2/2)/2 from zero to four.
                truncated_weight = 1. - 9. * np.exp(-8.)
                if np.sum(a=weight) < 0.99 * truncated_weight:
                    covered = False
            if covered:
                enclosed.append(index)
    if len(enclosed) < 64:
        raise ValueError("Fewer than 64 enclosed centers; inspect the footprint.")
    positions = np.linspace(0, len(enclosed) - 1, 64, dtype=int)
    centers = np.asarray(enclosed)[positions]
    row_numbers = []
    column_numbers = []
    first_values = []
    second_values = []
    geometry = np.zeros(shape=(64, 5))
    for center_number, center in enumerate(centers):
        neighbors = tree.query_ball_point(x=xyz[center], r=chord)
        neighbors = np.asarray(neighbors, dtype=int)
        cosine = xyz[neighbors] @ xyz[center]
        angle = np.arccos(np.clip(cosine, -1., 1.))
        # Bearing is measured at each shear pixel, toward the aperture center.
        # Using a bearing at the center instead would rotate the wrong basis.
        east = np.column_stack((-np.sin(ra[neighbors]), np.cos(ra[neighbors]),
                                np.zeros(shape=len(neighbors))))
        north = np.column_stack((-np.sin(dec[neighbors]) * np.cos(ra[neighbors]),
                                 -np.sin(dec[neighbors]) * np.sin(ra[neighbors]),
                                 np.cos(dec[neighbors])))
        east_projection = east @ xyz[center]
        north_projection = north @ xyz[center]
        tangent_length2 = east_projection**2 + north_projection**2
        not_center = angle > 1.e-7
        cos2 = np.zeros(shape=len(neighbors))
        sin2 = np.zeros(shape=len(neighbors))
        cos2[not_center] = (east_projection[not_center]**2
                           - north_projection[not_center]**2)
        cos2[not_center] /= tangent_length2[not_center]
        sin2[not_center] = 2. * east_projection[not_center]
        sin2[not_center] *= north_projection[not_center]
        sin2[not_center] /= tangent_length2[not_center]
        geometry[center_number, :3] = [indices[center],
                                      np.rad2deg(ra[center]), np.rad2deg(dec[center])]
        for radius_index, radius in enumerate(radii):
            x = angle / radius
            weight = pixel_area * x**2 * np.exp(-0.5 * x**2)
            weight /= 4. * np.pi * radius**2
            weight[x > 4.] = 0.
            weight[~not_center] = 0.
            geometry[center_number, 3 + radius_index] = np.sum(a=weight)
            used = weight > 0.
            count = np.count_nonzero(a=used)
            row_numbers.extend([radius_index * 64 + center_number] * count)
            column_numbers.extend(neighbors[used])
            first_values.extend(-weight[used] * cos2[used])
            second_values.extend(-weight[used] * sin2[used])
    shape = (128, len(indices))
    first = csr_matrix((first_values, (row_numbers, column_numbers)), shape=shape)
    second = csr_matrix((second_values, (row_numbers, column_numbers)), shape=shape)
    return first, second, geometry


def main():
    """Stream selected complete maps into compact measured-vector tables.

    The filename list is fixed once at startup. Files modified during the
    previous two minutes are deferred, as are files whose NPY payload is
    incomplete. Size and timestamp are rechecked after measurement. These
    checks detect ordinary partial transfers, not corruption or a completed
    release; producer checksums are still needed for scientific use.

    Returns:
      nothing; prints counts, elapsed seconds, and output location.
    Side effects:
      Writes only small text files under the chosen chains/ directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int, default=48)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output
    if output is None:
        output = PROJECT / "chains" / "covariance_maps_pilot"
        if args.all:
            output = PROJECT / "chains" / "covariance_maps_full"
    output = output.resolve()
    if not output.is_relative_to((PROJECT / "chains").resolve()):
        raise ValueError("Generated measurements must stay in des_3pcf/chains/.")
    if args.max_runs < 3:
        raise ValueError("--max-runs must be at least three.")
    # Completed passes can append newly downloaded files without rereading
    # hundreds of GB. The small manifest identifies exactly what was measured.
    previous = {}
    if args.resume:
        for line in (output / "inputs.txt").read_text().splitlines():
            if line and not line.startswith("#"):
                name, size, modified = line.split()
                previous[name] = (int(size), int(modified))
    folder = PROJECT / "chains/rafael/maps_cov_512_v2"
    records = []
    for path in sorted(folder.glob("ShearMaps*.npy")):
        match = re.fullmatch(
            r"ShearMaps_DESY3_run_(\d+)_ncut(\d+)_nreal(\d+)\.npy", path.name,
        )
        if match is not None:
            run, cut, draw = map(int, match.groups())
            records.append((run, cut, draw, path))
    records.sort()
    if not args.all:
        selected_runs = sorted({row[0] for row in records})[:args.max_runs]
        records = [row for row in records
                   if row[0] in selected_runs and row[1] == 0 and row[2] < 4]
    indices = np.load(file=folder / "indices_512.npy", allow_pickle=False)
    if indices.ndim != 1 or np.unique(indices).size != indices.size:
        raise ValueError("Pixel IDs must be a one-dimensional unique list.")
    start = time.monotonic()
    with threadpool_limits(limits=1, user_api="blas"):
        first, second, geometry = aperture_weights(indices=indices)
        output.mkdir(parents=True, exist_ok=True)
        if args.resume:
            saved_geometry = np.loadtxt(fname=output / "centers.txt")
            np.testing.assert_allclose(saved_geometry, geometry, rtol=1.e-14)
        np.savetxt(fname=output / "centers.txt", X=geometry,
                   header="pixel_id RA_deg Dec_deg retained_Q30 retained_Q60")
        labels = []
        for order in (2, 3):
            for source in range(4):
                for radius in (30, 60):
                    labels.append(f"moment{order}_bin{source+1}_R{radius}arcmin")
        accepted = 0
        deferred = 0
        mode = "a" if args.resume else "w"
        with (output / "moments.txt").open(mode) as moments, \
             (output / "primary_apertures.txt").open(mode) as primary, \
             (output / "inputs.txt").open(mode) as inputs, \
             (output / "deferred.txt").open("w") as skipped:
            moments.write("# run cut draw " + " ".join(labels) + "\n")
            primary.write("# run followed by [source(4),radius(2),center(64)]\n")
            inputs.write("# filename bytes mtime_ns; no producer checksum available\n")
            for run, cut, draw, path in records:
                before = path.stat()
                if path.name in previous:
                    if previous[path.name] != (before.st_size, before.st_mtime_ns):
                        raise ValueError(f"Previously measured file changed: {path.name}.")
                    continue
                if time.time() - before.st_mtime < 120.:
                    skipped.write(path.name + " recently modified\n")
                    deferred += 1
                    continue
                try:
                    shear = np.load(file=path, mmap_mode="r", allow_pickle=False)
                except (OSError, ValueError) as error:
                    skipped.write(path.name + " " + str(error) + "\n")
                    deferred += 1
                    continue
                expected_bytes = shear.offset + shear.size * shear.dtype.itemsize
                if (shear.shape != (4, 2, len(indices)) or shear.dtype != np.float64
                        or expected_bytes != before.st_size):
                    skipped.write(path.name + " unexpected shape/type/payload length\n")
                    deferred += 1
                    del shear
                    continue
                # The sparse product performs the same angular integral in every
                # file. All four source bins are measured with identical centers.
                if not np.all(np.isfinite(shear)):
                    raise ValueError(f"Nonfinite shear in {path.name}; inspect input.")
                aperture = first @ shear[:, 0, :].T
                aperture += second @ shear[:, 1, :].T
                aperture = aperture.T.reshape(4, 2, 64)
                second_moment = np.mean(a=aperture**2, axis=2).ravel()
                third_moment = np.mean(a=aperture**3, axis=2).ravel()
                vector = np.concatenate((second_moment, third_moment))
                after = path.stat()
                del shear
                if (after.st_size != before.st_size
                        or after.st_mtime_ns != before.st_mtime_ns):
                    skipped.write(path.name + " changed during measurement\n")
                    deferred += 1
                    continue
                text = " ".join(f"{value:.17e}" for value in vector)
                moments.write(f"{run} {cut} {draw} " + text + "\n")
                if cut == 0 and draw == 0:
                    text = " ".join(f"{value:.17e}" for value in aperture.ravel())
                    primary.write(f"{run} " + text + "\n")
                inputs.write(f"{path.name} {before.st_size} {before.st_mtime_ns}\n")
                accepted += 1
                if accepted % 100 == 0:
                    moments.flush()
                    print(f"Measured {accepted}/{len(records)} files", flush=True)
    elapsed = time.monotonic() - start
    record = (
        f"Selected files: {len(records)}\nMeasured files: {accepted + len(previous)}\n"
        f"Newly measured in this pass: {accepted}\n"
        f"Deferred files: {deferred}\nElapsed seconds (this pass): {elapsed:.3f}\n"
        "OpenMP limit: 4\nBLAS workers: 1\n"
        "Pixel convention: RING/equatorial, geometry inference\n"
        "Shear basis: g1 east, g2 +45deg toward north, UNCONFIRMED\n"
        "Parent IDs: run labels, independence UNCONFIRMED\n"
        "Q radii: 30,60 arcmin; truncated at 4R; 64 fixed centers\n"
        "Rows: 8 raw squared moments then 8 raw cubed moments\n"
        "No noise self-product subtraction; no catalogue weights\n"
        "These are diagnostic moments, not the published xi/Map3 estimator.\n"
    )
    (output / "measurement.txt").write_text(record)
    print(record, flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
