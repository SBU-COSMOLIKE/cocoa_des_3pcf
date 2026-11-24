"""Survey adapter of the des_3pcf analytic covariance.

An adapter is the small project-specific layer between a driver program
and the compiled interface: it states this survey's numbers and the
numerical settings of the covariance, checks that the requested fiducial
point is one the covariance can describe, and forwards the calculation to
the C kernels compiled into cosmolike_des_3pcf_interface (sources under
cosmolike_core/cosmolike/3pcf/). The command-line program
compute_covariance.py in this folder uses its functions:

    configuration()  the survey numbers and covariance settings
    check_point()    refuse a fiducial point the covariance cannot describe
    check_theory()   refuse a halo-variance route the driver does not run
    compute()        the six covariance terms, their sum and the model mean
    save()           write one .npz archive with a minimal record

The covariance matrix C of the data vector gives the expected scatter of
each measured statistic and the correlations between entries. Here the
data vector is the joint DES Y3 shear vector of the likelihood
des_3pcf.joint_shear: the two-point correlation functions xi_+ and xi_-
(400 rows) and the third-order aperture-mass moment Map3 (80 rows). The
C routine returns C as six terms: the Gaussian term (cosmic variance and
shape noise), the squared-bispectrum term (BB), the power-spectrum times
one-halo trispectrum term (PT), the power times bispectrum term (PB, only
between xi and Map3 rows), the connected one-halo term and the
halo-abundance response (the response to density modes larger than the
survey, which is not a complete super-sample covariance).

The physical model of the matter field and the source redshift
distributions are not rebuilt here: compute() reads them from a Cobaya
likelihood that has already been evaluated at the fiducial point, so the
covariance and the likelihood mean share the same CAMB power, the same
BiHalofit bispectrum table and the same shifted n(z). The covariance has
three limits, described in covariance/README.md: it is the covariance of
the full-filter aperture estimator, its connected terms describe
gravitational lensing only (intrinsic alignments do not enter), and its
halo model treats massive neutrinos as part of one clustering fluid.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


# The six terms, in the order the C routine physical_covariance returns
# them along the first axis of its terms[6, nrow, nrow] array.
TERM_NAMES = (
    "gaussian",
    "bispectrum_squared",
    "power_trispectrum_one_halo",
    "power_bispectrum",
    "connected_one_halo",
    "halo_abundance_response",
)

# The two halo-model treatments of massive neutrinos that the README
# names. "massless" refuses a positive neutrino mass; the other one
# accepts it, with the massive-neutrino power in one clustering fluid.
NEUTRINO_MODELS = (
    "massless",
    "single_fluid_effective_growth",
)


def configuration(neutrino_model="massless"):
    """Return the survey numbers and numerical settings of the covariance.

    Every value the calculation uses is written out here, so an archive
    that records this mapping records exactly what ran. The mapping holds
    only plain numbers, strings, lists and dictionaries, which keeps it
    readable as JSON. The survey numbers are those of DES Y3; the
    physical and grid settings are those of the retired command-line
    example (y3_full_filter_example.yaml), kept unchanged here.

    Arguments:
        neutrino_model = "massless" (refuses a positive neutrino mass) or
            "single_fluid_effective_growth" (accepts it, treating massive
            neutrinos as part of one clustering matter fluid).
    Returns:
        dict of resolved settings; the comments below give units.
    Raises:
        ValueError for an unknown neutrino_model.
    """
    if neutrino_model not in NEUTRINO_MODELS:
        raise ValueError(
            f"neutrino_model = {neutrino_model!r} is not supported; "
            f"choose one of {list(NEUTRINO_MODELS)}")

    settings = {
        # --------------------------------------------------------------
        # Survey numbers. area_deg2 is the DES Y3 footprint in square
        # degrees, the value projects/des_y3 uses. The source densities
        # (effective galaxies per square arcminute) and sigma_e (the rms
        # of one ellipticity component) are the NGAL_i and SIG_E_i
        # headers of the nz_source extension of the DES Y3 release file
        # 2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate.v2.fits (in
        # projects/des_y3/data/); the DES Y3 files in this project's
        # data/ do not carry those headers.
        # --------------------------------------------------------------
        "area_deg2": 4143.0,
        "source_density_arcmin2": [
            1.475584985490327,
            1.479383426887689,
            1.483671693529899,
            1.461247850098986,
        ],
        "sigma_e_component": [
            0.2435002682964671,
            0.2621945854120855,
            0.2588927276307921,
            0.3096827008528927,
        ],

        # --------------------------------------------------------------
        # Model limits of the covariance (covariance/README.md). The C
        # routine computes only the full-filter estimator (physical
        # estimator = 0) and gravitational lensing (physical ia_model =
        # 0); the two strings record that in the archive.
        # variance_source names the halo mass-variance route: "local",
        # the variance integrated by the C routine itself, is the only
        # route this driver runs.
        # --------------------------------------------------------------
        "aperture_estimator": "full_filter",
        "ia_covariance": "gravitational_only",
        "neutrino_model": neutrino_model,
        "variance_source": "local",

        # --------------------------------------------------------------
        # Physical settings passed to the C routine (field meanings in
        # cosmolike/3pcf/covariance_physics_3pcf.h). area is filled in
        # below, in steradians.
        # --------------------------------------------------------------
        "physical": {
            # 1 = a circular footprint with the survey area.
            "center_window": 1,
            # 1 = large-area limit of the short-mode integrals.
            "large_area": 1,
            # 1 = closed-form white shape noise of each xi annulus (the
            # README formula); needs large_area = 1 and uniform_area.
            "noise_mode": 1,
            # 0 = full-filter aperture estimator, the only one coded.
            "estimator": 0,
            # 0 = pure gravitational lensing, the only one coded.
            "ia_model": 0,
            # Sheth-Tormen halo mass function: a, p and normalization.
            "st_a": 0.707,
            "st_p": 0.3,
            "st_norm": 0.3222,
            # Spherical-collapse threshold; collapse_model = 0 keeps it
            # constant in redshift.
            "delta_c": 1.686,
            "collapse_model": 0,
            # Duffy et al. (2008) M200m concentration
            # c = a (M/pivot)^b (1+z)^c, pivot in Msun/h.
            "concentration_a": 10.14,
            "concentration_b": -0.081,
            "concentration_c": -1.01,
            "concentration_pivot": 2.0e12,
            # Fine linear-read intervals per halo-profile node (the C
            # default, written out so the record is complete).
            "profile_upsample": 8,
            # 0 = no BiHalofit baryon transfer in the covariance.
            "baryon_strength": 0.0,
        },

        # --------------------------------------------------------------
        # Quadrature rules (lower, upper, count, logarithmic spacing).
        # ell: Fourier modes of the short-mode integrals. window_ell: the
        # low multipoles that resolve the survey window. phi: relative
        # angle of two Fourier vectors, 0 to 2 pi. radius: halo-offset
        # and center-separation angle, radians. k: wavenumber of the halo
        # mass variance, h/Mpc. mass: halo mass M200m, Msun/h.
        # profile_radius: nodes (not a rule) of the halo shear profile,
        # radians, with one extra node at zero.
        # --------------------------------------------------------------
        "grids": {
            "ell": {
                "lower": 1.0,
                "upper": 30000.0,
                "count": 64,
                "logarithmic": True,
            },
            "window_ell": {
                "lower": 0.01,
                "upper": 1000.0,
                "count": 64,
                "logarithmic": True,
            },
            "phi": {
                "lower": 0.0,
                "upper": 2.0 * np.pi,
                "count": 32,
                "logarithmic": False,
            },
            "radius": {
                "lower": 1.0e-7,
                "upper": 0.25,
                "count": 128,
                "logarithmic": True,
            },
            "k": {
                "lower": 1.0e-5,
                "upper": 1000.0,
                "count": 256,
                "logarithmic": True,
            },
            "mass": {
                "lower": 1.0e8,
                "upper": 1.0e16,
                "count": 48,
                "logarithmic": True,
            },
            "profile_radius": {
                "lower": 1.0e-7,
                "upper": 0.5,
                "count": 512,
                "logarithmic": True,
                "include_zero": True,
            },
        },

        # Separation nodes per xi bin, weighted by the area of the annulus
        # ("uniform_area"), as the likelihood weights its xi bins.
        "theta_count": 16,
        "pair_weight": "uniform_area",
    }

    # The C routine reads the area in steradians:
    # 1 square degree = (pi/180)^2 sr.
    degree = np.pi / 180.0
    settings["physical"]["area"] = settings["area_deg2"] * degree * degree
    return settings


def check_point(settings, parameters, nbin):
    """Refuse a fiducial point that the covariance cannot describe.

    The lensing weights of the C routine carry no multiplicative shear
    calibration, so the fiducial m_i must be zero (a Fisher calculation
    may still vary m_i in the mean around that fixed covariance). Its halo
    model treats the matter field as one clustering fluid, so a positive
    neutrino mass needs the explicit single_fluid_effective_growth choice.
    The check reads the input parameters Cobaya resolves from the yaml
    (fixed values, the evaluate override and value expressions), so it
    runs before CAMB or the likelihood starts.

    Arguments:
        settings = mapping returned by configuration().
        parameters = dict {name: value} of the resolved input parameters.
        nbin = number of source bins; DES_M1 to DES_M<nbin> are checked.
    Returns:
        nothing when the point is accepted.
    Raises:
        ValueError naming the parameter and its value: a nonzero m_i, a
        missing mnu, or a positive mnu with neutrino_model = "massless".
    """
    # A calibration absent from the yaml reads as zero in the likelihood.
    for index in range(nbin):
        name = f"DES_M{index + 1}"
        value = parameters.get(name, 0.0)
        if value != 0.0:
            raise ValueError(
                f"{name} = {value} at the fiducial point; the analytic "
                f"covariance requires every m_i = 0. Set {name}: 0 in "
                f"sampler.evaluate.override (or as a fixed value).")

    # CAMB would apply its own default mass when mnu is absent, so the
    # neutrino choice is checked only against an explicit value.
    if "mnu" not in parameters:
        raise ValueError(
            "mnu is missing from params; state the neutrino mass (eV) "
            "explicitly so the covariance's neutrino treatment can be checked.")
    mnu = parameters["mnu"]
    if mnu > 0.0 and settings["neutrino_model"] == "massless":
        raise ValueError(
            f"mnu = {mnu} eV with covariance.neutrino_model = massless; a "
            f"positive mass needs neutrino_model: "
            f"single_fluid_effective_growth (one clustering matter fluid).")


def check_theory(settings, theory):
    """Refuse a halo-variance route other than the local one.

    Under variance_source "local" (the bihalofit block's default) the C
    covariance integrates the halo mass variance itself on its k rule.
    The shared route of the bihalofit block would need a precomputed
    variance table handed to the covariance, which this driver does not
    prepare. The bihalofit block refuses evolving variance growth under
    the local route, so the factorized growth that
    single_fluid_effective_growth requires holds whenever this check
    passes.

    Arguments:
        settings = mapping returned by configuration().
        theory = the theory components of the Cobaya model, a mapping
            {name: component} (model.theory).
    Returns:
        nothing when the route is accepted.
    Raises:
        ValueError when no bihalofit block exists or its variance_source
        is not "local".
    """
    if "bihalofit" not in theory:
        raise ValueError(
            f"theory blocks {list(theory)} contain no bihalofit block; the "
            f"covariance needs the BiHalofit matter bispectrum.")
    variance_source = theory["bihalofit"].variance_source
    if variance_source != settings["variance_source"]:
        raise ValueError(
            f"bihalofit variance_source = {variance_source!r}; the covariance "
            f"driver supports only 'local'. Remove variance_source from the "
            f"bihalofit theory block.")


def profile_nodes(grid):
    """Return the halo-profile radius nodes, radians, starting at zero.

    The positive nodes are uniform in log10(radius) between lower and
    upper; one extra node at zero goes in front (include_zero), because
    the C routine requires a profile grid that starts at zero.

    Arguments:
        grid = the "profile_radius" mapping of configuration()["grids"].
    Returns:
        float64 array [count + 1], increasing.
    """
    log10_nodes = np.linspace(
        start=np.log10(grid["lower"]),
        stop=np.log10(grid["upper"]),
        num=grid["count"],
    )
    positive = 10.0**log10_nodes
    return np.concatenate((np.zeros(1), positive))


def compute(interface, likelihood, settings, indices=None):
    """Return the six covariance terms, their sum and the model mean.

    The rows (kind, source bins, angles) come from likelihood.rows and the
    matter and source model from likelihood.model, both set when Cobaya
    evaluated the likelihood at the fiducial point. The white shape-noise
    power N_i = sigma_e,i^2 / n_i (n_i per steradian) comes from
    shape_noise; covariance_inputs turns the grid settings into fixed
    quadrature nodes and weights (refined by the likelihood's three-point
    accuracy preset, the same one its mean uses), and
    production.physical_covariance runs the C terms.

    Arguments:
        interface = the compiled cosmolike_des_3pcf_interface module, built
            with covariance support.
        likelihood = the des_3pcf.joint_shear likelihood object after one
            evaluation (model.likelihood[name] of a Cobaya model).
        settings = mapping returned by configuration().
        indices = None for every row, or an integer array selecting rows
            of likelihood.rows (in that order).
    Returns:
        dict with "covariance" [nrow, nrow] (sum of the terms), "terms"
        [6, nrow, nrow], "term_names" (TERM_NAMES), "mean" [nrow] (the
        gravitational, noise-free mean of the covariance model) and
        "labels" [nrow] (row labels).
    Raises:
        RuntimeError when the interface was built without covariance.
        ValueError for a density count that differs from the source bins;
        the bindings raise ValueError for malformed rows or grids.
    """
    # ------------------------------------------------------------------
    # Build check: the covariance-off build has no shape_noise binding.
    # ------------------------------------------------------------------
    if not hasattr(interface, "shape_noise"):
        raise RuntimeError(
            f"{interface.__file__} was built without covariance support "
            f"(IGNORE_COSMOLIKE_DES_3PCF_COVARIANCE set); unset it and "
            f"recompile the project interface.")

    # ------------------------------------------------------------------
    # Rows of the evaluated likelihood, optionally a subset.
    # ------------------------------------------------------------------
    rows = likelihood.rows
    if indices is not None:
        rows = rows.select(indices=np.asarray(indices, dtype=np.int64))

    # ------------------------------------------------------------------
    # Estimator check. The likelihood builds its Map3 mean with the
    # finite-SAS estimator (measured side bins), while the C covariance
    # codes only the full-filter estimator. The two estimators fluctuate
    # differently, so a full-filter covariance does not describe the
    # Map3 rows the likelihood scores; it is refused until that question
    # is settled. The xi rows use the same estimator in both and remain
    # available (compute_covariance.py --xi-only).
    # ------------------------------------------------------------------
    map3_count = int(np.count_nonzero(rows.kind == "map3"))
    if 0 != map3_count:
        raise ValueError(
            f"{map3_count} Map3 rows requested; the likelihood's Map3 mean "
            f"uses the finite-SAS estimator, but the C covariance codes only "
            f"the full-filter estimator, so their covariance is refused. "
            f"Select the xi rows (compute_covariance.py --xi-only).")

    # ------------------------------------------------------------------
    # Shape noise: a diagonal [nbin, nbin] matrix of N_i (steradians).
    # ------------------------------------------------------------------
    density = np.asarray(settings["source_density_arcmin2"], dtype=np.float64)
    sigma = np.asarray(settings["sigma_e_component"], dtype=np.float64)
    if density.size != likelihood.source_ntomo:
        raise ValueError(
            f"{density.size} source densities for "
            f"{likelihood.source_ntomo} source bins of the likelihood; "
            f"give one density and one sigma_e per bin.")
    noise = interface.shape_noise(number_density=density, sigma_component=sigma)

    # ------------------------------------------------------------------
    # Quadrature nodes and row tables. dict(...) copies the grid mapping,
    # so the node array below does not replace the recorded settings.
    # ------------------------------------------------------------------
    grids = dict(settings["grids"])
    grids["profile_radius"] = profile_nodes(grid=settings["grids"]["profile_radius"])
    inputs = interface.covariance_inputs(
        kind=rows.kind.tolist(),
        bins=rows.bins,
        angles=rows.angles,
        noise=noise,
        grids=grids,
        physical=settings["physical"],
        theta_count=settings["theta_count"],
        pair_weight=settings["pair_weight"],
        accuracy=likelihood.accuracy,
    )

    # ------------------------------------------------------------------
    # The C terms: mean has shape [nrow], terms [6, nrow, nrow].
    # ------------------------------------------------------------------
    mean, terms = interface.production.physical_covariance(
        model=likelihood.model,
        inputs=inputs,
    )
    covariance = np.sum(terms, axis=0)
    return {
        "covariance": covariance,
        "terms": terms,
        "term_names": TERM_NAMES,
        "mean": mean,
        "labels": rows.labels,
    }


def file_sha256(path):
    """Return the SHA-256 hex digest of one file's bytes.

    Arguments:
        path = file to read.
    Returns:
        str of 64 hexadecimal characters.
    """
    digest = hashlib.sha256()
    # The with-statement closes the file at the end of the block; reading
    # 1 MiB at a time bounds the memory used for large FITS files.
    with open(path, "rb") as stream:
        block = stream.read(1 << 20)
        while block:
            digest.update(block)
            block = stream.read(1 << 20)
    return digest.hexdigest()


def save(path, result, settings, input_files, save_terms, overwrite):
    """Write the covariance and a minimal record to one .npz archive.

    The record holds what is needed to reproduce the matrix and nothing
    else: the resolved settings, the SHA-256 digest of each input file
    (named by its path relative to the project folder) and the producing
    script. It is stored as one JSON string under "metadata"; the archive
    reads back with numpy.load(path, allow_pickle=False).

    Arguments:
        path = output .npz file.
        result = mapping returned by compute().
        settings = mapping returned by configuration().
        input_files = dict {name relative to the project: Path}.
        save_terms = True also stores terms [6, nrow, nrow] and term_names.
        overwrite = True permits replacing an existing file.
    Returns:
        the Path written.
    Raises:
        FileExistsError when path exists and overwrite is False.
    Side effects:
        writes path.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it.")

    # sorted(...) fixes the order of the digests, so two runs on the same
    # inputs write the same record.
    digests = {}
    for name in sorted(input_files):
        digests[name] = file_sha256(path=input_files[name])
    record = {
        "settings": settings,
        "input_sha256": digests,
        "script": "covariance/compute_covariance.py",
    }

    arrays = {
        "covariance": result["covariance"],
        "model_mean": result["mean"],
        "row_labels": np.asarray(result["labels"], dtype=str),
        "metadata": np.asarray(json.dumps(record, sort_keys=True)),
    }
    if save_terms:
        arrays["terms"] = result["terms"]
        arrays["term_names"] = np.asarray(result["term_names"], dtype=str)
    # **arrays passes each entry as one named array of the archive.
    np.savez(path, **arrays)
    return path
