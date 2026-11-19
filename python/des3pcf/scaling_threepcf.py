"""Prepare reproducible three-point workloads without running on import.

The measured calculation is the entire eighty-row, four-source-bin Y3
finite-SAS aperture vector. Every point changes physical cosmology and
nuisance parameters. CAMB and source shifting produce saved inputs in a
separate preparation step. Each evaluation then constructs a fresh C
PhysicsModel and computes all twenty source triplets; only the fixed
transform geometry survives between calls. No nonce or outer mean cache
is used to manufacture or suppress work.

The two benchmark YAML files declare NLA or TATT in their notes. Both draw
the same random sequence. NLA then fixes its inactive quadratic-alignment
and velocity-shear parameters to zero, preserving every shared cosmology,
linear-alignment, calibration and source-redshift draw.

This module contains no timers. Its preflight and point-bank functions
read inputs and draw small parameter arrays; neither imports the compiled
interface nor calls CAMB. Preparation and initialization are explicit
actions for the controller to invoke when the machine is available.
"""

import hashlib
import importlib
from io import BytesIO
from itertools import combinations_with_replacement
from pathlib import Path
import sys

import numpy as np

from .accuracy import accuracy_settings
from .inputs import camb_tables, load_tables, save_tables, shifted_sources
from .inputs import camb_accuracy_settings
from .inputs import _camb_power_limits
from .mean import aligned_fft_grid, physics_model
from .ownership import immutable_copy
from .pipeline import COSMOLOGY_NAMES, IA_NAMES
from .survey import ARCMIN_TO_RADIAN, read_source_distributions, read_survey


PROJECT = Path(__file__).resolve().parents[2]
FORMAT = "cosmolike-threepcf-scaling-v1"
TIMING_SOURCE_POLICY = (
    "Timing only: set negative source-density samples to zero and "
    "renormalize each bin."
)
# These literal notes extend only this benchmark's interpretation of the
# existing YAML schema; the survey CLI does not infer a different IA model.
TIMING_IA_DECLARATIONS = {
    "NLA": "Timing IA model: NLA.",
    "TATT": "Timing IA model: TATT.",
}
NLA_INACTIVE = (
    "a2",
    "eta2",
    "bta",
    "ct",
)


def _ia_model(config):
    """Read the declared benchmark model and reject contradictory NLA values.

    Configurations without a benchmark declaration retain the full TATT draw
    used by this helper. NLA requires an explicit declaration and exactly zero
    quadratic and velocity-shear coefficients, including parameter overrides.

    Arguments:
      config = parsed survey configuration with optional benchmark notes.
    Returns:
      The string NLA or TATT.
    Raises:
      ValueError for two model declarations or a nonzero inactive NLA parameter.
    """
    declared = []
    for name, declaration in TIMING_IA_DECLARATIONS.items():
        if declaration in config.get("notes", []):
            declared.append(name)
    if len(declared) > 1:
        raise ValueError("Declare only one timing IA model: NLA or TATT.")
    model = "TATT"
    if declared:
        model = declared[0]
    if model == "NLA":
        parameters = dict(config["ia"])
        parameters.update(config["parameters"])
        for name in NLA_INACTIVE:
            if parameters.get(name, 0.0) != 0.0:
                raise ValueError(
                    f"NLA timing requires {name}=0; remove its nonzero override."
                )
    return model


def _configuration(path):
    """Reuse the CLI's scientific schema without importing its C extension."""
    directory = str(PROJECT / "scripts")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    driver = importlib.import_module(name="run_3pcf")
    config, location, digest = driver.load_configuration(path=path, command="prepare")
    if config["source"]["nbin"] != 4:
        raise ValueError("This workload requires four source bins and eighty Map3 rows.")
    if config["mean"].get("aperture_estimator") != "finite_sas":
        raise ValueError("This workload requires the declared finite-SAS estimator.")
    if config["physics"]["cosmology_mode"] != "camb":
        raise ValueError("Random cosmologies require explicit CAMB preparation.")
    if config["rows"]["mode"] != "measurement" or config["rows"]["aperture"] is None:
        raise ValueError("The scaling workload requires the labeled Y3 aperture FITS.")
    _ia_model(config=config)
    return driver, config, location, digest


def _baseline(config):
    """Keep physical defaults and fixed neutrino/IA conventions in each point."""
    parameters = dict(config["ia"])
    parameters.update(config["parameters"])
    parameters.setdefault("mnu", 0.0)
    parameters.setdefault("num_massive_neutrinos", 1)
    parameters.setdefault("nnu", 3.046)
    for name in ("a1", "eta1", "a2", "eta2", "bta", "ct"):
        parameters.setdefault(name, 0.0)
    for source in range(1, 5):
        parameters.setdefault(f"m{source}", 0.0)
        parameters.setdefault(f"dz{source}", 0.0)
    return parameters


def _draw_bounds(config):
    """Declare moderate physical draws, not a posterior or a prior model.

    The cosmological intervals avoid extreme CAMB models. Their dark-energy
    upper corner has w0+wa=-0.5, away from an early matter-like component.
    Calibration/source intervals span twice each supplied Gaussian width;
    where none is supplied the explicit fallback half-width is 0.02.
    The neutrino mass, species count and discrete IA conventions stay fixed.
    """
    bounds = {
        "omega_m": [0.24, 0.34],
        "omega_b": [0.043, 0.053],
        "h": [0.64, 0.80],
        "ns": [0.92, 1.00],
        "sigma8": [0.72, 0.86],
        "w0": [-1.2, -0.8],
        "wa": [-0.3, 0.3],
        "a1": [-1.0, 0.8],
        "eta1": [-2.0, 3.0],
        "a2": [-0.8, 1.2],
        "eta2": [-2.0, 3.0],
        "bta": [0.2, 1.2],
    }
    baseline = _baseline(config=config)
    priors = config.get("fisher", {}).get("priors", {})
    for source in range(1, 5):
        for prefix in ("m", "dz"):
            name = f"{prefix}{source}"
            half_width = 2.0 * float(priors.get(name, 0.01))
            if not np.isfinite(half_width) or half_width <= 0.0:
                raise ValueError(f"Draw width for {name} must be positive and finite.")
            bounds[name] = [baseline[name] - half_width, baseline[name] + half_width]
    return bounds


def _active_draw_bounds(config):
    """Separate physical draws from random values discarded for pure NLA.

    Sampling still consumes every TATT random value, in the same order.
    Removing inactive entries only from this returned dictionary identifies
    which parameters can change in the accepted physical point.
    """
    bounds = _draw_bounds(config=config)
    if _ia_model(config=config) == "NLA":
        for name in NLA_INACTIVE:
            bounds.pop(name, None)
    return bounds


def threepcf_point_bank(config_path, npoints=10, seed=137):
    """Return the same JSON-compatible random points for every worker count.

    Each measured point has its own PCG64 seed, seed+index. A separate
    warmup uses seed+npoints. Bounds and draw order are saved explicitly;
    there is no rejection loop or dependence on a previous random draw.
    """
    if (isinstance(npoints, bool) or int(npoints) != npoints or npoints < 1
            or isinstance(seed, bool) or int(seed) != seed or seed < 0):
        raise ValueError("Require positive integer npoints and nonnegative integer seed.")
    _, config, path, digest = _configuration(path=config_path)
    ia_model = _ia_model(config=config)
    baseline = _baseline(config=config)
    random_bounds = _draw_bounds(config=config)
    bounds = _active_draw_bounds(config=config)
    points = []
    warmup = None
    for index in range(int(npoints) + 1):
        point_seed = int(seed) + index
        random = np.random.Generator(np.random.PCG64(point_seed))
        parameters = dict(baseline)
        for name, interval in random_bounds.items():
            parameters[name] = float(random.uniform(low=interval[0], high=interval[1]))
        if ia_model == "NLA":
            for name in NLA_INACTIVE:
                parameters[name] = 0.0
        record = {
            "index": index if index < npoints else -1,
            "seed": point_seed,
            "configuration_sha256": digest,
            "ia_model": ia_model,
            "parameters": parameters,
        }
        if index < npoints:
            points.append(record)
        else:
            warmup = record
    fixed = {}
    for name, value in baseline.items():
        if name not in bounds:
            fixed[name] = value
    return {
        "format": FORMAT,
        "configuration": str(path),
        "configuration_sha256": digest,
        "ia_model": ia_model,
        "warmup_point": warmup,
        "points": points,
        "draw_settings": {
            "generator": "NumPy PCG64",
            "distribution": "independent_uniform_within_recorded_bounds",
            "seed": int(seed),
            "point_seed_rule": "seed+index; warmup uses seed+npoints",
            "parameter_order": list(bounds),
            "bounds": bounds,
            "random_draw_order": list(random_bounds),
            "random_draw_bounds": random_bounds,
            "inactive_after_draw": list(NLA_INACTIVE) if ia_model == "NLA" else [],
            "fixed_parameters": fixed,
            "thread_counts": [
                1,
                4,
                8,
            ],
            "blas_threads": 1,
            "interpretation": "Computational test points, not a posterior or prior.",
        },
    }


def _point_parameters(point, config, digest):
    """Reject a stale point or a changed fixed convention before physical work."""
    if not isinstance(point, dict) or point.get("configuration_sha256") != digest:
        raise ValueError("Point belongs to a different configuration digest.")
    if point.get("ia_model", "TATT") != _ia_model(config=config):
        raise ValueError("Point IA model differs from the declared timing model.")
    baseline = _baseline(config=config)
    parameters = point.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != set(baseline):
        raise ValueError("Point must include exactly the complete parameter dictionary.")
    for name, value in parameters.items():
        if isinstance(value, bool) or not np.isfinite(value):
            raise ValueError(f"Point parameter {name} must be a finite number.")
    bounds = _active_draw_bounds(config=config)
    for name, value in parameters.items():
        if name in bounds:
            if not bounds[name][0] <= value <= bounds[name][1]:
                raise ValueError(
                    f"Point parameter {name} is outside its declared draw bounds."
                )
        elif value != baseline[name]:
            raise ValueError(f"The scaling bank holds {name} fixed.")
    if parameters["w0"] + parameters["wa"] >= -0.01:
        raise ValueError("Require w0+wa<-0.01 for this CAMB workload.")
    return dict(parameters)


def _aperture_rows(driver, config, path):
    """Read every aperture row and verify the full twenty-triplet workload."""
    filename = driver.resolve_path(path, config["rows"]["aperture"])
    rows = read_survey(
        two_point_path=None, aperture_path=filename, nbin=4,
        assumed_angle_unit=config["rows"].get("assumed_angle_unit", "arcmin"),
    )
    expected = set(combinations_with_replacement(range(4), 3))
    actual = set()
    for triplet in rows.bins:
        actual.add(tuple(int(value) for value in triplet))
    if rows.values.size != 80 or actual != expected:
        raise ValueError("Require all eighty aperture rows and twenty source triplets.")
    for triplet in expected:
        matching = np.all(rows.bins == np.asarray(triplet)[None, :], axis=1)
        if np.count_nonzero(matching) != 4:
            raise ValueError("Every source triplet must retain its four aperture rows.")
    return rows


def _clip_timing_sources(redshift, densities):
    """Apply the declared timing treatment and retain its numerical accounting.

    Negative calibrated samples are not interpreted as a physical source
    probability here. Setting them to zero is a timing-only input choice,
    not a replacement calibration. Record their integrated magnitude and
    the normalization of each clipped curve. Caller arrays are unchanged.
    """
    z = np.asarray(redshift, dtype=np.float64)
    values = np.asarray(densities, dtype=np.float64)
    if (z.ndim != 1 or z.size < 2 or np.any(np.diff(z) <= 0.0)
            or np.any(z < 0.0) or not np.all(np.isfinite(z))
            or values.ndim != 2 or values.shape[1] != z.size
            or not np.all(np.isfinite(values))):
        raise ValueError(
            "Timing source curves require finite samples on increasing z>=0."
        )
    negative_area = np.trapz(np.maximum(-values, 0.0), x=z, axis=1)
    raw_integrals = np.trapz(values, x=z, axis=1)
    clipped = np.maximum(values, 0.0)
    positive_integrals = np.trapz(clipped, x=z, axis=1)
    if np.any(positive_integrals <= 0.0):
        raise ValueError(
            "Every timing source bin needs positive probability after clipping."
        )
    normalized = clipped / positive_integrals[:, None]
    record = {
        "policy": TIMING_SOURCE_POLICY,
        "timing_only": True,
        "negative_sample_count": np.count_nonzero(values < 0.0, axis=1).tolist(),
        "negative_area_magnitude": negative_area.tolist(),
        "raw_integrals": raw_integrals.tolist(),
        "clipped_integrals_before_normalization": positive_integrals.tolist(),
        "normalization_factors": (1.0 / positive_integrals).tolist(),
        "negative_area_fraction_of_positive": (
            negative_area / positive_integrals
        ).tolist(),
    }
    return normalized, record


def _source_input(driver, config, path):
    """Read strict sources unless the exact timing-only declaration is present."""
    source_path = driver.resolve_path(path, config["source"]["path"])
    content = source_path.read_bytes()
    if TIMING_SOURCE_POLICY in config.get("notes", []):
        from astropy.io import fits

        # Hash and parse the same byte string, even if another process
        # replaces the original file after it has been read.
        with fits.open(BytesIO(content), memmap=False) as hdus:
            table = hdus[config["source"]["extension"]].data
            required = ["Z_MID", "BIN1", "BIN2", "BIN3", "BIN4"]
            for name in required:
                if name not in table.names:
                    raise ValueError(f"Timing source FITS lacks {name}.")
            source_z = np.array(table["Z_MID"], dtype=np.float64, copy=True)
            raw = np.empty((4, source_z.size), dtype=np.float64)
            for source in range(4):
                raw[source] = table[f"BIN{source + 1}"]
        densities, record = _clip_timing_sources(redshift=source_z, densities=raw)
    else:
        source_z, densities = read_source_distributions(
            path=source_path, nbin=4, extension=config["source"]["extension"],
        )
        record = {"policy": "strict_nonnegative", "timing_only": False}
    record["raw_input_sha256"] = hashlib.sha256(content).hexdigest()
    record["raw_input_path"] = str(source_path)
    return source_z, densities, source_path, record


def _sources(driver, config, path, shifts):
    """Apply shifts on the configured grid without silently losing high-z support."""
    source_z, densities, source_path, record = _source_input(
        driver=driver, config=config, path=path,
    )
    redshift = driver.grid_values(settings=config["physics"]["z"])
    for source in range(4):
        positive = source_z[densities[source] > 0.0]
        if positive.size == 0 or positive[-1] + max(0.0, shifts[source]) > redshift[-1]:
            raise ValueError(
                f"Source {source + 1} shifted support exceeds physics.z; "
                "extend the declared table before CAMB preparation."
            )
    sampled = np.empty((4, redshift.size), dtype=np.float64)
    for source in range(4):
        sampled[source] = np.interp(
            x=redshift, xp=source_z, fp=densities[source], left=0.0, right=0.0,
        )
    integrals = np.trapz(sampled, x=redshift, axis=1)
    baseline = shifted_sources(z=redshift, sources=sampled, shifts=np.zeros(4))
    shifted = shifted_sources(z=redshift, sources=baseline, shifts=shifts)
    record["sampled_integrals_before_grid_normalization"] = integrals.tolist()
    record["applied_shifts"] = shifts.tolist()
    record["final_shifted_integrals"] = np.trapz(shifted, x=redshift, axis=1).tolist()
    return redshift, shifted, source_path, integrals, record


def preflight_threepcf(config_path):
    """Report file/input blockers and aligned-grid bounds without C or CAMB.

    Source validation includes the largest positive draw in every source
    shift, so the controller does not begin eleven costly preparations with
    a known truncation. Negative source samples are refused unless the
    exact TIMING_SOURCE_POLICY declaration authorizes the recorded
    benchmark-only treatment; no repair is inferred from a filename.
    """
    driver, config, path, digest = _configuration(path=config_path)
    blockers = []
    bounds = _draw_bounds(config=config)
    # Every random point must fit the interpolation tail, including the
    # largest allowed h. Checking only the YAML's central h would miss this.
    camb_limits = {}
    try:
        solved, extended = _camb_power_limits(
            k=driver.grid_values(settings=config["physics"]["k"]),
            h=bounds["h"][1],
            kmax_boltzmann=config["physics"].get("kmax_boltzmann", 5.0),
            extrap_kmax=config["physics"].get("extrap_kmax"),
        )
        camb_limits = {
            "kmax_boltzmann_1_per_Mpc": solved,
            "extrap_kmax_at_largest_h_1_per_Mpc": extended,
            "largest_h_checked": bounds["h"][1],
        }
    except ValueError as error:
        blockers.append(f"CAMB interpolation range: {error}")
    maximum_shifts = np.array([bounds[f"dz{i}"][1] for i in range(1, 5)])
    source_record = {}
    try:
        redshift, _, filename, integrals, processing = _sources(
            driver=driver, config=config, path=path, shifts=maximum_shifts,
        )
        source_record = {
            "path": str(filename),
            "sha256": processing["raw_input_sha256"],
            "table_zmax": float(redshift[-1]),
            "integrals_before_normalization": integrals.tolist(),
            "maximum_positive_shifts_checked": maximum_shifts.tolist(),
            "processing": processing,
        }
    except (OSError, ValueError) as error:
        blockers.append(f"Source input: {error}")
    row_labels = []
    aperture_record = {}
    try:
        rows = _aperture_rows(driver=driver, config=config, path=path)
        row_labels = rows.labels.tolist()
        filename = driver.resolve_path(path, config["rows"]["aperture"])
        aperture_record = {
            "path": str(filename),
            "sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
            "bins": rows.bins.tolist(),
            "radii_radians": rows.angles.tolist(),
        }
    except (OSError, ValueError) as error:
        blockers.append(f"Aperture input: {error}")
    settings = config["mean"]["aperture_settings"]
    edges = driver.grid_values(settings=settings["theta_edges_arcmin"])
    widths = np.diff(np.log(edges))
    if not np.allclose(widths, widths[0], rtol=1.0e-12, atol=1.0e-14):
        raise ValueError("Finite-SAS side bins must have equal logarithmic widths.")
    controls = accuracy_settings(value=config["accuracy"])
    ell, _ = aligned_fft_grid(
        theta_lower=edges[:-1] * ARCMIN_TO_RADIAN, dlnt=float(widths[0]),
        ell_min=settings["ell_min"], ell_max=settings["ell_max"],
        minimum_size=controls.count(settings["minimum_size"], stage="fft_boost"),
    )
    projection = settings.get("projection_settings", {})
    if settings.get("projection_mode", "prepared") == "prepared":
        if (projection.get("ell_min", ell[0]) > ell[0]
                or projection.get("ell_max", ell[-1]) < ell[-1]):
            blockers.append("Projection bounds do not cover the aligned FFT ell grid.")
    extension = PROJECT / "interface" / "cosmolike_des_3pcf_interface.so"
    if not extension.is_file():
        blockers.append(f"Missing compiled extension: {extension}")
    return {
        "format": FORMAT,
        "configuration": str(path),
        "configuration_sha256": digest,
        "ia_model": _ia_model(config=config),
        "camb_limits": camb_limits,
        "native_ia_work": {
            "residual_physics_skipped": _ia_model(config=config) == "NLA",
            "prepared_mode_required_for_skip": True,
            "shared_work": (
                "BiHalofit/NLA baseline, multipole preparation and all finite-SAS "
                "transforms remain. Zero residual multipole loops also remain."
            ),
        },
        "ready": not blockers,
        "blockers": blockers,
        "row_count": len(row_labels),
        "row_labels": row_labels,
        "source": source_record,
        "aperture": aperture_record,
        "extension": str(extension),
        "aligned_fft_size": int(ell.size),
        "aligned_ell_bounds": [float(ell[0]), float(ell[-1])],
        "thread_counts": [1, 4, 8],
        "blas_threads": 1,
        "numerical_work_executed": False,
    }


def _table_request(
    cosmology, halofit_version, axes, processing,
    kmax_boltzmann=5.0, extrap_kmax=None,
    camb_accuracy_boost=1.0, camb_k_per_logint=0,
):
    """Identify every input to the physical table producer before running CAMB.

    The producer file hash records its implicit CAMB settings, including flat
    geometry, PPF dark energy and amplitude calibration. Each array hash records
    exact float64 bytes and shape. IA and calibration amplitudes do not enter
    CAMB or source shifting, so they are deliberately absent from this record.

    Arguments:
      cosmology = complete CAMB cosmology dictionary.
      halofit_version = explicitly selected nonlinear-power prescription.
      axes = k, z, los_z and already shifted source arrays in physical h units.
      processing = source-file hash and declared source-treatment record.
      kmax_boltzmann = upper evolved CAMB wavenumber in 1/Mpc.
      extrap_kmax = interpolator read limit in 1/Mpc; None derives the declared
          default from this point's h and largest output k in h/Mpc.
      camb_accuracy_boost, camb_k_per_logint = independent CAMB precision
          choices, included even when their historical defaults are used.
    Returns:
      JSON-compatible physical-request dictionary for exact equality checks.
    """
    producer = importlib.import_module(name=camb_tables.__module__)
    camb_accuracy_boost, camb_k_per_logint = camb_accuracy_settings(
        accuracy_boost=camb_accuracy_boost, k_per_logint=camb_k_per_logint,
    )
    if isinstance(kmax_boltzmann, np.bool_) or isinstance(extrap_kmax, np.bool_):
        raise ValueError("CAMB limits must be physical wavenumbers, not Boolean.")
    kmax_boltzmann, resolved_extrapolation = _camb_power_limits(
        k=axes["k"], h=cosmology["h"],
        kmax_boltzmann=kmax_boltzmann, extrap_kmax=extrap_kmax,
    )
    arrays = {}
    for name, values in axes.items():
        array = np.ascontiguousarray(values, dtype=np.float64)
        arrays[name] = {
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    return {
        "producer": "CAMB",
        "producer_source_sha256": hashlib.sha256(
            Path(producer.__file__).read_bytes()
        ).hexdigest(),
        "parameters": dict(cosmology),
        "halofit_version": halofit_version,
        "camb_accuracy_boost": camb_accuracy_boost,
        "camb_k_per_logint": camb_k_per_logint,
        "kmax_boltzmann": float(kmax_boltzmann),
        "extrap_kmax": float(resolved_extrapolation),
        "extrap_kmax_requested": extrap_kmax,
        "source_fits_sha256": processing["raw_input_sha256"],
        "source_policy": processing["policy"],
        "arrays": arrays,
    }


def _reuse_tables(path, request, axes):
    """Accept an archive only when its physical inputs match this request exactly.

    Array comparisons check the actual loaded grids and shifted sources in
    addition to their recorded hashes. The original CAMB version, amplitude
    calibration and power arrays remain attached to the copied table; reusing
    a calculation does not claim it was recomputed by the current environment.

    Arguments:
      path = existing scaling NPZ archive proposed for IA-independent reuse.
      request = physical-request dictionary produced for the target point.
      axes = target k, z, los_z and shifted source arrays.
    Returns:
      Validated arrays, original metadata and the reused file's identity.
    Raises:
      ValueError if producer choices, cosmology, arrays or file bytes differ.
    """
    filename = Path(path).resolve()
    digest = hashlib.sha256(filename.read_bytes()).hexdigest()
    tables, metadata = load_tables(path=filename)
    if hashlib.sha256(filename.read_bytes()).hexdigest() != digest:
        raise ValueError("Reused table file changed during loading; retry its read.")
    if (metadata.get("scaling_format") != FORMAT
            or metadata.get("scaling_table_request") != request):
        raise ValueError(
            "Reused tables have different physical preparation inputs; "
            "prepare this point separately."
        )
    required = {
        "producer": "CAMB",
        "parameters": request["parameters"],
        "halofit_version": request["halofit_version"],
        "camb_accuracy_boost": request["camb_accuracy_boost"],
        "camb_k_per_logint": request["camb_k_per_logint"],
        "kmax_boltzmann": request["kmax_boltzmann"],
        "extrap_kmax": request["extrap_kmax"],
        "flat_background": True,
        "growth_definition": "linear sigma8(z)/sigma8(0)",
        "massive_neutrino_effective_growth": request["parameters"]["mnu"] > 0.0,
    }
    for name, expected in required.items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"Reused table metadata differs in {name}; prepare it separately."
            )
    for name, expected in axes.items():
        actual = np.ascontiguousarray(tables[name], dtype=np.float64)
        reference = np.ascontiguousarray(expected, dtype=np.float64)
        if (actual.shape != reference.shape
                or not np.array_equal(actual.view(np.uint64), reference.view(np.uint64))):
            raise ValueError(
                f"Reused table {name} differs from the target request; "
                "prepare this point separately."
            )
    record = {
        "path": str(filename),
        "sha256": digest,
        "configuration_sha256": metadata.get("configuration_sha256"),
        "ia_model": metadata.get("scaling_ia_model"),
        "verification": "exact_physical_request_and_loaded_axes",
    }
    return tables, metadata, record


def prepare_threepcf_tables(config_path, point, output_path, reuse_path=None):
    """Prepare or exactly reuse this point's CAMB and shifted-source tables.

    This is the expensive preparation entry. It is never called by
    preflight or point-bank creation. Reuse is allowed only after the cosmology,
    CAMB options and actual grids/sources match; changing IA alone then requires
    no second Boltzmann solution. Existing files are preserved.

    Arguments:
      config_path = explicitly declared NLA or TATT benchmark YAML.
      point = complete point record belonging to that configuration.
      output_path = new .npz destination, never an existing file.
      reuse_path = optional archive for the same cosmology and source shifts.
    Returns:
      Absolute path to the archive with target configuration/point metadata.
    Raises:
      ValueError for conflicting identities, source support or reuse inputs.
    """
    driver, config, path, digest = _configuration(path=config_path)
    parameters = _point_parameters(point=point, config=config, digest=digest)
    destination = Path(output_path).resolve()
    if destination.suffix != ".npz" or destination.exists():
        raise ValueError("Choose a new .npz destination for this point's tables.")
    shifts = np.array([parameters[f"dz{i}"] for i in range(1, 5)])
    redshift, sources, source_path, integrals, processing = _sources(
        driver=driver, config=config, path=path, shifts=shifts,
    )
    cosmology = {}
    for name in COSMOLOGY_NAMES:
        cosmology[name] = parameters[name]
    settings = config["physics"]
    axes = {
        "k": driver.grid_values(settings=settings["k"]),
        "z": redshift,
        "sources": sources,
        "los_z": driver.grid_values(settings=settings["los_z"]),
    }
    request = _table_request(
        cosmology=cosmology, halofit_version=settings["halofit_version"],
        axes=axes, processing=processing,
        kmax_boltzmann=settings.get("kmax_boltzmann", 5.0),
        extrap_kmax=settings.get("extrap_kmax"),
        camb_accuracy_boost=settings.get("camb_accuracy_boost", 1.0),
        camb_k_per_logint=settings.get("camb_k_per_logint", 0),
    )
    reused = None
    if reuse_path is None:
        tables, metadata = camb_tables(
            parameters=cosmology, k=axes["k"], z=axes["z"], sources=axes["sources"],
            los_z=axes["los_z"], halofit_version=settings["halofit_version"],
            kmax_boltzmann=settings.get("kmax_boltzmann", 5.0),
            extrap_kmax=settings.get("extrap_kmax"),
            camb_accuracy_boost=settings.get("camb_accuracy_boost", 1.0),
            camb_k_per_logint=settings.get("camb_k_per_logint", 0),
        )
    else:
        tables, metadata, reused = _reuse_tables(
            path=reuse_path, request=request, axes=axes,
        )
    metadata["scaling_format"] = FORMAT
    metadata["scaling_ia_model"] = _ia_model(config=config)
    metadata["scaling_table_request"] = request
    metadata["scaling_reused_from"] = reused
    metadata["scaling_point"] = point
    metadata["configuration_sha256"] = digest
    metadata["source_fits"] = str(source_path)
    metadata["source_fits_sha256"] = processing["raw_input_sha256"]
    metadata["source_integrals_before_normalization"] = integrals.tolist()
    metadata["source_shifts_applied"] = shifts.tolist()
    metadata["source_convention"] = "linear_midpoint; n(z-dz); normalized after shift"
    metadata["source_processing"] = processing
    destination.parent.mkdir(parents=True, exist_ok=True)
    return save_tables(path=destination, tables=tables, metadata=metadata)


def load_threepcf_tables(path):
    """Load and own one staged bundle outside the controller's timed interval."""
    tables, metadata = load_tables(path=path)
    if metadata.get("scaling_format") != FORMAT or "scaling_point" not in metadata:
        raise ValueError("Table file lacks the complete scaling-point provenance.")
    return {"tables": immutable_copy(value=tables),
            "metadata": immutable_copy(value=metadata)}


class ThreePointWorkload:
    """Own one finite-SAS geometry and no retained physical prediction."""

    def __init__(self, driver, config, path, digest, compiled, threads):
        from threadpoolctl import threadpool_limits

        # Resolve the flat control dictionary before freezing nested config
        # mappings. Accuracy itself is immutable and accepted by both engines.
        self._accuracy = accuracy_settings(value=config["accuracy"])
        self._config = immutable_copy(value=config)
        self._digest = digest
        self._compiled = compiled
        # Limit any BLAS library already loaded in this process. Changing
        # environment variables alone cannot resize an existing worker pool.
        self._blas_limit = threadpool_limits(limits=1, user_api="blas")
        self.set_threads(n=threads)
        self.rows = _aperture_rows(driver=driver, config=config, path=path)
        geometry = driver.prediction_for_run(
            compiled=compiled, config=config, path=path, rows=self.rows, provider=None,
        )
        self._engine = geometry.map_engine

    def set_threads(self, n):
        """Set the actual extension's OpenMP team before any geometry or mean work."""
        if isinstance(n, bool) or int(n) != n or n not in (1, 4, 8):
            raise ValueError("The declared scaling comparison uses 1, 4 or 8 workers.")
        self._compiled.set_omp_threads(nthreads=int(n))

    def evaluate(self, table_bundle, point):
        """Build fresh physical C tables and return all eighty aperture values.

        Inputs are already loaded and source shifted. The measured call
        includes lightweight identity checks, C-model preparation and the
        complete mean. It excludes CAMB, disk reads and geometry creation.
        No second source shift and no outer prediction cache is applied.
        """
        parameters = _point_parameters(
            point=point, config=self._config, digest=self._digest,
        )
        metadata = table_bundle["metadata"]
        if metadata["configuration_sha256"] != self._digest:
            raise ValueError("Tables and workload use different configurations.")
        recorded = metadata["scaling_point"]["parameters"]
        if dict(recorded) != parameters:
            raise ValueError(
                "Tables were prepared for a different complete physical point."
            )
        settings = self._config["physics"]
        model = physics_model(
            compiled=self._compiled, tables=table_bundle["tables"],
            kernel_nz=settings["kernel_nz"],
            source_readers=settings.get("source_readers", False),
            table_boost=settings.get("table_boost", 8), fit_nz=settings.get("fit_nz", 33),
            accuracy=self._accuracy,
        )
        ia = {}
        for name in IA_NAMES:
            if name in parameters:
                ia[name] = parameters[name]
        calibration = np.array([parameters[f"m{i}"] for i in range(1, 5)])
        return self._engine.evaluate(model=model, ia=ia, calibration=calibration)


def initialize_threepcf(config_path, threads, extension_path=None):
    """Import the chosen binary and prepare the fixed geometry before timing.

    Snapshots retain the canonical extension filename. A different binary
    already imported in the worker is refused; the controller uses a fresh
    process for every worker count instead of replacing Python modules.
    """
    driver, config, path, digest = _configuration(path=config_path)
    filename = PROJECT / "interface" / "cosmolike_des_3pcf_interface.so"
    if extension_path is not None:
        filename = Path(extension_path).resolve()
    if filename.name != "cosmolike_des_3pcf_interface.so" or not filename.is_file():
        raise ValueError("Supply an existing cosmolike_des_3pcf_interface.so snapshot.")
    module_name = "cosmolike_des_3pcf_interface"
    if module_name in sys.modules:
        current = Path(sys.modules[module_name].__file__).resolve()
        if current != filename.resolve():
            raise ValueError("A different binary is loaded; use a fresh worker process.")
    sys.path.insert(0, str(filename.parent))
    compiled = importlib.import_module(name=module_name)
    if Path(compiled.__file__).resolve() != filename.resolve():
        raise ValueError("Imported binary does not match the requested snapshot.")
    return ThreePointWorkload(
        driver=driver, config=config, path=path, digest=digest,
        compiled=compiled, threads=threads,
    )
