"""Prepare reproducible three-point workloads without running on import.

The measured calculation is the entire eighty-row, four-source-bin Y3
finite-SAS aperture vector. Every point changes physical cosmology and
nuisance parameters. CAMB and source shifting produce saved inputs in a
separate preparation step. Each evaluation then constructs a fresh C
PhysicsModel and computes all twenty source triplets; only the fixed
transform geometry survives between calls. No nonce or outer mean cache
is used to manufacture or suppress work.

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
    return driver, config, location, digest


def _baseline(config):
    """Keep physical defaults and fixed neutrino/IA conventions in each point."""
    parameters = dict(config["ia"])
    parameters.update(config["parameters"])
    parameters.setdefault("mnu", 0.0)
    parameters.setdefault("num_massive_neutrinos", 1)
    parameters.setdefault("nnu", 3.046)
    for name in ("a1", "eta1", "a2", "eta2", "bta"):
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
    baseline = _baseline(config=config)
    bounds = _draw_bounds(config=config)
    points = []
    warmup = None
    for index in range(int(npoints) + 1):
        point_seed = int(seed) + index
        random = np.random.Generator(np.random.PCG64(point_seed))
        parameters = dict(baseline)
        for name, interval in bounds.items():
            parameters[name] = float(random.uniform(low=interval[0], high=interval[1]))
        record = {
            "index": index if index < npoints else -1,
            "seed": point_seed,
            "configuration_sha256": digest,
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
        "warmup_point": warmup,
        "points": points,
        "draw_settings": {
            "generator": "NumPy PCG64",
            "distribution": "independent_uniform_within_recorded_bounds",
            "seed": int(seed),
            "point_seed_rule": "seed+index; warmup uses seed+npoints",
            "parameter_order": list(bounds),
            "bounds": bounds,
            "fixed_parameters": fixed,
            "thread_counts": [1, 4, 8],
            "blas_threads": 1,
            "interpretation": "Computational test points, not a posterior or prior.",
        },
    }


def _point_parameters(point, config, digest):
    """Reject a stale point or a changed fixed convention before physical work."""
    if not isinstance(point, dict) or point.get("configuration_sha256") != digest:
        raise ValueError("Point belongs to a different configuration digest.")
    baseline = _baseline(config=config)
    parameters = point.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != set(baseline):
        raise ValueError("Point must include exactly the complete parameter dictionary.")
    for name, value in parameters.items():
        if isinstance(value, bool) or not np.isfinite(value):
            raise ValueError(f"Point parameter {name} must be a finite number.")
    bounds = _draw_bounds(config=config)
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


def prepare_threepcf_tables(config_path, point, output_path):
    """Explicitly run CAMB once and save this point's shifted source tables.

    This is the expensive preparation entry. It is never called by
    preflight or point-bank creation. Existing files are preserved.
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
    tables, metadata = camb_tables(
        parameters=cosmology, k=driver.grid_values(settings=settings["k"]),
        z=redshift, sources=sources,
        los_z=driver.grid_values(settings=settings["los_z"]),
        halofit_version=settings["halofit_version"],
    )
    metadata["scaling_format"] = FORMAT
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
            accuracy=self._config["accuracy"],
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
