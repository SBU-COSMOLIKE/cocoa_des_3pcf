r"""Measure controlled Map3 covariance calculations in separate CLI processes.

Example, from the project directory:
  python -B tests/studies/covariance_scaling.py run \
    --extension /path/to/cosmolike_des_3pcf_interface.so \
    --tables chains/openmp_scaling_ia_extrapolated/tables/threepcf_nla \
    --output chains/covariance_pilot --threads 1 --points 1

The table bank supplies one warmup and changed physical cosmologies. No CAMB
calculation runs here. Four source bins and four equal aperture radii give
80 Map3 rows. The area and noise below are declared test choices, not DES
catalogue measurements. The fixed quadratures are performance-study inputs,
not demonstrated survey precision. All intrinsic-alignment coefficients and
multiplicative shear offsets are excluded from this gravitational covariance.
The saved shifts of the source redshift distributions remain in their tables.

The physical case times the public C covariance call, including its internal
geometry, projected spectra and halo preparation. Model construction, Python
quadrature construction, file reads and result saving remain outside timing.
The Gaussian case times filtered-pair construction and all fifteen Wick
pairings for raw cubic maps with a finite circular center window. Its supplied
angular spectra are prepared before timing. The physical large-area branch
keeps only six Gaussian pairings, so these cases are different calculations.

The controller starts workers sequentially. Each thread count has a fresh
process; it checks all saved output bits against the first thread count and,
when requested, a prior study. A pilot uses --threads 1 --points 1. Increase
individual --nell, --nradius or other grid counts for a larger workload.
"""


import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
FORMAT = "des3pcf-covariance-scaling-v1"
EXTENSION_NAME = "cosmolike_des_3pcf_interface"
BLAS_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GOTO_NUM_THREADS",
)
TERM_NAMES = (
    "gaussian",
    "bispectrum_squared",
    "power_trispectrum_one_halo",
    "power_bispectrum",
    "connected_one_halo",
    "halo_abundance_response",
)


import cosmolike_des_3pcf_interface as ci


def save_json(path, value):
    """Write resolved settings or measurements as indented JSON.

    Arguments:
      path = destination filename; an existing record is replaced.
      value = settings or measurements containing JSON-compatible values.
    Returns:
      None; writes one file with an ending newline.
    Raises:
      ValueError for nonfinite numbers that JSON cannot represent here.
      OSError when the destination cannot be written.
    """
    contents = json.dumps(
        obj=value, indent=2, sort_keys=True, allow_nan=False,
    )
    destination = Path(path)
    destination.write_text(data=contents + "\n")


def identity(path):
    """Read a file in blocks and record its SHA-256 content identifier.

    Arguments:
      path = existing filename, relative or absolute.
    Returns:
      Dictionary containing its absolute path, byte count and SHA-256 string.
    Raises:
      OSError if the file does not exist or cannot be read.
    """
    supplied_path = Path(path)
    filename = supplied_path.resolve(strict=True)
    digest = hashlib.sha256()
    with filename.open(mode="rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return {
        "path": str(filename),
        "bytes": filename.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def peak_memory_bytes():
    """Return the process resident-memory high-water mark, including setup.

    macOS reports ru_maxrss in bytes; Linux reports kibibytes. This number
    is cumulative, so it cannot isolate memory allocated inside one call.

    Returns:
      Integer bytes at the largest recorded resident-memory use.
    Raises:
      RuntimeError on a platform whose ru_maxrss units are not specified.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(peak)
    if sys.platform.startswith("linux"):
        return int(peak * 1024)
    raise RuntimeError("Peak-memory units are defined here for macOS/Linux.")


def resolved_settings(args):
    """Resolve the numerical choices before any output directory is written.

    These counts define a coarse workload, not an integration-error target.
    Four aperture radii match the Y3 example's row structure. The area and
    noise use controlled-test scales. Halo coefficients follow the physical
    covariance fixture's ST mass function and Duffy concentration convention.

    Arguments:
      args = parsed CLI options, including the case and optional grid counts.
    Returns:
      JSON-compatible dictionary containing every applied numerical default.
    Raises:
      ValueError for fewer than four LOS nodes or two other quadrature nodes.
    """
    counts = {
        "nell": 16,
        "nphi": 16,
        "nradius": 64,
        "nmass": 8,
        "nk": 192,
        "nwindow": 32,
        "nlos": 9,
    }
    if args.case == "gaussian":
        counts["nell"] = 256
        counts["nradius"] = 512
    requested_counts = {
        "nell": args.nell,
        "nphi": args.nphi,
        "nradius": args.nradius,
        "nmass": args.nmass,
        "nk": args.nk,
        "nwindow": args.nwindow,
        "nlos": args.nlos,
    }
    for name in counts:
        requested = requested_counts[name]
        if requested is not None:
            counts[name] = requested
        minimum = 4 if name == "nlos" else 2
        if counts[name] < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")
    return {
        "counts": counts,
        "aperture_radii_arcmin": [
            7.0,
            14.0,
            25.0,
            40.0,
        ],
        "ell_bounds": [1.0, 30000.0],
        "phi_bounds": [0.0, 6.283185307179586],
        "radius_bounds": [1.0e-7, 0.25],
        "mass_bounds": [1.0e8, 1.0e16],
        "k_bounds": [1.0e-5, 1000.0],
        "window_bounds": [0.01, 1000.0],
        "los_bounds": [0.005, 3.0],
        "profile_radius_bounds": [1.0e-7, 0.5],
        "profile_positive_count": 96,
        "profile_upsample": 8,
        "kernel_nz": 101,
        "table_boost": 8,
        "fit_nz": 33,
        "source_readers": False,
        "area_steradian": 0.2,
        "diagonal_noise_steradian": 2.0e-9,
        "large_area": int(args.case == "physical"),
        "center_window": 1,
        "noise_mode": 0,
        "estimator": 0,
        "ia_model": 0,
        "theta_count": 2,
        "pair_weight": "uniform_area",
        "st_a": 0.707,
        "st_p": 0.3,
        "st_norm": 0.3222,
        "delta_c": 1.686,
        "collapse_model": 0,
        "concentration_a": 10.14,
        "concentration_b": -0.081,
        "concentration_c": -1.01,
        "concentration_pivot": 2.0e12,
        "baryon_strength": 0.0,
        "neutrino_model": "single_fluid_effective_growth",
        "area_and_noise_origin": "declared controlled test values",
        "ia_and_shear_calibration": "zero; saved nuisance draws not applied",
    }


def table_records(directory, points):
    """Check warmup/point ordering and the preparation log's file hashes.

    preparation.json identifies point -1 as the warmup. Archive filenames
    are resolved inside the supplied directory, so moving an unchanged table
    bank does not require its earlier absolute paths to remain available.

    Arguments:
      directory = prepared NLA table directory with preparation.json.
      points = number of measured points, excluding the warmup.
    Returns:
      Ordered file records for the warmup and the requested measurements.
    Raises:
      ValueError for missing points, changed file bytes or repeated files.
      OSError when the log or an archive cannot be read.
    """
    supplied_directory = Path(directory)
    bank = supplied_directory.resolve(strict=True)
    preparation_path = bank / "preparation.json"
    preparation_contents = preparation_path.read_text()
    prepared = json.loads(s=preparation_contents)
    if len(prepared) < points + 1:
        raise ValueError(
            "The table bank needs a warmup and all requested points."
        )
    records = []
    hashes = set()
    for index in range(points + 1):
        entry = prepared[index]
        if entry["point"] != index - 1:
            raise ValueError(
                "Require warmup point -1 followed by points 0,1,..."
            )
        old_file = Path(entry["table"]["path"])
        record = identity(path=bank / old_file.name)
        if record["sha256"] != entry["table"]["sha256"]:
            raise ValueError(f"Prepared table bytes changed: {record['path']}")
        if record["sha256"] in hashes:
            raise ValueError(
                "Each physical point needs a different table file."
            )
        hashes.add(record["sha256"])
        record["point"] = entry["point"]
        records.append(record)
    return records


def build_inputs(settings):
    """Prepare 80 Map3 rows and the public C covariance input dictionary.

    Twenty unordered source triplets, each at four aperture radii, give
    80 outputs. Sixteen fields identify the four source/radius combinations
    for each source bin. All three legs of an individual row have one radius.

    Arguments:
      settings = resolved dimensions, integration ranges and model choices.
    Returns:
      rows = SurveyRows with source labels and angles in radians.
      inputs = owned C-contiguous arrays and scalar physical settings.
    Raises:
      ValueError if the shared quadrature or survey-row checks reject inputs.
    """
    import numpy as np
    from cosmolike_des_3pcf_interface import covariance_inputs
    from des3pcf.survey import ARCMIN_TO_RADIAN, forecast_rows

    radii = np.asarray(settings["aperture_radii_arcmin"])
    rows = forecast_rows(nbin=4, aperture_radii=radii * ARCMIN_TO_RADIAN)
    counts = settings["counts"]
    grids = {}
    # Entries name the C input, its bounds, its count and the coordinate
    # choice. A logarithmic rule still returns weights for dx, not d ln x.
    specifications = (
        (
            "ell",
            "ell_bounds",
            "nell",
            True,
        ),
        (
            "phi",
            "phi_bounds",
            "nphi",
            False,
        ),
        (
            "radius",
            "radius_bounds",
            "nradius",
            True,
        ),
        (
            "mass",
            "mass_bounds",
            "nmass",
            True,
        ),
        (
            "k",
            "k_bounds",
            "nk",
            True,
        ),
        (
            "window_ell",
            "window_bounds",
            "nwindow",
            True,
        ),
    )
    for name, bounds_name, count_name, logarithmic in specifications:
        bounds = settings[bounds_name]
        grids[name] = {
            "lower": bounds[0],
            "upper": bounds[1],
            "count": counts[count_name],
            "logarithmic": logarithmic,
        }
    profile_bounds = settings["profile_radius_bounds"]
    positive = np.geomspace(
        start=profile_bounds[0], stop=profile_bounds[1],
        num=settings["profile_positive_count"],
    )
    grids["profile_radius"] = np.concatenate(([0.0], positive))
    scalar_names = (
        "center_window",
        "large_area",
        "noise_mode",
        "estimator",
        "ia_model",
        "st_a",
        "st_p",
        "st_norm",
        "delta_c",
        "collapse_model",
        "profile_upsample",
        "concentration_a",
        "concentration_b",
        "concentration_c",
        "concentration_pivot",
        "baryon_strength",
    )
    physical = {"area": settings["area_steradian"]}
    for name in scalar_names:
        physical[name] = settings[name]
    noise = np.eye(4) * settings["diagonal_noise_steradian"]
    inputs = covariance_inputs(
        kind=rows.kind.tolist(),
        bins=rows.bins,
        angles=rows.angles,
        noise=noise,
        grids=grids,
        physical=physical,
        theta_count=settings["theta_count"],
        pair_weight=settings["pair_weight"],
    )
    return rows, inputs


def gaussian_inputs(model, inputs):
    """Prepare angular power and filters outside Gaussian assembly timing.

    A field combines one source bin and one aperture radius. The raw-map
    Gaussian formula includes six all-cross and nine one-cross pairings.
    Circular-window weights are overlap area divided by survey area squared.
    Finite ell/radius ranges remain the explicitly chosen quadrature model.

    Arguments:
      model = owned physical model for one saved cosmology and source sample.
      inputs = covariance dictionary with radians and steradian-valued noise.
    Returns:
      Dictionary with power[16,16,nell], filters[16,nell], radii[nr+1],
      triplets[80,3] and radial weights[nr+1]. The extra radial node is zero;
      it supplies K(0) and has zero integration weight. nell and nr are the
      requested Fourier and positive radial quadrature counts.
    Raises:
      The physical-model interface refuses invalid power queries.
    """
    import numpy as np

    ell = inputs["ell"]
    source = inputs["field_source"]
    radius = inputs["field_radius"]
    signal = np.empty(shape=(4, 4, ell.size), dtype=np.float64)
    for first in range(4):
        for second in range(first, 4):
            bins = np.asarray([first, second], dtype=np.int32)
            values = model.nla_power(
                ell=ell, bins=bins, parameters={"a1": 0.0},
            )
            signal[first, second] = values
            signal[second, first] = values
    signal += inputs["noise"][:, :, None]
    power = signal[source[:, None], source[None, :], :]
    argument = radius[:, None] * ell[None, :]
    filters = 0.5 * argument**2 * np.exp(-0.5 * argument**2)
    separation = np.concatenate(([0.0], inputs["radius"]))
    circle_radius = np.sqrt(inputs["area"] / np.pi)
    scaled = inputs["radius"] / (2.0 * circle_radius)
    overlap = np.zeros_like(scaled)
    inside = scaled < 1.0
    fraction = scaled[inside]
    overlap[inside] = 2.0 / np.pi / inputs["area"] * (
        np.arccos(fraction) - fraction * np.sqrt(1.0 - fraction**2)
    )
    measure = 2.0 * np.pi * inputs["radius"] * inputs["dr"] * overlap
    return {
        "ell": ell,
        "ell_weights": inputs["dell"],
        "r": separation,
        "filters": np.ascontiguousarray(filters),
        "power": np.ascontiguousarray(power),
        "triplets": inputs["row_fields"],
        "radial_weights": np.concatenate(([0.0], measure)),
    }


def evaluate_gaussian(compiled, prepared):
    """Return pair functions and the raw-map Gaussian covariance.

    The measured interval includes the pair Bessel table, its integrations,
    the contiguous K(0) copy required by the C interface, and Wick sums.

    Arguments:
      compiled = imported project extension with the production adapter.
      prepared = supplied spectra, filters, quadrature and field labels.
    Returns:
      pair[16,16,nr+1], pair_zero[16,16], covariance[80,80], all float64.
      nr is the requested positive radial quadrature count. The first two
      arrays are dimensionless map correlations; the last is their cubic
      product integrated over the normalized survey-center separation.
    Raises:
      The C interface refuses incompatible arrays or nonfinite outputs.
    """
    import numpy as np

    pair = compiled.production.filtered_pair(
        ell=prepared["ell"], ell_weights=prepared["ell_weights"],
        r=prepared["r"], filters=prepared["filters"],
        power=prepared["power"],
    )
    pair_zero = np.ascontiguousarray(pair[:, :, 0])
    covariance = compiled.production.gaussian_aperture_covariance(
        triplets=prepared["triplets"], pair=pair, pair_zero=pair_zero,
        radial_weights=prepared["radial_weights"],
    )
    return pair, pair_zero, covariance


def run_worker(job_path):
    """Load one binary, warm it, then measure changed saved physical points.

    Model ownership is limited to one point. Reusing the covariance input
    dictionary retains fixed numerical geometry, not a saved physical answer.
    The public physical call rebuilds its internal arrays on every invocation.

    Arguments:
      job_path = JSON file declaring binary, tables, settings and worker count.
    Returns:
      None; writes numerical NPZ outputs, timing records and thread settings.
    Raises:
      ValueError for a wrong table-bank model, grid coverage or point identity.
      RuntimeError for changed files, wrong thread limits, nonfinite output,
      missing physical families or a prediction unaffected by changed inputs.
    """
    import numpy as np
    from des3pcf.physical_provider import load_tables
    from theory_inputs import physics_model
    from des3pcf.scaling import _runtime_check

    job_file = Path(job_path)
    job_contents = job_file.read_text()
    job = json.loads(s=job_contents)
    extension = Path(job["extension"]["path"])
    if identity(path=extension)["sha256"] != job["extension"]["sha256"]:
        raise RuntimeError("The selected extension changed before import.")
    os.environ["OMP_NUM_THREADS"] = str(job["threads"])
    sys.path.insert(0, str(extension.parent))
    compiled = importlib.import_module(name=EXTENSION_NAME)
    if Path(compiled.__file__).resolve() != extension:
        raise RuntimeError(
            "Python imported another extension; use a fresh CLI."
        )
    compiled.set_omp_threads(nthreads=job["threads"])
    runtime = _runtime_check(extension=extension, threads=job["threads"])
    directory = Path(job["output"])
    directory.mkdir()
    settings = job["settings"]
    rows, inputs = build_inputs(settings=settings)
    records = []
    previous_cosmology = None
    previous_output = None
    for index, source in enumerate(job["tables"]):
        if identity(path=source["path"])["sha256"] != source["sha256"]:
            raise RuntimeError("A saved table changed before its calculation.")
        tables, metadata = load_tables(path=source["path"])
        point = metadata.get("scaling_point", {})
        if (metadata.get("scaling_format") != "cosmolike-threepcf-scaling-v1"
                or point.get("index") != source["point"]
                or point.get("ia_model") != "NLA"):
            raise ValueError("Require the prepared NLA bank's recorded points.")
        if tables["sources"].shape[0] != 4:
            raise ValueError("This workload requires exactly four source bins.")
        cosmos = {}
        for name in (
            "omega_m",
            "omega_de",
            "w0",
            "wa",
            "ns",
            "sigma8",
        ):
            cosmos[name] = tables[name]
        if previous_cosmology == cosmos:
            raise ValueError("Consecutive saved points must change cosmology.")
        previous_cosmology = cosmos
        lower, upper = settings["los_bounds"]
        if upper > tables["z"][-1]:
            raise ValueError("The saved redshift table must cover z=3.")
        tables["los_z"] = np.linspace(
            start=lower, stop=upper, num=settings["counts"]["nlos"],
        )
        from theory.bihalofit.model import bispectrum_axes, prepare_reference_bispectrum

        model_tables, baseline = prepare_reference_bispectrum(
            tables=tables, axes=bispectrum_axes(zmax=tables["z"][-1]),
            fit_nz=settings["fit_nz"], table_boost=settings["table_boost"],
            source_readers=settings["source_readers"],
        )
        model = physics_model(
            compiled=compiled, tables=model_tables, baseline=baseline, kernel_nz=settings["kernel_nz"],
            table_boost=settings["table_boost"], fit_nz=settings["fit_nz"],
            source_readers=settings["source_readers"],
        )
        prepared = None
        if job["case"] == "gaussian":
            prepared = gaussian_inputs(model=model, inputs=inputs)
        peak_before = peak_memory_bytes()
        started = time.perf_counter_ns()
        if job["case"] == "physical":
            mean, terms = compiled.production.physical_covariance(
                model=model, inputs=inputs,
            )
        else:
            pair, pair_zero, gaussian = evaluate_gaussian(
                compiled=compiled, prepared=prepared,
            )
        seconds = (time.perf_counter_ns() - started) * 1.0e-9
        peak_after = peak_memory_bytes()
        arrays = {}
        if job["case"] == "physical":
            arrays["mean"] = mean
            arrays["terms"] = terms
            for term, name in enumerate(TERM_NAMES):
                arrays[name] = terms[term]
            comparison = terms
            required = (
                0,
                1,
                2,
                4,
                5,
            )
            for term in required:
                if not np.any(terms[term] != 0.0):
                    raise RuntimeError(f"The {TERM_NAMES[term]} term is zero.")
            if np.any(terms[3] != 0.0):
                raise RuntimeError(
                    "Pure Map3 rows must have zero PB cross term."
                )
        else:
            arrays["pair"] = pair
            arrays["pair_zero"] = pair_zero
            arrays["gaussian"] = gaussian
            comparison = gaussian
        for name, array in arrays.items():
            if array.dtype != np.float64 or not np.all(np.isfinite(array)):
                raise RuntimeError(
                    f"{name} must contain finite float64 outputs."
                )
        if previous_output is not None:
            current_bits = comparison.view(dtype=np.uint64)
            previous_bits = previous_output.view(dtype=np.uint64)
            same = np.array_equal(
                a1=current_bits, a2=previous_bits,
            )
            if same:
                raise RuntimeError(
                    "Changed physical points left covariance bits fixed."
                )
        previous_output = comparison.copy()
        output_path = directory / f"point_{index:02d}.npz"
        np.savez_compressed(file=output_path, **arrays)
        record = {
            "point": source["point"],
            "warmup": index == 0,
            "seconds": seconds,
            "threads": job["threads"],
            "physical_point": point,
            "table": source,
            "peak_process_bytes_before_call": peak_before,
            "peak_process_bytes_after_call": peak_after,
            "memory_scope": "cumulative process peak, including setup",
            "rows": int(rows.values.size),
            "fields": int(inputs["field_source"].size),
            "term_max_abs": {},
        }
        for name, array in arrays.items():
            magnitude = np.abs(array)
            maximum = np.max(magnitude)
            record["term_max_abs"][name] = float(maximum)
        records.append(record)
        save_json(path=directory / "records.json", value=records)
        progress = {"point": source["point"], "seconds": seconds}
        message = json.dumps(obj=progress)
        print(message, flush=True)
        del model
    runtime_after = _runtime_check(extension=extension, threads=job["threads"])
    runtime_record = {
        "before": runtime,
        "after": runtime_after,
        "peak_process_bytes_including_saving": peak_memory_bytes(),
    }
    save_json(path=directory / "runtime.json", value=runtime_record)


def compare_outputs(reference, actual, points):
    """Require equal float64 bits for every saved intermediate and family.

    Arguments:
      reference, actual = Paths containing corresponding point_NN.npz files.
      points = measured-point count; point_00 is also checked as the warmup.
    Returns:
      None; reads files without changing them.
    Raises:
      RuntimeError if array names, shapes, dtypes or any float64 bits differ.
    """
    import numpy as np

    for index in range(points + 1):
        filename = f"point_{index:02d}.npz"
        with np.load(file=reference / filename, allow_pickle=False) as expected:
            with np.load(file=actual / filename, allow_pickle=False) as result:
                if set(expected.files) != set(result.files):
                    raise RuntimeError(
                        f"Different saved families in {filename}."
                    )
                for name in expected.files:
                    left = expected[name]
                    right = result[name]
                    equal = left.shape == right.shape
                    equal = equal and left.dtype == right.dtype == np.float64
                    if equal:
                        left_bits = left.view(dtype=np.uint64)
                        right_bits = right.view(dtype=np.uint64)
                        equal = np.array_equal(
                            a1=left_bits, a2=right_bits,
                        )
                    if not equal:
                        raise RuntimeError(
                            f"Changed float64 bits: {actual}/{filename}:{name}"
                        )


def run_controller(args):
    """Copy the selected binary and run one process per worker count.

    Input hashes identify exact file contents. Each child receives the same
    numerical settings and table list. No child overlaps another child's
    calculation, so the requested OpenMP count describes the active workload.

    Arguments:
      args = parsed CLI options, including selected case and output directory.
    Returns:
      None; writes jobs, child logs, a manifest and timing summaries.
    Raises:
      ValueError for repeated worker counts or a mismatched baseline study.
      FileExistsError if the requested output directory already exists.
      RuntimeError for copied-file or numerical-output differences.
      CalledProcessError or TimeoutExpired if a child fails or exceeds its
      declared wall-clock limit.
    """
    if len(set(args.threads)) != len(args.threads):
        raise ValueError("Choose each thread count only once.")
    settings = resolved_settings(args=args)
    extension = identity(path=args.extension)
    tables = table_records(directory=args.tables, points=args.points)
    table_hashes = []
    for record in tables:
        table_hashes.append(record["sha256"])
    definition = {
        "format": FORMAT,
        "case": args.case,
        "settings": settings,
        "table_sha256": table_hashes,
    }
    definition["python_source_sha256"] = {}
    for name in (
        "inputs",
        "mean",
        "covariance",
        "survey",
        "accuracy",
    ):
        source = PROJECT / "scripts" / "des3pcf" / f"{name}.py"
        source_record = identity(path=source)
        definition["python_source_sha256"][name] = source_record["sha256"]
    baseline = None
    if args.baseline is not None:
        baseline = args.baseline.resolve(strict=True)
        manifest_path = baseline / "manifest.json"
        manifest_contents = manifest_path.read_text()
        previous = json.loads(s=manifest_contents)
        if not previous.get("completed", False):
            raise ValueError(
                "The baseline study must have finished its checks."
            )
        if previous["definition"] != definition:
            raise ValueError("Baseline grids, case or physical tables differ.")
    directory = args.output.resolve()
    if directory.exists():
        raise FileExistsError(
            "The output directory exists; choose another path."
        )
    directory.mkdir(parents=True)
    binary_directory = directory / "extension"
    binary_directory.mkdir()
    binary = binary_directory / f"{EXTENSION_NAME}.so"
    shutil.copyfile(src=extension["path"], dst=binary)
    frozen = identity(path=binary)
    if frozen["sha256"] != extension["sha256"]:
        raise RuntimeError("Extension bytes changed while copying them.")
    scope = {
        "physical": (
            "Gravitational full-filter Map3 covariance; internal geometry, "
            "projected spectra and halo preparation included; five nonzero "
            "families and zero PB cross family; the selected response "
            "is not full SSC."
        ),
        "gaussian": (
            "Finite-circle raw-map Gaussian covariance; supplied angular P; "
            "pair Bessel preparation/integrals and all fifteen Wick pairings "
            "included; K(0) packing included; spectra/model preparation "
            "excluded."
        ),
    }
    manifest = {
        "definition": definition,
        "extension_input": extension,
        "extension_frozen": frozen,
        "driver": identity(path=SCRIPT),
        "tables": tables,
        "threads": args.threads,
        "points": args.points,
        "scope": scope[args.case],
        "excluded": "CAMB, model and Python geometry setup, file I/O",
        "baseline": str(baseline) if baseline is not None else None,
        "python": sys.version,
        "platform": platform.platform(),
        "child_timeout_seconds": args.timeout_seconds,
        "completed": False,
    }
    save_json(path=directory / "manifest.json", value=manifest)
    summary = []
    reference = None
    for threads in args.threads:
        destination = directory / f"{args.case}_{threads}"
        job = {
            "extension": frozen,
            "case": args.case,
            "settings": settings,
            "tables": tables,
            "threads": threads,
            "output": str(destination),
        }
        job_path = directory / f"worker_{threads}.json"
        save_json(path=job_path, value=job)
        command = [
            sys.executable,
            "-B",
            str(SCRIPT),
            "_worker",
            "--job",
            str(job_path),
        ]
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = str(threads)
        log_path = directory / f"worker_{threads}.log"
        with log_path.open(mode="w") as log:
            subprocess.run(
                args=command, cwd=PROJECT, env=environment, stdout=log,
                stderr=subprocess.STDOUT, check=True,
                timeout=args.timeout_seconds,
            )
        if reference is None:
            reference = destination
        else:
            compare_outputs(
                reference=reference, actual=destination, points=args.points,
            )
        if baseline is not None:
            prior_threads = previous["threads"][0]
            prior = baseline / f"{args.case}_{prior_threads}"
            compare_outputs(
                reference=prior, actual=destination, points=args.points,
            )
        records_path = destination / "records.json"
        runtime_path = destination / "runtime.json"
        records_contents = records_path.read_text()
        runtime_contents = runtime_path.read_text()
        records = json.loads(s=records_contents)
        runtime = json.loads(s=runtime_contents)
        measured = []
        for record in records:
            if not record["warmup"]:
                measured.append(record["seconds"])
        item = {
            "threads": threads,
            "seconds": measured,
            "median_seconds": statistics.median(data=measured),
            "warmup_seconds": records[0]["seconds"],
            "peak_process_bytes": runtime[
                "peak_process_bytes_including_saving"
            ],
            "matches_first_worker_bits": True,
            "matches_baseline_bits": baseline is not None,
        }
        summary.append(item)
        save_json(path=directory / "summary.json", value=summary)
        message = json.dumps(obj=item)
        print(message, flush=True)
    manifest["completed"] = True
    manifest["cross_thread_comparison_performed"] = len(args.threads) > 1
    manifest["all_saved_bits_match_across_threads"] = True
    save_json(path=directory / "manifest.json", value=manifest)


def main():
    """Set library limits before numerical imports, then dispatch the CLI.

    The run command starts sequential children. The internal _worker command
    reads a job created by that controller. BLAS has one thread; OpenMP allows
    the explicitly selected 1, 4 or 8 workers. The child checks both loaded
    runtimes before calculation and after saving its outputs.

    Returns:
      None; dispatches the controller or one child and writes their records.
    Raises:
      ArgumentParser exits for a missing option or nonpositive timeout.
      Calculation and file failures propagate from the selected command.
    """
    for name in BLAS_VARIABLES:
        os.environ[name] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_THREAD_LIMIT"] = "8"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run sequential isolated workers")
    run.add_argument("--extension", type=Path, required=True)
    run.add_argument("--tables", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--case", choices=("physical", "gaussian"), default="physical",
    )
    allowed_threads = (
        1,
        4,
        8,
    )
    default_threads = [
        1,
        4,
        8,
    ]
    run.add_argument(
        "--threads", type=int, nargs="+", choices=allowed_threads,
        default=default_threads,
    )
    run.add_argument("--points", type=int, choices=range(1, 11), default=3)
    run.add_argument("--baseline", type=Path)
    run.add_argument("--timeout-seconds", type=int, default=900)
    for name in (
        "nell",
        "nphi",
        "nradius",
        "nmass",
        "nk",
        "nwindow",
        "nlos",
    ):
        run.add_argument("--" + name, type=int)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run" and args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive.")
    sys.path.insert(0, str(PROJECT / "scripts"))
    sys.path.insert(0, str(PROJECT.parents[1] / "external_modules/code/cosmolike_core"))
    from threadpoolctl import threadpool_limits

    # The environment limits libraries loaded later. This context also fixes
    # any BLAS library already loaded by the selected Python installation.
    with threadpool_limits(limits=1, user_api="blas"):
        if args.command == "_worker":
            run_worker(job_path=args.job)
        else:
            run_controller(args=args)


if __name__ == "__main__":
    main()
