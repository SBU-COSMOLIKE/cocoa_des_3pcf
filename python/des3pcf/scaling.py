"""Measure changed physical states through isolated command-line processes.

Each workload has its own explicit ten-point parameter bank. Thread counts
reuse identical points, physical tables and numerical resolutions. The first
evaluation is a separate warmup point. A complete block visits all ten random
points, so successive calls cannot become same-cosmology cache hits.

CAMB, the solver that supplies the matter power spectrum, and file I/O are
outside the data-vector interval. Per-point preparation durations are saved
separately. No covariance or Fisher matrix is timed. The explicit entry point
is ``scripts/benchmark_scaling.py run --config configs/openmp_scaling.yaml``;
the ``preflight`` command checks the declared inputs without running a model.
"""

import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml


# Paths follow this project's layout; each child uses the same command-line
# script so its imports and environment can be recorded before calculation.
PROJECT = Path(__file__).resolve().parents[2]
COCOA = PROJECT.parents[1]
SCRIPT = PROJECT / "scripts" / "benchmark_scaling.py"
LABELS = {
    "lsst_nla": "LSST Y1 3x2pt NLA",
    "lsst_tatt": "LSST Y1 3x2pt TATT",
    "threepcf_nla": "DES Y3 three-point NLA",
    "threepcf_tatt": "DES Y3 three-point TATT",
}


def file_record(path):
    """Record an input's identity without reading the entire file at once.

    SHA-256 identifies the file bytes, so changing a table after its preparation
    is detected before it can enter a measured calculation.

    Arguments:
      path = existing input file.
    Returns:
      Dictionary containing its absolute path, SHA-256 and size in bytes.
    """
    filename = Path(path).resolve()
    digest = hashlib.sha256()
    with filename.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return {
        "path": str(filename),
        "sha256": digest.hexdigest(),
        "bytes": filename.stat().st_size,
    }


def load_configuration(path):
    """Resolve the declared study without creating its output directory.

    Relative filenames belong to the YAML's directory, independently of the
    shell's current directory. The exact field set prevents misspelled options
    from silently changing what the study measures.

    Arguments:
      path = scaling YAML filename.
    Returns:
      Dictionary with resolved paths and the configuration file's identity.
    Raises:
      ValueError when fields or thread counts differ from this study's design.
    """
    filename = Path(path).resolve()
    config = yaml.safe_load(filename.read_text())
    expected = {
        "format",
        "seed",
        "points",
        "threads",
        "thread_order",
        "threepcf_nla_configuration",
        "threepcf_tatt_configuration",
        "output",
        "child_timeout_seconds",
    }
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("Scaling YAML must supply exactly the documented fields.")
    if config["format"] != "des3pcf-openmp-scaling-v2" or config["points"] != 10:
        raise ValueError("This study requires its v2 format and ten random points.")
    required_threads = [
        1,
        4,
        8,
    ]
    if (config["threads"] != required_threads
            or sorted(config["thread_order"]) != required_threads):
        raise ValueError("Use each of 1, 4 and 8 OpenMP threads exactly once.")
    for name in ("threads", "thread_order"):
        # Python treats True as an integer in arithmetic. The exact type check
        # prevents a Boolean YAML option from being accepted as one worker.
        if any(type(value) is not int for value in config[name]):
            raise ValueError("Thread counts must be integers, not Boolean values.")
    for name in ("seed", "points", "child_timeout_seconds"):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    if config["child_timeout_seconds"] == 0:
        raise ValueError("A child timeout must be positive.")
    for name in ("threepcf_nla_configuration", "threepcf_tatt_configuration", "output"):
        config[name] = str((filename.parent / config[name]).resolve())
    config["configuration_file"] = file_record(filename)
    return config


def preflight(path):
    """Check inputs and draw both parameter lists without running a model.

    Each list contains ten reproducible random cosmologies and one distinct
    warmup cosmology. The same lists enter all three thread-count runs, so
    differences in parameter values cannot imitate thread scaling.

    Arguments:
      path = scaling YAML filename.
    Returns:
      JSON-compatible input report, including any three-point input blockers.
    """
    from .scaling_lsst import default_lsst_settings, preflight_lsst, lsst_point_bank
    from .scaling_threepcf import preflight_threepcf, threepcf_point_bank

    config = load_configuration(path)
    cases = {}
    for case in LABELS:
        workload, model = case.split("_")
        if workload == "lsst":
            settings = default_lsst_settings(COCOA, ia_model=model)
            checked = preflight_lsst(settings)
            bank = lsst_point_bank(settings, npoints=10, seed=config["seed"])
            specification = {"lsst_settings": settings}
        else:
            filename = config[f"threepcf_{model}_configuration"]
            checked = preflight_threepcf(filename)
            bank = threepcf_point_bank(filename, npoints=10, seed=config["seed"])
            specification = {"threepcf_configuration": filename}
        cases[case] = {
            "workload": workload,
            "ia_model": model,
            "inputs": checked,
            "bank": bank,
            "specification": specification,
        }
    report = {
        "configuration": config,
        "cases": cases,
        "scope": (
            "120 timed vectors; twelve separate untimed warmup vectors; "
            "CAMB excluded from data-vector intervals. Identical physical "
            "tables may be reused between IA models after exact input checks."
        ),
        "numerical_work_executed": False,
    }
    report["ready"] = all(item["inputs"].get("ready", True) for item in cases.values())
    return report


def _read_record(path):
    """Read a YAML record or its frozen JSON predecessor under tests/."""
    path = Path(path)
    if not path.is_file():
        archived = path.with_suffix(".json")
        if archived.is_file():
            path = archived
    return yaml.safe_load(path.read_text())


def _write_yaml(path, value):
    """Write a readable YAML record, refusing nonfinite numerical values.

    Arguments:
      path = output filename; an existing record is replaced.
      value = JSON-compatible object to save.
    Returns:
      None. The supplied file is written; missing directories are not created.
    Raises:
      ValueError if a numerical value is NaN or infinite.
    """
    plain = json.loads(json.dumps(value, allow_nan=False))
    Path(path).write_text(yaml.safe_dump(plain, sort_keys=False))


def _child(command, job, threads, timeout):
    """Run one child with explicit thread limits and separate output logs.

    The parent waits for this child to exit before starting another. BLAS
    matrix-library threads stay at one while OpenMP controls the C calculation.
    This prevents two independent thread pools from competing for the CPUs.

    Arguments:
      command = internal CLI command, either _prepare or _worker.
      job = JSON filename containing this child's inputs.
      threads = requested OpenMP worker count.
      timeout = maximum duration in seconds before the child is stopped.
    Returns:
      None. Standard output and errors are saved beside the job file.
    Raises:
      subprocess errors if the child fails or exceeds its timeout.
    """
    environment = dict(os.environ)
    for name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    environment.update(
        OMP_NUM_THREADS=str(threads),
        OMP_THREAD_LIMIT="8",
        OMP_DYNAMIC="FALSE",
        PYTHONDONTWRITEBYTECODE="1",
    )
    environment["OMP_PROC_BIND"] = "disabled" if sys.platform == "darwin" else "close"
    environment["OMP_PLACES"] = "cores"
    prefix = Path(job).with_suffix("")
    with Path(str(prefix) + ".stdout.log").open("w") as stdout:
        with Path(str(prefix) + ".stderr.log").open("w") as stderr:
            arguments = [
                sys.executable,
                "-B",
                str(SCRIPT),
                command,
                "--job",
                str(job),
            ]
            subprocess.run(
                arguments,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=True,
                timeout=timeout,
            )


def prepare_job(path):
    """Prepare one workload's warmup and ten random-point physical archives.

    These durations include CAMB, source-table preparation and file writing.
    They are separate from the later mean-vector calculation; they must not
    be interpreted as CAMB-only durations.

    Arguments:
      path = preparation job's JSON filename.
    Returns:
      None. Writes eleven NPZ tables and preparation.yaml in the job directory.
    """
    from .scaling_lsst import prepare_lsst_tables
    from .scaling_threepcf import prepare_threepcf_tables

    job = yaml.safe_load(Path(path).read_text())
    points = [job["bank"]["warmup_point"]] + job["bank"]["points"]
    records = []
    for index, point in enumerate(points):
        destination = Path(job["tables_directory"]) / f"point_{index:02d}.npz"
        reuse_path = None
        if job.get("reuse_tables") is not None:
            previous = Path(job["reuse_tables"])
            prior_records = _read_record(previous / "preparation.yaml")
            source = prior_records[index]["table"]
            if file_record(source["path"])["sha256"] != source["sha256"]:
                raise RuntimeError("A saved preparation archive changed before reuse.")
            reuse_path = source["path"]
        started = time.perf_counter_ns()
        if job["workload"] == "lsst":
            prepare_lsst_tables(
                job["lsst_settings"], destination, point, reuse_path=reuse_path,
            )
        else:
            prepare_threepcf_tables(
                job["threepcf_configuration"], point, destination, reuse_path=reuse_path,
            )
        elapsed = (time.perf_counter_ns() - started) * 1.e-9
        records.append({
            "point": index - 1,
            "preparation_seconds": elapsed,
            "reused_archive": reuse_path,
            "table": file_record(destination),
        })
        _write_yaml(Path(job["tables_directory"]) / "preparation.yaml", records)
        print(f"Prepared {job['workload']} point {index - 1}", flush=True)


def _runtime_check(extension, threads):
    """Verify actual runtime thread settings after loading the C extension.

    Environment variables alone cannot prove that a loaded library obeyed the
    requested settings. Calling this extension's OpenMP runtime and inspecting
    loaded BLAS pools checks the active libraries before and after the run.

    Arguments:
      extension = path to the imported shared library.
      threads = requested OpenMP worker count.
    Returns:
      Dictionary of OpenMP limits and loaded thread-pool settings.
    Raises:
      RuntimeError if OpenMP or BLAS settings differ from the declared limits.
    """
    from threadpoolctl import threadpool_info

    library = ctypes.CDLL(str(extension))
    for symbol in ("omp_get_max_threads", "omp_get_thread_limit"):
        function = getattr(library, symbol)
        function.argtypes = []
        function.restype = ctypes.c_int
    maximum = library.omp_get_max_threads()
    limit = library.omp_get_thread_limit()
    pools = threadpool_info()
    if maximum != threads or limit < threads:
        raise RuntimeError(
            f"Requested {threads} OpenMP threads; runtime has {maximum}/{limit}."
        )
    for pool in pools:
        if pool["user_api"] == "blas" and pool["num_threads"] != 1:
            raise RuntimeError("Every loaded BLAS pool must use exactly one thread.")
    return {
        "omp_max_threads": maximum,
        "omp_thread_limit": limit,
        "pools": pools,
    }


def run_job(path):
    """Warm one fresh CLI process, then evaluate ten different physical states.

    Saved input tables are read before each interval. The interval includes
    the workload's physical-model preparation and mean prediction, including
    validation performed inside evaluate. Geometry shared by all points is
    prepared once when the workload is created.

    Arguments:
      path = worker job's JSON filename with tables, points and thread count.
    Returns:
      None. Writes eleven vectors, one warmup record and ten measured records.
    Raises:
      RuntimeError for changed input bytes, thread settings or nonfinite output.
    """
    from .scaling_lsst import initialize_lsst, load_lsst_tables
    from .scaling_threepcf import initialize_threepcf, load_threepcf_tables

    job = yaml.safe_load(Path(path).read_text())
    threads = job["threads"]
    directory = Path(job["result_directory"])
    directory.mkdir()
    if file_record(job["extension"])["sha256"] != job["extension_sha256"]:
        raise RuntimeError("The frozen extension differs from its recorded bytes.")
    if job["workload"] == "lsst":
        workload = initialize_lsst(
            job["lsst_settings"],
            threads=threads,
            extension_path=job["extension"],
        )
        loader = load_lsst_tables
    else:
        workload = initialize_threepcf(
            job["threepcf_configuration"],
            threads=threads,
            extension_path=job["extension"],
        )
        loader = load_threepcf_tables
    runtime = _runtime_check(job["extension"], threads)
    preparation_file = Path(job["tables_directory"]) / "preparation.yaml"
    prepared = _read_record(preparation_file)
    points = [job["bank"]["warmup_point"]] + job["bank"]["points"]
    records = []
    for index, point in enumerate(points):
        source = prepared[index]["table"]
        if file_record(source["path"])["sha256"] != source["sha256"]:
            raise RuntimeError("A physical table changed after preparation.")
        bundle = loader(source["path"])
        # Disk reads, file-identity checks, thread selection and output saving
        # are outside this interval. Index zero is a separate warmup point;
        # its duration is saved but does not enter the ten measured records.
        started = time.perf_counter_ns()
        if job["workload"] == "lsst":
            vector = workload.evaluate(bundle)
        else:
            vector = workload.evaluate(bundle, point)
        elapsed = (time.perf_counter_ns() - started) * 1.e-9
        values = np.asarray(vector, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise RuntimeError("A prediction must be a nonempty finite vector.")
        np.save(directory / f"point_{index:02d}.npy", values, allow_pickle=False)
        if index == 0:
            _write_yaml(directory / "warmup.yaml", {
                "seconds": elapsed,
                "runtime": runtime,
                "point": point,
                "rows": values.size,
            })
        else:
            records.append({
                "workload": LABELS[job["case"]],
                "threads": threads,
                "repetition": index - 1,
                "seconds": elapsed,
                "point": point,
                "rows": values.size,
            })
            _write_yaml(directory / "records.yaml", records)
        message = f"{job['workload']} threads={threads} point={index - 1} complete"
        print(message, flush=True)
    _runtime_check(job["extension"], threads)
    # Peak process memory includes imports, geometry and physical tables;
    # it is not the incremental memory of one timed prediction. macOS gives
    # bytes, whereas Linux gives KiB. Read this only after all intervals.
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    _write_yaml(directory / "memory.yaml", {
        "peak_process_resident_bytes": int(peak),
        "scope": "Whole worker process, including preparation and imports.",
    })


def validate_vectors(output, workloads=None):
    """Require identical float64 bits for each point across worker counts.

    Consecutive parameter points must also change the output. That second
    check detects a calculation which accidentally reuses an earlier result
    instead of responding to the changed physical parameters.

    Arguments:
      output = root directory containing workload_1, workload_4 and workload_8.
      workloads = names to check; None selects both declared workloads.
    Returns:
      None; saved vectors are read without modifying them.
    Raises:
      RuntimeError for an unchanged consecutive point or thread-dependent bits.
    """
    root = Path(output)
    if workloads is None:
        workloads = LABELS
    for workload in workloads:
        previous = None
        for index in range(11):
            reference = np.load(
                root / f"{workload}_1" / f"point_{index:02d}.npy",
                allow_pickle=False,
            )
            if previous is not None and np.array_equal(reference, previous):
                raise RuntimeError(
                    f"{workload}, point {index - 1}: a changed point returned "
                    "the identical vector; inspect state invalidation."
                )
            previous = reference
            for threads in (4, 8):
                filename = root / f"{workload}_{threads}" / f"point_{index:02d}.npy"
                actual = np.load(filename, allow_pickle=False)
                # Viewing float64 bytes as uint64 compares every stored bit,
                # including the sign of zero; an approximate equality check
                # would hide changes caused by the number of worker threads.
                if (actual.shape != reference.shape
                        or not np.array_equal(actual.view(np.uint64),
                                              reference.view(np.uint64))):
                    raise RuntimeError(
                        f"{workload}, point {index - 1}: "
                        f"output changed at {threads} threads."
                    )


def run_study(path, reuse_from=None):
    """Prepare and run the study, completing the LSST control first.

    Each workload finishes its eleven table preparations, three thread-count
    runs and bit comparisons before the next workload starts. A failure stops
    subsequent calculations while preserving records already written.

    Arguments:
      path = scaling YAML filename; its output directory must not exist.
      reuse_from = optional earlier study with saved LSST physical tables.
          The archives are hash checked and their physical requests compared
          before reuse. No saved timing is imported into this study.
    Returns:
      Path to the directory containing inputs, timings, vectors and figures.
    Raises:
      ValueError for unresolved inputs; child or vector-check failures propagate.
    """
    report = preflight(path)
    if not report["ready"]:
        raise ValueError("Preflight found unresolved inputs; inspect the case reports.")
    config = report["configuration"]
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    report["started_utc"] = datetime.now(timezone.utc).isoformat()
    report["machine"] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
    }
    report["numerical_work_executed"] = True
    _write_yaml(output / "manifest.yaml", report)
    records = []
    # Copy each binary once so both IA models use the same compiled code even
    # if another team rebuilds a library while this study is running.
    extensions = {
        "lsst": Path(
            report["cases"]["lsst_nla"]["specification"]["lsst_settings"]
            ["extension_path"]
        ),
        "threepcf": PROJECT / "interface" / "cosmolike_des_3pcf_interface.so",
    }
    binaries = {}
    for workload, extension in extensions.items():
        frozen = output / "extensions" / workload
        frozen.mkdir(parents=True)
        copied = frozen / extension.name
        shutil.copy2(extension, copied)
        binaries[workload] = file_record(copied)
    # NLA precedes TATT for each data vector. The physical tables depend on
    # cosmology and source shifts, so each TATT case can reuse its NLA tables
    # after the helper proves that those inputs are identical. C predictions
    # and physical C-model preparation are still recomputed inside the timer.
    for case, specification in report["cases"].items():
        workload = specification["workload"]
        tables = output / "tables" / case
        tables.mkdir(parents=True)
        reuse_tables = None
        if specification["ia_model"] == "tatt":
            reuse_tables = str(output / "tables" / f"{workload}_nla")
        elif workload == "lsst" and reuse_from is not None:
            reuse_tables = str(Path(reuse_from).resolve() / "tables" / "lsst")
        job = {
            "case": case,
            "workload": workload,
            "bank": specification["bank"],
            "extension": binaries[workload]["path"],
            "extension_sha256": binaries[workload]["sha256"],
            "tables_directory": str(tables),
            "reuse_tables": reuse_tables,
        }
        job.update(specification["specification"])
        filename = output / f"prepare_{case}.yaml"
        _write_yaml(filename, job)
        print(f"Preparing eleven {case} physical tables with one worker", flush=True)
        _child("_prepare", filename, threads=1, timeout=config["child_timeout_seconds"])
        template = job
        for threads in config["thread_order"]:
            job = dict(
                template,
                threads=threads,
                result_directory=str(output / f"{case}_{threads}"),
            )
            filename = output / f"worker_{case}_{threads}.yaml"
            _write_yaml(filename, job)
            print(f"Running {case}: {threads} threads, ten points", flush=True)
            _child(
                "_worker",
                filename,
                threads=threads,
                timeout=config["child_timeout_seconds"],
            )
            result_file = Path(job["result_directory"]) / "records.yaml"
            records.extend(_read_record(result_file))
        validate_vectors(output, workloads=[case])
    validate_vectors(output)
    _write_yaml(output / "records.yaml", records)
    with (output / "timings.csv").open("w", newline="") as stream:
        fields = ("workload", "threads", "repetition", "seconds", "rows")
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)
    _write_yaml(output / "validation.yaml", {
        "thread_results_bitwise_equal": True,
        "random_points_per_workload": 10,
        "timed_vectors": 120,
    })
    plot_study(path)
    return output


def plot_study(path):
    """Plot measured durations only after their thread-consistency check passes.

    Plotting reads saved records and performs no physical calculation. The
    required validation file keeps a failed thread comparison from producing
    a figure that could be mistaken for an accepted scaling measurement.

    Arguments:
      path = scaling YAML filename identifying an existing result directory.
    Returns:
      Path to the saved PDF; a PNG is also written in the same directory.
    Raises:
      ValueError if the saved thread-consistency check is absent or false.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from .scaling_plot import plot_thread_scaling

    config = load_configuration(path)
    output = Path(config["output"])
    validation = _read_record(output / "validation.yaml")
    if validation.get("thread_results_bitwise_equal") is not True:
        raise ValueError("Scaling plots require the saved thread-consistency check.")
    records = _read_record(output / "records.yaml")
    matplotlib.rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "xtick.bottom": True,
        "xtick.top": False,
        "ytick.right": False,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "axes.labelsize": "medium",
        "axes.grid": True,
        "grid.linewidth": 0.0,
        "grid.alpha": 0.18,
        "grid.color": "lightgray",
        "legend.labelspacing": 0.77,
        "savefig.bbox": "tight",
        "savefig.format": "pdf",
    })
    figure, axes = plot_thread_scaling(
        records, title="Ten random parameter points; CAMB excluded; BLAS = 1",
    )
    # LSST is a control calculation with much shorter elapsed times. A log
    # axis keeps those measured values visible alongside the three-point ones.
    elapsed = [record["seconds"] for record in records]
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.8 * min(elapsed), 1.2 * max(elapsed))
    figure.savefig(output / "openmp_scaling.pdf")
    figure.savefig(output / "openmp_scaling.png", dpi=180)
    plt.close(figure)
    threepoint = []
    for record in records:
        if record["workload"].startswith("DES Y3 three-point"):
            threepoint.append(record)
    figure, _ = plot_thread_scaling(
        threepoint, title="3pcf NLA and TATT; CAMB excluded; BLAS = 1",
    )
    figure.savefig(output / "threepcf_scaling.pdf")
    figure.savefig(output / "threepcf_scaling.png", dpi=180)
    plt.close(figure)
    return output / "openmp_scaling.pdf"
