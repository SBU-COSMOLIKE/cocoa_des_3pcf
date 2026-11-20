"""Compare one native implementation with the saved random-point timing bank.

This optional study uses the same CLI worker as the accepted timing run. It
never generates CAMB tables: each worker reads the existing physical tables
outside its measured intervals. A separate warmup precedes the parameter
points. Every output, including the warmup, must match the saved original
implementation bit for bit. Run variants sequentially on a quiet machine.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[2]


def save_json(path, value):
    """Save a study record as indented JSON with an ending newline.

    Arguments:
      path = destination Path; an existing record is replaced.
      value = JSON-compatible configuration or measurement record.
    Returns:
      None; writes the requested file.
    Raises:
      OSError when the destination cannot be written.
    """
    contents = json.dumps(obj=value, indent=2, sort_keys=True)
    path.write_text(data=contents + "\n")


def prepare_batch_variant(job, directory, batch_size, environment):
    """Change triplet batching while checking reuse of physical tables.

    Input paths become absolute because the copied configuration lives in the
    study output directory. The normal preparation helper checks the physical
    arrays and producer identity before reusing a saved CAMB archive. It writes
    new metadata for the new configuration; no digest check is bypassed.

    Arguments:
      job = original worker dictionary, including its saved parameter bank.
      directory = output Path for copied configuration and table metadata.
      batch_size = number of source triplets prepared together, from 1 to 20.
      environment = library-thread settings inherited by the preparation child.
    Returns:
      Worker dictionary with the copied configuration and reused table paths.
    Raises:
      RuntimeError if the copied configuration changes physical parameters.
      CalledProcessError if table-identity checks fail in the preparation child.
    Side effects:
      Writes configuration, table metadata and a preparation log. The child
      verifies reuse through the project's preparation entry point.
    """
    sys.path.insert(0, str(PROJECT / "scripts"))
    sys.path.insert(0, str(PROJECT.parents[1] / "external_modules/code/cosmolike_core"))
    from des3pcf.scaling_threepcf import _configuration, threepcf_point_bank

    _, config, source_path, _ = _configuration(
        path=job["threepcf_configuration"],
    )
    path_fields = [
        (config["source"], "path"),
        (config["physics"], "tables"),
        (config["rows"], "two_point"),
        (config["rows"], "aperture"),
        (config["mean"]["two_point"], "directory"),
        (config["covariance"], "path"),
        (config["outputs"], "evaluate"),
        (config["outputs"], "fisher"),
    ]
    for container, key in path_fields:
        if container.get(key) is not None:
            input_path = source_path.parent / container[key]
            absolute_path = input_path.resolve()
            container[key] = str(absolute_path)
    config["mean"]["aperture_settings"]["projection_batch_size"] = batch_size
    configuration = directory / f"{job['case']}_batch_{batch_size}.json"
    save_json(path=configuration, value=config)
    bank = threepcf_point_bank(config_path=configuration, npoints=10, seed=137)
    original_points = [job["bank"]["warmup_point"]] + job["bank"]["points"]
    new_points = [bank["warmup_point"]] + bank["points"]
    for original, changed in zip(original_points, new_points):
        if original["parameters"] != changed["parameters"]:
            raise RuntimeError(
                "A batching experiment changed a physical parameter."
            )
    prepared = dict(job)
    prepared["threepcf_configuration"] = str(configuration)
    prepared["bank"] = bank
    prepared["reuse_tables"] = job["tables_directory"]
    tables = directory / f"tables_{job['case']}"
    tables.mkdir()
    prepared["tables_directory"] = str(tables)
    preparation_job = directory / f"prepare_{job['case']}.json"
    save_json(path=preparation_job, value=prepared)
    command = [
        sys.executable,
        "-B",
        str(PROJECT / "scripts/benchmark_scaling.py"),
        "_prepare",
        "--job",
        str(preparation_job),
    ]
    log_path = directory / f"prepare_{job['case']}.log"
    with log_path.open(mode="w") as log:
        subprocess.run(
            args=command, cwd=PROJECT, env=environment,
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    return prepared


def main():
    """Copy one binary, run selected cases, and reject changed output bits.

    Command-line options identify the baseline, selected NLA/TATT cases,
    worker counts and number of physical points. Copying the binary prevents
    a concurrent build from changing the library that later children import.

    Returns:
      None; writes worker logs, output comparisons and timing summaries.
    Raises:
      FileExistsError if the labeled output directory already exists.
      RuntimeError if any measured vector differs from its baseline bits.
      CalledProcessError if a preparation or measurement child fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extension", type=Path, default=(
        PROJECT / "interface/cosmolike_des_3pcf_interface.so"
    ))
    parser.add_argument("--points", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--threads", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--cases", nargs="+", default=["tatt", "nla"],
                        choices=["tatt", "nla"])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--batch-size", type=int, choices=range(1, 21))
    args = parser.parse_args()
    allowed_threads = (
        1,
        4,
        8,
    )
    for count in args.threads:
        if count not in allowed_threads:
            parser.error("Choose 1, 4 or 8 OpenMP threads for this study.")
    directory = args.output.resolve() / args.label
    directory.mkdir(parents=True, exist_ok=False)
    baseline = args.baseline.resolve()
    extension = directory / "cosmolike_des_3pcf_interface.so"
    input_extension = args.extension.resolve()
    shutil.copyfile(src=input_extension, dst=extension)
    binary_bytes = extension.read_bytes()
    binary_hash = hashlib.sha256(binary_bytes)
    digest = binary_hash.hexdigest()
    environment = os.environ.copy()
    for name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    environment["OMP_THREAD_LIMIT"] = "8"
    environment["OMP_DYNAMIC"] = "FALSE"
    environment["MPLCONFIGDIR"] = "/tmp/des3pcf-scaling-matplotlib"
    records = []
    # Import after limiting BLAS. This process performs only small comparisons
    # between workers; it never runs numerical work concurrently with them.
    for name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    import numpy as np

    for case in args.cases:
        batch_template = None
        for threads in args.threads:
            tag = f"threepcf_{case}_{threads}"
            baseline_job = baseline / f"worker_{tag}.json"
            job_contents = baseline_job.read_text()
            job = json.loads(s=job_contents)
            if args.batch_size is not None:
                if batch_template is None:
                    batch_template = prepare_batch_variant(
                        job=job, directory=directory,
                        batch_size=args.batch_size,
                        environment=environment,
                    )
                # A separate nested dictionary keeps each worker's changes
                # from changing the saved batching template.
                template_contents = json.dumps(obj=batch_template)
                job = json.loads(s=template_contents)
                job["threads"] = threads
            job["bank"]["points"] = job["bank"]["points"][:args.points]
            job["extension"] = str(extension)
            job["extension_sha256"] = digest
            result = directory / tag
            job["result_directory"] = str(result)
            job_path = directory / f"worker_{tag}.json"
            save_json(path=job_path, value=job)
            command = [sys.executable, "-B"]
            if args.profile:
                profile_path = directory / f"{tag}.prof"
                command += [
                    "-m",
                    "cProfile",
                    "-o",
                    str(profile_path),
                ]
            command += [
                str(PROJECT / "scripts/benchmark_scaling.py"),
                "_worker",
                "--job",
                str(job_path),
            ]
            environment["OMP_NUM_THREADS"] = str(threads)
            log_path = directory / f"{tag}.log"
            with log_path.open(mode="w") as log:
                subprocess.run(
                    args=command, cwd=PROJECT, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, check=True,
                )
            for index in range(args.points + 1):
                filename = f"point_{index:02d}.npy"
                actual = np.load(file=result / filename, allow_pickle=False)
                original = np.load(
                    file=baseline / tag / filename, allow_pickle=False,
                )
                actual_bits = actual.view(dtype=np.uint64)
                original_bits = original.view(dtype=np.uint64)
                if not np.array_equal(a1=actual_bits, a2=original_bits):
                    raise RuntimeError(f"Changed output bits: {tag}/{filename}")
            measured_path = result / "records.json"
            original_path = baseline / tag / "records.json"
            measured_contents = measured_path.read_text()
            original_contents = original_path.read_text()
            measured = json.loads(s=measured_contents)
            original = json.loads(s=original_contents)
            original = original[:args.points]
            elapsed = [record["seconds"] for record in measured]
            original_elapsed = [record["seconds"] for record in original]
            median = statistics.median(data=elapsed)
            original_median = statistics.median(data=original_elapsed)
            record = {
                "case": case,
                "threads": threads,
                "points": args.points,
                "median_seconds": median,
                "original_median_seconds": original_median,
                "seconds": elapsed,
                "bitwise_original_including_warmup": True,
                "profiled": args.profile,
                "batch_size_override": args.batch_size,
            }
            records.append(record)
            save_json(path=directory / "summary.json", value=records)
            message = json.dumps(obj=record)
            print(message, flush=True)
    manifest = {
        "label": args.label,
        "extension_sha256": digest,
        "baseline": str(baseline),
        "profiled": args.profile,
        "points": args.points,
        "all_vectors_bitwise_original": True,
    }
    save_json(path=directory / "manifest.json", value=manifest)


if __name__ == "__main__":
    main()
