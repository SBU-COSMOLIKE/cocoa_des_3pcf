"""Retain the completed LSST controls and run corrected three-point inputs.

Only the LSST records are imported. They must match the complete current
case specification and pass the same saved-vector checks. Both three-point
cases prepare or verify their own inputs and run every measured prediction.
"""

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

for name in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_THREAD_LIMIT"] = "8"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MPLCONFIGDIR"] = "/tmp/des3pcf-scaling-matplotlib"

project = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(project / "python"))
from des3pcf.scaling import (
    preflight, file_record, validate_vectors, plot_study, _child, _write_json,
)

configuration = project / "configs/openmp_scaling.yaml"
previous = project / "outputs/openmp_scaling_ia"
report = preflight(configuration)
if not report["ready"]:
    raise RuntimeError("Resolve the current input preflight before continuing.")
old = json.loads((previous / "manifest.json").read_text())
controls = ["lsst_nla", "lsst_tatt"]
validate_vectors(previous, workloads=controls)
for case in controls:
    current_identity = json.dumps(report["cases"][case], sort_keys=True)
    saved_identity = json.dumps(old["cases"][case], sort_keys=True)
    if current_identity != saved_identity:
        raise RuntimeError(f"The LSST control specification changed: {case}.")

config = report["configuration"]
output = Path(config["output"])
output.mkdir(parents=True, exist_ok=False)
report["started_utc"] = datetime.now(timezone.utc).isoformat()
report["machine"] = old["machine"]
report["numerical_work_executed"] = True
report["continued_from"] = str(previous)
report["copied_control_cases"] = controls
report["continuation_script"] = file_record(__file__)
_write_json(output / "manifest.json", report)
shutil.copy2(__file__, output / "continue_des3pcf_scaling.py")
for name in ("hardware.json", "environment.json"):
    shutil.copy2(previous / name, output / name)

# Reuse the captured binaries so subsequent work in another checkout cannot
# change the compiled implementation midway through this comparison.
shutil.copytree(previous / "extensions", output / "extensions")
records = []
for case in controls:
    for threads in config["threads"]:
        directory = f"{case}_{threads}"
        shutil.copytree(previous / directory, output / directory)
        records.extend(json.loads((output / directory / "records.json").read_text()))

for case in ("threepcf_nla", "threepcf_tatt"):
    spec = report["cases"][case]
    tables = output / "tables" / case
    tables.mkdir(parents=True)
    binary = output / "extensions/threepcf/cosmolike_des_3pcf_interface.so"
    expected = json.loads((previous / "worker_threepcf_nla_4.json").read_text())
    if file_record(binary)["sha256"] != expected["extension_sha256"]:
        raise RuntimeError("The copied three-point binary changed.")
    reuse = None
    if case == "threepcf_tatt":
        reuse = str(output / "tables/threepcf_nla")
    template = {
        "case": case,
        "workload": "threepcf",
        "bank": spec["bank"],
        "extension": str(binary),
        "extension_sha256": file_record(binary)["sha256"],
        "tables_directory": str(tables),
        "reuse_tables": reuse,
    }
    template.update(spec["specification"])
    path = output / f"prepare_{case}.json"
    _write_json(path, template)
    print(f"Preparing {case} with bounded CAMB and interpolated tails", flush=True)
    _child("_prepare", path, threads=1, timeout=config["child_timeout_seconds"])
    for threads in config["thread_order"]:
        job = dict(template)
        job["threads"] = threads
        job["result_directory"] = str(output / f"{case}_{threads}")
        path = output / f"worker_{case}_{threads}.json"
        _write_json(path, job)
        print(f"Running {case}: {threads} threads, ten points", flush=True)
        _child("_worker", path, threads=threads,
               timeout=config["child_timeout_seconds"])
        result = Path(job["result_directory"]) / "records.json"
        records.extend(json.loads(result.read_text()))
    validate_vectors(output, workloads=[case])

validate_vectors(output)
_write_json(output / "records.json", records)
with (output / "timings.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=("workload", "threads", "repetition", "seconds", "rows"),
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(records)
_write_json(output / "validation.json", {
    "thread_results_bitwise_equal": True,
    "random_points_per_workload": 10,
    "timed_vectors": 120,
})
print(plot_study(configuration), flush=True)
