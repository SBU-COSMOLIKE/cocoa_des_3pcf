"""Time the Cobaya evaluation of a numbered example at several thread counts.

Usage (from Cocoa/, with the cocoa environment active and start_cocoa.sh
sourced):

  python projects/des_3pcf/tests/benchmark_scaling.py \
      --yaml projects/des_3pcf/EXAMPLE_EVALUATE1.yaml \
      --output projects/des_3pcf/chains/benchmark_scaling/nla \
      --threads 1 4 8

EXAMPLE_EVALUATE1 is the NLA case and EXAMPLE_EVALUATE2 the TATT case; run
them separately. Use a quiet machine: other programs change the times.

What is measured. Each thread count runs in a fresh child process with
OMP_NUM_THREADS set and every BLAS library at one thread. The child builds
the Cobaya model once, evaluates one warm-up point (the yaml's fiducial),
then the same saved parameter points, each with model.logposterior. Every
point changes cosmology and nuisance parameters, so CAMB, the bispectrum
block and the likelihood all recompute; geometry and FFT plans keep their
ordinary reuse. Three quantities come from the same evaluations:

  full  = elapsed time of model.logposterior;
  camb  = CAMB's own intervals, read from Cobaya's component timers;
          CAMB is two components, camb.transfers (the Boltzmann solve)
          and camb (the power spectra), and both are counted;
  other = full - camb: the bispectrum block, any feedback block and the
          likelihood's native preparation and projection.

Each component's timer interval is saved too. The run fails if a
component does not recompute at a point, if two points give the same
vector, or if the thread counts disagree beyond the tolerance.

Outputs (in --output, which belongs under the ignored chains/ folder):
points.npz (the saved points), threads_<n>.npz (one child) and
summary.json (settings, medians and checks).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np


BLAS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

LIKELIHOOD = "des_3pcf.joint_shear"


def parse(argv):
    """Read the command line; --child is the internal one-process mode."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--yaml", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--npoints", type=int, default=10)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    parser.add_argument("--child", type=int, default=0)
    return parser.parse_args(argv)


def model_information(path, threads):
    """Load the example yaml for timing: no sampler, no output, timers on."""
    from cobaya.yaml import yaml_load_file

    info = yaml_load_file(str(path))
    fiducial = dict(info["sampler"]["evaluate"]["override"])
    info.pop("sampler")
    info.pop("output", None)
    info["timing"] = True
    info["likelihood"][LIKELIHOOD]["omp_threads"] = threads
    return info, fiducial


def saved_points(model, arguments, digest):
    """Read points.npz, or draw the points from the prior and write it.

    The points are drawn once, with Cobaya's prior sampler and a fixed
    seed, and every later child reads the same file. A file written with
    another yaml, seed or count raises instead of being reused.
    """
    path = arguments.output / "points.npz"
    names = list(model.prior.params)

    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved["yaml_sha256"]) != digest
                    or int(saved["seed"]) != arguments.seed
                    or saved["values"].shape[0] != arguments.npoints
                    or list(saved["names"]) != names):
                raise ValueError(f"{path} was written with other settings; "
                                 "choose a new --output.")
            values = saved["values"].copy()
    else:
        random = np.random.RandomState(arguments.seed)
        values = model.prior.sample(n=arguments.npoints, ignore_external=True,
                                    random_state=random)
        np.savez(path, names=np.array(names), values=values,
                 seed=arguments.seed, yaml_sha256=digest)

    points = []
    for row in values:
        points.append(dict(zip(names, (float(value) for value in row))))
    return points


def child(arguments):
    """Build the model once and time the warm-up and the saved points."""
    from cobaya.model import get_model

    digest = hashlib.sha256(arguments.yaml.read_bytes()).hexdigest()
    info, fiducial = model_information(path=arguments.yaml,
                                       threads=arguments.child)
    model = get_model(info_or_yaml_or_file=info)
    likelihood = model.likelihood[LIKELIHOOD]
    components = [component for component in model.components
                  if component.timer is not None]
    names = [component.get_name() for component in components]
    points = saved_points(model=model, arguments=arguments, digest=digest)

    # ----------------------------------------------------------------------
    # Warm-up: the first call fills plans and caches; it is not timed.
    # ----------------------------------------------------------------------
    model.logposterior(params_values=fiducial, cached=False)

    full = np.empty(len(points))
    spent = np.empty((len(points), len(components)))
    calls = np.empty((len(points), len(components)), dtype=np.int64)
    vectors = []
    for index, point in enumerate(points):
        before = [(c.timer.n, c.timer.time_sum) for c in components]
        start = time.perf_counter()
        model.logposterior(params_values=point, cached=False)
        full[index] = time.perf_counter() - start
        for j, component in enumerate(components):
            calls[index, j] = component.timer.n - before[j][0]
            spent[index, j] = component.timer.time_sum - before[j][1]
        vectors.append(np.array(likelihood.last_datavector, copy=True))

    np.savez(arguments.output / f"threads_{arguments.child}.npz",
             names=np.array(names), full=full, spent=spent, calls=calls,
             datavector=np.array(vectors))


def summary(arguments):
    """Check the children's records and write the medians to summary.json."""
    records = {}
    for threads in arguments.threads:
        path = arguments.output / f"threads_{threads}.npz"
        with np.load(path, allow_pickle=False) as saved:
            records[threads] = {name: saved[name].copy() for name in saved.files}
    names = list(records[arguments.threads[0]]["names"])
    camb = []
    for j, name in enumerate(names):
        if "camb" == name or name.startswith("camb."):
            camb.append(j)
    if 0 == len(camb):
        raise ValueError("The model has no CAMB component to time.")

    # ----------------------------------------------------------------------
    # Checks: every component recomputes at every point, consecutive
    # points differ, and each thread count matches the first one.
    # ----------------------------------------------------------------------
    reference = records[arguments.threads[0]]["datavector"]
    agreement = {}
    for threads, record in records.items():
        if np.any(1 != record["calls"]):
            raise ValueError(f"{threads} threads: a component was not "
                             "evaluated exactly once at every point.")
        vectors = record["datavector"]
        for index in range(1, vectors.shape[0]):
            if np.array_equal(vectors[index], vectors[index - 1]):
                raise ValueError(f"{threads} threads: points {index - 1} and "
                                 f"{index} gave the same vector.")
        scale = np.max(np.abs(reference), axis=1)
        change = np.max(np.abs(vectors - reference), axis=1) / scale
        agreement[threads] = float(np.max(change))
        if agreement[threads] > arguments.tolerance:
            raise ValueError(f"{threads} threads differ from "
                             f"{arguments.threads[0]} by {agreement[threads]}.")

    medians = {}
    for threads, record in records.items():
        camb_time = np.sum(record["spent"][:, camb], axis=1)
        medians[threads] = {
            "full": float(np.median(record["full"])),
            "camb": float(np.median(camb_time)),
            "other": float(np.median(record["full"] - camb_time)),
        }
        for j, name in enumerate(names):
            medians[threads][f"component:{name}"] = float(
                np.median(record["spent"][:, j]))

    result = {
        "yaml": str(arguments.yaml),
        "yaml_sha256": hashlib.sha256(arguments.yaml.read_bytes()).hexdigest(),
        "threads": arguments.threads,
        "npoints": arguments.npoints,
        "seed": arguments.seed,
        "tolerance": arguments.tolerance,
        "boundaries": {
            "full": "elapsed time of model.logposterior(point, cached=False)",
            "camb": "CAMB's Cobaya component timers (camb.transfers + "
                    "camb) within that call",
            "other": "full minus camb: bispectrum and feedback blocks plus "
                     "the likelihood's native preparation and projection",
        },
        "components": names,
        "median_seconds": medians,
        "max_relative_difference_to_first_thread_count": agreement,
        "machine": {"platform": platform.platform(),
                    "processor": platform.processor(),
                    "cpu_count": os.cpu_count()},
    }
    (arguments.output / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    """Run one child per thread count, then check and summarize."""
    arguments = parse(sys.argv[1:] if argv is None else argv)
    arguments.yaml = arguments.yaml.resolve()
    arguments.output = arguments.output.resolve()

    if arguments.child:
        child(arguments)
        return

    if not Path("projects/des_3pcf").is_dir():
        raise SystemExit("Run this driver from Cocoa/.")
    arguments.output.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # One fresh process per thread count; BLAS stays at one thread.
    # ----------------------------------------------------------------------
    for threads in arguments.threads:
        environment = dict(os.environ, OMP_NUM_THREADS=str(threads))
        for name in BLAS:
            environment[name] = "1"
        command = [sys.executable, "-B", str(Path(__file__).resolve()),
                   "--yaml", str(arguments.yaml),
                   "--output", str(arguments.output),
                   "--threads", *[str(n) for n in arguments.threads],
                   "--npoints", str(arguments.npoints),
                   "--seed", str(arguments.seed),
                   "--child", str(threads)]
        subprocess.run(args=command, env=environment, check=True)

    result = summary(arguments)
    for threads in arguments.threads:
        median = result["median_seconds"][threads]
        print(f"{threads:3d} threads: full {median['full']:.3f} s, "
              f"camb {median['camb']:.3f} s, other {median['other']:.3f} s")


if __name__ == "__main__":
    main()
