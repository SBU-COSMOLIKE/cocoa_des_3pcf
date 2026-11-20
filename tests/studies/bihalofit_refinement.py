"""Measure supplied-table errors in a small joint two/three-point example.

This driver uses controlled positive power and source curves from the existing
physical tests. Its four rows are Map3, xi+, xi- and Map2. The complete sum of
the six covariance terms defines one fixed metric for every comparison;
discarding the two/three-point cross block would change that metric.

The quadratures are deliberately small. This is an integration regression,
not a DES accuracy or likelihood result. No survey data or mask is loaded:
the archived internal-model mean is the synthetic data and every row is retained.
All inputs are written before the first calculation and then reloaded.

Example (one command):
  python tests/studies/bihalofit_refinement.py --extension /path/to/interface \
      --output /path/to/new/results --threads 6
"""

from theory.bihalofit.variance import reference_physics_model

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def arguments():
    """Require an explicit extension, output location and worker limit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1, choices=range(1, 7))
    return parser.parse_args()


def write_json(path, record):
    """Save finite JSON values with visible indentation and a final newline."""
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def pin_inputs(path, dictionaries, numpy):
    """Save and reload inputs, preserving scalar/array distinctions."""
    saved = {}
    for group, mapping in dictionaries.items():
        for name, value in mapping.items():
            saved[group + "__" + name] = value
    numpy.savez(file=path, **saved)
    restored = {}
    with numpy.load(file=path, allow_pickle=False) as loaded:
        for group, mapping in dictionaries.items():
            restored[group] = {}
            for name, value in mapping.items():
                array = loaded[group + "__" + name]
                if numpy.isscalar(value):
                    restored[group][name] = array.item()
                else:
                    restored[group][name] = array.copy()
    return restored


def run(options):
    """Separate table interpolation from fit quadrature/root convergence."""
    # Select numerical libraries' worker counts before importing NumPy or C.
    # BLAS is serial; the requested workers belong to native OpenMP loops.
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["OMP_NUM_THREADS"] = str(options.threads)
    os.environ["OMP_THREAD_LIMIT"] = "6"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project / "scripts"))
    sys.path.insert(0, str(project.parents[1] / "external_modules/code/cosmolike_core"))
    sys.path.insert(0, str(project / "tests" / "data_vector"))
    sys.path.insert(0, str(project / "tests" / "covariance"))
    sys.path.insert(0, str(options.extension.resolve()))
    import numpy as np
    import cosmolike_des_3pcf_interface as compiled
    from theory.bihalofit import model as bihalofit
    from theory.bihalofit.model import BiHalofit, BiHalofitProducer, input_hashes
    from test_bihalofit_python import comparison_inputs, query_triangles
    from test_physical_covariance import covariance_arguments

    compiled.set_omp_threads(nthreads=options.threads)
    options.output.mkdir(parents=True, exist_ok=False)
    # Align the old fit anchors with the background nodes. Supplying the
    # exact same Omega samples then isolates B interpolation from any
    # different background sampling in the halo covariance.
    native, supplied = comparison_inputs(source_readers=False, fit_nz=41)
    native["los_z"] = np.asarray([0.15, 0.5, 1.0, 1.8])
    covariance_input = covariance_arguments()
    query_sides, query_z = query_triangles()
    all_inputs = {
        "native": native,
        "supplied": supplied,
        "covariance": covariance_input,
        "query": {"sides": query_sides, "z": query_z},
    }
    input_path = options.output / "frozen_inputs.npz"
    restored = pin_inputs(path=input_path, dictionaries=all_inputs, numpy=np)
    native = restored["native"]
    supplied = restored["supplied"]
    covariance_input = restored["covariance"]
    query_sides = restored["query"]["sides"]
    query_z = restored["query"]["z"]
    # The internal C fit is retired. Retain its committed synthetic
    # measurement and metric, checking every input before reusing them.
    archive = Path(__file__).resolve().parents[1] / "data_vector/reference_bihalofit"
    with np.load(archive / "frozen_inputs.npz", allow_pickle=False) as saved:
        with np.load(input_path, allow_pickle=False) as current:
            if set(saved.files) != set(current.files):
                raise ValueError("The archived reference has a different input schema.")
            for key in saved.files:
                np.testing.assert_array_equal(current[key], saved[key])
    with np.load(archive / "frozen_measurement.npz", allow_pickle=False) as saved:
        reference_mean = saved["data"].copy()
        covariance_terms = saved["terms"].copy()
    fixed_covariance = np.sum(covariance_terms, axis=0)
    measurement_path = options.output / "frozen_measurement.npz"
    mask = np.ones(shape=reference_mean.size, dtype=np.int32)
    np.savez(file=measurement_path, data=reference_mean,
             covariance=fixed_covariance, terms=covariance_terms, mask=mask)
    with np.load(file=measurement_path, allow_pickle=False) as saved:
        reference_mean = saved["data"].copy()
        fixed_covariance = saved["covariance"].copy()
        mask = saved["mask"].copy()
    # This factorization is also a check that the actual joint metric is
    # positive definite. No ridge, discarded eigenmode or diagonal fallback
    # may hide a failed covariance in this study.
    cholesky = np.linalg.cholesky(a=fixed_covariance)
    report = {
        "scope": "controlled four-row integration regression, not DES",
        "row_order": ["Map3", "xi+", "xi-", "Map2"],
        "survey_data": "none; frozen internal mean is synthetic data",
        "input_npz_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "measurement_npz_sha256": hashlib.sha256(
            measurement_path.read_bytes()).hexdigest(),
        "measurement_array_sha256": input_hashes(
            arrays={"data": reference_mean, "covariance": fixed_covariance,
                    "mask": mask}, settings={}),
        "extension_sha256": hashlib.sha256(
            Path(compiled.__file__).read_bytes()).hexdigest(),
        "producer_source_sha256": hashlib.sha256(
            Path(bihalofit.__file__).read_bytes()).hexdigest(),
        "equation_source_sha256": bihalofit.C_SOURCE_SHA256,
        "threads": options.threads,
        "reference_mean": reference_mean.tolist(),
        "joint_covariance": fixed_covariance.tolist(),
        "table_cases": [],
        "equation_refinements": [],
    }
    previous_mean = None
    # The domain covers all triangles reached by this fixed small quadrature.
    # Refining these axes changes only the supplied-table approximation.
    for name, counts in (("coarse", (5, 17, 9, 5)),
                         ("medium", (9, 33, 17, 9)),
                         ("fine", (17, 65, 33, 17))):
        producer = BiHalofitProducer(
            z=np.linspace(start=0.0, stop=2.0, num=counts[0]),
            logk=np.linspace(start=-8.0, stop=5.0, num=counts[1]),
            logu=np.linspace(start=-9.0, stop=0.0, num=counts[2]),
            v=np.linspace(start=0.0, stop=1.0, num=counts[3]),
            boost=2, domain="error")
        baseline = producer.prepare(tables=supplied, fit_nz=41, table_boost=8)
        inputs = dict(native)
        for unused in ("omega_de", "w0", "wa", "ns"):
            del inputs[unused]
        inputs["omega_m_z"] = baseline["background_omega_m_z"]
        inputs["baseline"] = dict(baseline["native"])
        model = reference_physics_model(factory=compiled.production.PhysicsModelWithBispectrum, **inputs)
        mean, terms = compiled.production.physical_covariance(
            model=model, inputs=covariance_input)
        np.testing.assert_array_equal(mean[1:], reference_mean[1:])
        residual = mean - reference_mean
        whitened = np.linalg.solve(a=cholesky, b=residual)
        weighted_error = float(whitened @ whitened)
        case = {
            "name": name,
            "coarse_counts": list(counts),
            "dense_boost": 2,
            "delta_chi2_internal": weighted_error,
            "mean": mean.tolist(),
            "baseline": baseline["metadata"],
            "maximum_covariance_scaled_change": float(np.max(
                np.abs(np.sum(terms, axis=0) - fixed_covariance)
                / np.sqrt(np.outer(np.diag(fixed_covariance),
                                   np.diag(fixed_covariance))))),
        }
        if previous_mean is not None:
            step = np.linalg.solve(a=cholesky, b=mean - previous_mean)
            case["delta_chi2_previous"] = float(step @ step)
        previous_mean = mean.copy()
        report["table_cases"].append(case)
        np.savez(file=options.output / (name + "_outputs.npz"),
                 mean=mean, covariance_terms=terms)
        write_json(path=options.output / "report.json", record=report)
        print(name, "delta_chi2_internal", weighted_error, flush=True)

    for name, nodes, slope_nodes, tolerance in (
            ("legacy", 1000, 2000, 1.0e-5),
            ("refined", 2001, 4001, 1.0e-8),
            ("tight", 4001, 8001, 1.0e-10)):
        equation = BiHalofit(
            **supplied, fit_nz=41, variance_nodes=nodes,
            slope_nodes=slope_nodes,
            radius_rtol=tolerance)
        values = equation.evaluate(k1=query_sides[:, 0], k2=query_sides[:, 1],
                                   k3=query_sides[:, 2], z=query_z)
        case = {
            "name": name,
            "variance_nodes": nodes,
            "slope_nodes": slope_nodes,
            "radius_rtol": tolerance,
            "full_B": values.tolist(),
            "radius_anchors": equation.radius_anchors.tolist(),
            "neff_anchors": equation.neff_anchors.tolist(),
        }
        report["equation_refinements"].append(case)
    write_json(path=options.output / "report.json", record=report)


if __name__ == "__main__":
    run(options=arguments())
