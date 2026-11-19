"""Gate supplied B in a small fixed joint two/three-point covariance metric.

The saved refinement study gives Delta chi2=0.0797234, 0.00622678 and
1.62025e-5 for its coarse, medium and fine tables. The permanent fine-grid
limit 5e-5 allows about three times that measured error; it is also much
stricter than the group's 0.2 route-comparison ceiling. This is a controlled
four-row integration regression, not a DES convergence declaration.
"""

from des3pcf.variance import reference_physics_model

import hashlib
import json
from pathlib import Path

import numpy as np

from des3pcf.bihalofit import BiHalofitProducer, input_hashes


REFERENCE_DIRECTORY = Path(__file__).with_name("reference_bihalofit")


def test_supplied_full_mean_in_fixed_joint_covariance(compiled, tmp_path):
    """Refinement reduces the full joint error and leaves two-point rows fixed.

    Input arrays, synthetic data, summed covariance and the all-one mask
    are committed binary fixtures. Their archive hashes are checked before
    they are read. Thus changing a helper fixture cannot silently change
    the scientific comparison or its metric. The full covariance includes
    its two/three-point cross block; no diagonal approximation, ridge,
    pseudoinverse or excluded row is permitted.
    """
    manifest_path = REFERENCE_DIRECTORY / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, record in manifest["files"].items():
        path = REFERENCE_DIRECTORY / name
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_hash == record["sha256"], name
    input_path = REFERENCE_DIRECTORY / "frozen_inputs.npz"
    measurement_path = REFERENCE_DIRECTORY / "frozen_measurement.npz"
    loaded = {}
    with np.load(file=input_path, allow_pickle=False) as saved:
        for key in saved.files:
            group, name = key.split("__", maxsplit=1)
            if group not in loaded:
                loaded[group] = {}
            value = saved[key]
            if value.ndim == 0:
                loaded[group][name] = value.item()
            else:
                loaded[group][name] = value.copy()
    native = loaded["native"]
    supplied = loaded["supplied"]
    covariance_input = loaded["covariance"]
    with np.load(file=measurement_path, allow_pickle=False) as saved:
        data = saved["data"].copy()
        covariance = saved["covariance"].copy()
        mask = saved["mask"].copy()
    np.testing.assert_array_equal(mask, np.ones(shape=4, dtype=np.int32))
    cholesky = np.linalg.cholesky(a=covariance)
    assert np.any(covariance[0, 1:] != 0.0)
    # The archived full vector is the independent reference. C no longer
    # contains a second BiHalofit evaluation. Keep its original covariance
    # and the stronger fine-grid 5e-5 budget below unchanged.
    two_point_reference = None
    report = {
        "row_order": ["Map3", "xi+", "xi-", "Map2"],
        "scope": "controlled integration fixture, not survey data",
        "inputs_npz_sha256": hashlib.sha256(
            input_path.read_bytes()).hexdigest(),
        "measurement_npz_sha256": hashlib.sha256(
            measurement_path.read_bytes()).hexdigest(),
        "native_extension_sha256": hashlib.sha256(
            Path(compiled.__file__).read_bytes()).hexdigest(),
        "measurement_array_sha256": input_hashes(
            arrays={"data": data, "covariance": covariance, "mask": mask},
            settings={}),
        "cases": [],
    }
    errors = []
    for counts in ((9, 33, 17, 9), (17, 65, 33, 17)):
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
        inputs["omega_m_z"] = baseline.background_omega_m_z
        inputs["baseline"] = dict(baseline.native)
        model = reference_physics_model(factory=compiled.production.PhysicsModelWithBispectrum, **inputs)
        mean, unused_terms = compiled.production.physical_covariance(
            model=model, inputs=covariance_input)
        if two_point_reference is None:
            two_point_reference = mean[1:].copy()
        np.testing.assert_array_equal(mean[1:], two_point_reference)
        residual = np.linalg.solve(a=cholesky, b=mean - data)
        error = float(residual @ residual)
        errors.append(error)
        report["cases"].append({
            "coarse_counts": list(counts),
            "delta_chi2": error,
            "baseline": baseline.record(),
        })
    (tmp_path / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
    assert errors[1] < errors[0]
    assert errors[1] < 5.0e-5
