"""Run a small real Cobaya/CAMB/BiHalofit provider graph, without a survey.

This checks component requirements, units, provider background arrays and
cosmology-cache replacement. Its tiny bispectrum table is not an accuracy
choice for a likelihood. The likelihood is a validating consumer that
returns zero. No data vector, covariance, mask or timing study is involved.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def consume_baseline(_self):
    """Request products through the real provider, using no patched methods."""
    baseline = _self.provider.get_matter_bispectrum()
    background = _self.provider.get_matter_bispectrum_background()
    if baseline["background_z"].shape != background["z"].shape:
        raise ValueError("The provider returned different background axes.")
    if len(baseline["native"]["log_bispectrum"]) != 4**4:
        raise ValueError("The test consumer expected its declared small table.")
    return 0.0


def run(output, threads):
    """Resolve actual requirements and evaluate A, cached A, B, then A again."""
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OMP_THREAD_LIMIT"] = "6"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["COBAYA_NOMPI"] = "1"
    project = Path(__file__).resolve().parents[2]
    cocoa = project.parents[1]
    sys.path.insert(0, str(project / "scripts"))
    sys.path.insert(0, str(project.parents[1] / "external_modules/code/cosmolike_core"))
    import numpy as np
    from cobaya.model import get_model
    from theory.bihalofit import model as bihalofit, bihalofit as bihalofit_theory

    component = "bihalofit"
    settings = {
        "baseline_z": np.linspace(start=0.0, stop=1.5, num=4).tolist(),
        "baseline_logk": np.linspace(start=-4.0, stop=2.0, num=4).tolist(),
        "baseline_logu": np.linspace(start=-6.0, stop=0.0, num=4).tolist(),
        "baseline_v": np.linspace(start=0.0, stop=1.0, num=4).tolist(),
        "power_k": np.geomspace(start=1.0e-5, stop=1.0e3, num=96).tolist(),
        "fit_nz": 9,
        "table_boost": 2,
        "kmax_boltzmann": 5.0,
        "stop_at_error": True,
    }
    parameters = {
        "As": {"prior": {"min": 1.0e-9, "max": 3.0e-9}},
        "ns": 0.965,
        "H0": 67.5,
        "ombh2": 0.0224,
        "omch2": 0.12,
        "mnu": 0.06,
        "tau": 0.054,
    }
    info = {
        "theory": {
            "camb": {
                "path": str(cocoa / "external_modules" / "code" / "CAMB"),
                "extra_args": {
                    "num_massive_neutrinos": 1,
                    "nnu": 3.046,
                    "dark_energy_model": "ppf",
                    "k_per_logint": 5,
                },
            },
            component: settings,
        },
        "likelihood": {
            "baseline_consumer": {
                "external": consume_baseline,
                "requires": {
                    "matter_bispectrum": None,
                    "matter_bispectrum_background": None,
                },
            },
        },
        "params": parameters,
        "stop_at_error": True,
        "debug": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    records = []
    arrays = {}
    with get_model(info_or_yaml_or_file=info) as model:
        requirements = model.theory[component].get_requirements()
        assert requirements["Pk_interpolator"]["k_max"] == 5.0
        old = None
        first = None
        for label, amplitude in (("A", 2.1e-9), ("A_cached", 2.1e-9),
                                  ("B", 2.15e-9), ("A_restored", 2.1e-9)):
            loglike = model.loglike(params_values={"As": amplitude},
                                    return_derived=False)
            assert loglike == 0.0
            baseline = model.provider.get_matter_bispectrum()
            if label == "A":
                first = baseline
            elif label == "A_cached":
                assert baseline is old
            elif label == "B":
                assert baseline is not old
                assert not np.array_equal(baseline["native"]["log_bispectrum"],
                                          old["native"]["log_bispectrum"])
            else:
                np.testing.assert_array_equal(
                    baseline["native"]["log_bispectrum"],
                    first["native"]["log_bispectrum"])
            for name, value in baseline["native"].items():
                arrays[label + "__" + name] = value
            arrays[label + "__background_z"] = baseline["background_z"]
            arrays[label + "__omega_m_z"] = baseline["background_omega_m_z"]
            records.append({
                "label": label,
                "As": amplitude,
                "loglike": float(loglike),
                "baseline": baseline["metadata"],
            })
            old = baseline
        wave, redshift, power = model.theory["camb"].get_Pk_grid(
            var_pair=("delta_tot", "delta_tot"), nonlinear=False)
        assert wave[-1] < 10.0
        arrays["physical_power_k"] = wave
        arrays["physical_power_z"] = redshift
        arrays["physical_power"] = power
        camb_version = model.theory["camb"].camb.__version__
    array_path = output / "provider_arrays.npz"
    np.savez_compressed(file=array_path, **arrays)
    report = {
        "scope": "actual Cobaya provider graph, not a survey likelihood",
        "survey_data_covariance_mask": "none",
        "theory_settings": settings,
        "parameters": parameters,
        "requirements": requirements,
        "camb_version": camb_version,
        "actual_power_kmax_Mpc_inverse": float(wave[-1]),
        "source_hashes": {
            "bihalofit": hashlib.sha256(
                Path(bihalofit.__file__).read_bytes()).hexdigest(),
            "theory": hashlib.sha256(
                Path(bihalofit_theory.__file__).read_bytes()).hexdigest(),
        },
        "arrays_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
        "evaluations": records,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Actual Cobaya/CAMB/BiHalofit A/A/B/A provider check passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=range(1, 7), default=1)
    options = parser.parse_args()
    run(output=options.output, threads=options.threads)
