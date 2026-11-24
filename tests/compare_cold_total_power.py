"""Measure the error from using total-matter power for the cold field.

Run this file first with --mode exact and then --mode total, using the
same output directory. Each command starts a fresh Cobaya/native state.
Only requests for linear delta_nonu power are replaced; background,
neutrino density, total power, nuisance parameters and covariance stay
fixed. The reported distance is delta_model.T @ precision @ delta_model,
not a difference between two fits to the observed data.
"""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["COBAYA_NOMPI"] = "1"
project = Path(__file__).resolve().parents[1]
root = project.parents[1]
sys.path.insert(0, str(root / "cobaya"))
sys.path.insert(0, str(root / "external_modules/code/cosmolike_core"))
sys.path.insert(0, str(project / "interface"))

import numpy as np
from cobaya.model import get_model
from cobaya.yaml import yaml_load_file, yaml_dump
import cosmolike_des_3pcf_interface as ci

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["exact", "total"], required=True)
    parser.add_argument("--configuration", default="EXAMPLE_EVALUATE1.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    info = yaml_load_file(str(project / args.configuration))
    fiducial = dict(info.pop("sampler")["evaluate"]["override"])
    info.pop("output", None)
    info["params"]["mnu"] = {"value": 0.06}
    fiducial.pop("mnu", None)
    # The exact same physical calculation feeds both routes. Refining
    # CAMB avoids confusing a spectrum-computation error with this test.
    camb = info["theory"]["camb"]["extra_args"]
    camb["AccuracyBoost"] = 3.0
    camb["k_per_logint"] = 50
    name = next(iter(info["likelihood"]))
    configuration = yaml_dump(info)
    if args.mode == "total":
        assert configuration == (output / "exact_configuration.yaml").read_text()
    (output / (args.mode + "_configuration.yaml")).write_text(configuration)
    points = [fiducial, dict(fiducial, omegam=0.25), dict(fiducial, omegam=0.40)]
    np.savetxt(output / "points.txt", [[p[k] for k in fiducial] for p in points],
               header=" ".join(fiducial))
    model = get_model(info)
    likelihood = model.likelihood[name]
    original = model.provider.get_Pk_interpolator
    cold_requests = []

    def read_power(var_pair=("delta_tot", "delta_tot"), **kwargs):
        # This test-only substitution intercepts the provider request,
        # not the likelihood or native implementation. A new process for
        # each route prevents a cached native table hiding the change.
        if tuple(var_pair) == ("delta_nonu", "delta_nonu"):
            assert kwargs.get("nonlinear", False) is False
            cold_requests.append(args.mode)
            if args.mode == "total":
                var_pair = ("delta_tot", "delta_tot")
        return original(var_pair=var_pair, **kwargs)

    model.provider.get_Pk_interpolator = read_power
    vectors = []
    full_vectors = []
    for index, point in enumerate(points):
        posterior = model.logposterior(params_values=point, cached=False)
        if not np.isfinite(posterior.loglike):
            raise ValueError("Likelihood failed; no previous vector may be reused.")
        full = np.array(likelihood.last_datavector, copy=True)
        vector = np.asarray(ci.select_joint_data_vector(prediction=full))
        precision = np.linalg.inv(ci.get_joint_covariance())
        assert precision.shape == (len(vector), len(vector))
        assert np.all(np.isfinite(vector))
        full_vectors.append(full.copy())
        vectors.append(vector.copy())
        # Inspect the two true CAMB spectra even in the substitution run.
        # These k values are physical 1/Mpc, with no h rescaling.
        k = np.geomspace(1e-4, float(camb["kmax"]), 301)
        z = np.array([0.0, 0.5, 1.0, 2.0])
        cold = original(var_pair=("delta_nonu", "delta_nonu"), nonlinear=False)
        total = original(var_pair=("delta_tot", "delta_tot"), nonlinear=False)
        ratio = total.P(z, k) / cold.P(z, k) - 1.0
        np.savetxt(output / (args.mode + f"_power_{index}.txt"),
                   np.column_stack((k, ratio.T)), header="k_per_Mpc Pmm_over_Pcb_minus1_z0 z0p5 z1 z2")
        fraction = float(model.provider.get_Omega_nu_massive(z=np.array([0.0]))[0]) / point["omegam"]
        print(args.mode, index, "f_nu", fraction,
              "power_fraction_range", ratio.min(), ratio.max(), flush=True)
    assert cold_requests, "The test must actually exercise a cold-power request."
    np.save(output / (args.mode + "_vectors.npy"), vectors)
    np.save(output / (args.mode + "_full_vectors.npy"), full_vectors)
    if args.mode == "exact":
        np.save(output / "precision.npy", precision)
    else:
        np.testing.assert_array_equal(precision, np.load(output / "precision.npy"))
        reference = np.load(output / "exact_vectors.npy")
        difference = np.asarray(vectors) - reference
        distances = np.einsum("ni,ij,nj->n", difference, precision, difference)
        np.savetxt(output / "distances.txt", distances, header="direct_vector_delta_chi2")
        print("Direct-vector distances:", distances, flush=True)
    native = Path(ci.__file__)
    record = ["mode " + args.mode, "cold_requests " + str(len(cold_requests)),
              "native_sha256 " + hashlib.sha256(native.read_bytes()).hexdigest()]
    for repo in [project, root / "external_modules/code/cosmolike_core", root / "external_modules/code/CAMB"]:
        revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        record.append(str(repo) + " " + revision)
    (output / (args.mode + "_provenance.txt")).write_text("\n".join(record) + "\n")
    model.close()
