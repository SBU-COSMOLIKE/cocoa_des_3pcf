"""Check Fisher-derivative convergence in the Legendre cutoff map_lmax.

The Map3 aperture transform expands the bispectrum in Legendre multipoles
up to map_lmax (default 20; the published theory used 50). At the
fiducial the mean moves by at most 0.26 % between 20 and 50
(tests/test_map_multipole_truncation.py), but a Fisher forecast uses
derivatives, which can converge differently. This study recomputes the
derivatives of notes/intrinsic_alignments/EXAMPLE_FISHER_IA.ipynb with
map_lmax = 50 and compares them with that notebook's saved map_lmax = 20
results (chains/fisher_ia/{nla,tatt}.npz):

  - each derivative column's change in measurement-error units,
    |L^-1 (D50 - D20)| / |L^-1 D20| with C = L L^T;
  - the marginalized error of every parameter, sigma(S8), FoM(w, wa)
    and FoM(Omega_m, S8), with the notebook's Gaussian priors.

The steps, priors, fiducials and opened dark energy are read from the
notebook's own cells, so both cutoffs use exactly the same settings.
Run from activated Cocoa (a few hours):

    python projects/des_3pcf/tests/studies/fisher_lmax_convergence.py

Writes chains/fisher_lmax50/{nla,tatt}.npz and prints the comparison.
"""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_triangular

PROJECT = Path(__file__).resolve().parents[2]
COCOA = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / "interface"))
sys.path.insert(0, str(COCOA / "external_modules/code/cosmolike_core"))
import cosmolike_des_3pcf_interface as ci  # noqa: E402
import cosmolike_notebook_utils.fisher as cnu  # noqa: E402
from cobaya.model import get_model  # noqa: E402
from cobaya.yaml import yaml_load_file  # noqa: E402

LMAX = 50
THREADS = 6
NOTEBOOK = PROJECT / "notes/intrinsic_alignments/EXAMPLE_FISHER_IA.ipynb"


def notebook_settings():
    """Return the notebook's steps and priors (its cell that defines them)."""
    cells = json.load(open(NOTEBOOK, encoding="utf-8"))["cells"]
    for cell in cells:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.lstrip().startswith("# These are absolute physical"):
            namespace = {}
            exec(source, namespace)
            return namespace["steps"], namespace["priors"]
    raise RuntimeError("The notebook's steps cell was not found.")


def derivatives(example, steps):
    """Five-point derivatives of the selected data vector at map_lmax = LMAX."""
    info = yaml_load_file(str(PROJECT / example))
    fiducial = dict(info["sampler"]["evaluate"]["override"])
    info.pop("sampler")
    info.pop("output", None)
    info["theory"]["camb"]["path"] = str(COCOA / "external_modules/code/CAMB")
    settings = info["likelihood"]["des_3pcf.joint_shear"]
    settings["omp_threads"] = THREADS
    defaults = yaml_load_file(str(PROJECT / "likelihood/joint_shear.yaml"))
    settings["map_lmax"] = LMAX
    settings["projection_settings"] = dict(defaults["projection_settings"], lmax=LMAX)
    # As in the notebook: dark energy opened for the stencil coordinates.
    info["params"]["w"] = {"prior": {"min": -2.0, "max": -0.3}, "ref": -1.0}
    info["params"]["wa"] = {"prior": {"min": -2.0, "max": 2.0}, "ref": 0.0}
    fiducial["w"] = -1.0
    fiducial["wa"] = 0.0
    with get_model(info_or_yaml_or_file=info) as model:
        likelihood = model.likelihood["des_3pcf.joint_shear"]

        def get_dv(parameters):
            # cached=False: last_datavector must belong to this point.
            value = model.loglike(params_values=parameters, return_derived=False, cached=False)
            if not np.isfinite(value):
                raise RuntimeError(f"Nonfinite likelihood at {parameters}.")
            return np.array(ci.select_joint_data_vector(prediction=likelihood.last_datavector),
                            dtype=np.float64, copy=True)

        names, center, dh, dh2 = cnu.differentiate(predict=get_dv, parameters=fiducial, steps=steps)
    return names, center, dh, dh2


def errors(fisher, names, gradient):
    """Marginalized errors, sigma(S8), FoM(w, wa) and FoM(Omega_m, S8)."""
    covariance = cnu.parameter_covariance(fisher=fisher, names=names)
    sigma = np.sqrt(np.diag(covariance))
    s8 = float(np.sqrt(gradient @ covariance @ gradient))
    jacobian = np.zeros((2, len(names)))
    jacobian[0, names.index("omegam")] = 1.0
    jacobian[1] = gradient
    amplitude = jacobian @ covariance @ jacobian.T
    return sigma, s8, cnu.get_FoM(i=names.index("w"), j=names.index("wa"), F=fisher), \
        1.0 / np.sqrt(np.linalg.det(amplitude))


def main():
    steps, priors = notebook_settings()
    output = PROJECT / "chains/fisher_lmax50"
    output.mkdir(parents=True, exist_ok=True)
    for model, example in (("nla", "EXAMPLE_EVALUATE3.yaml"), ("tatt", "EXAMPLE_EVALUATE4.yaml")):
        case_steps = dict(steps)
        if model == "tatt":
            case_steps.update(DES_A2_1=0.02, DES_A2_2=0.03, DES_BTA_1=0.02)

        # Step 1: the map_lmax = 20 reference the notebook saved.
        reference = np.load(PROJECT / f"chains/fisher_ia/{model}.npz")
        names20 = [str(name) for name in reference["names"]]
        if names20 != list(case_steps) or not np.array_equal(reference["steps"], list(case_steps.values())):
            raise RuntimeError(f"{model}: the saved reference used other steps; rerun EXAMPLE_FISHER_IA.")
        covariance = reference["covariance"]

        # Step 2: the same derivatives at map_lmax = LMAX.
        names, center, dh, dh2 = derivatives(example, case_steps)
        assert names == names20
        np.savez(output / f"{model}.npz", names=np.asarray(names), center=center,
                 derivatives_h=dh, derivatives_h2=dh2, map_lmax=LMAX)

        # Step 3: column changes in measurement-error units.
        factor = np.linalg.cholesky(covariance)
        d20 = reference["derivatives_h2"]
        change = np.linalg.norm(solve_triangular(factor, dh2 - d20, lower=True), axis=0)
        size = np.linalg.norm(solve_triangular(factor, d20, lower=True), axis=0)
        print(f"\n=== {model}: map_lmax 20 -> {LMAX}; mean max |dMap3/Map3| = "
              f"{np.max(np.abs(center - reference['mean'])[400:] / np.abs(reference['mean'][400:])):.3e}")
        for name, c, s in zip(names, change, size):
            print(f"  {name:10s} derivative change {100 * c / s:8.4f} %")

        # Step 4: forecasts with the notebook's priors.
        gradient = reference["s8_gradient"]
        f20 = cnu.fisher_matrix(derivatives=d20, names=names, covariance=covariance, priors=priors)
        f50 = cnu.fisher_matrix(derivatives=dh2, names=names, covariance=covariance, priors=priors)
        s20, s8_20, fom20, amp20 = errors(f20, names, gradient)
        s50, s8_50, fom50, amp50 = errors(f50, names, gradient)
        print(f"  sigma(S8)        {s8_20:.6e} -> {s8_50:.6e}  ratio {s8_50 / s8_20:.6f}")
        print(f"  FoM(w, wa)       {fom20:.6e} -> {fom50:.6e}  ratio {fom50 / fom20:.6f}")
        print(f"  FoM(Omega_m, S8) {amp20:.6e} -> {amp50:.6e}  ratio {amp50 / amp20:.6f}")
        worst = int(np.argmax(np.abs(s50 / s20 - 1)))
        print(f"  largest marginalized-error change: {names[worst]} ratio {s50[worst] / s20[worst]:.6f}")
        for name, a, b in zip(names, s20, s50):
            print(f"  sigma({name}) ratio {b / a:.6f}")


if __name__ == "__main__":
    main()
