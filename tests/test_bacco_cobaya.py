"""Check the registered CAMB/BACCO chain with the installed neural networks.

Small response grids test units, dependency tracking and state restoration;
they are not forecast accuracy settings. No notebook or sampler is run.
"""

from copy import deepcopy
from pathlib import Path
import shlex
import subprocess

import numpy as np
import pytest
import yaml
from cobaya.model import get_model

from theory.bacco_bispectrum.response import BaccoFeedback


PROJECT = Path(__file__).resolve().parents[1]
COCOA = PROJECT.parents[1]


def consume_response(_self):
    """Read the registered result without replacing any provider method."""
    response = _self.provider.get_result("baryon_response")
    assert np.all(np.isfinite(response["native"]["log_bispectrum"]))
    return 0.0


def response_configuration():
    """Declare a real provider graph with inexpensive response sampling."""
    return {
        "theory": {
            "camb": {
                "path": str(COCOA / "external_modules/code/CAMB"),
                "extra_args": {
                    "WantCls": False,
                    "halofit_version": "takahashi",
                    "dark_energy_model": "ppf",
                    "num_massive_neutrinos": 1,
                },
            },
            "bacco_bispectrum": {
                "background_nz": 8,
                "power_nk": 64,
                "zmax": 3.1,
                "response_settings": {
                    "nz": 4,
                    "nk": 9,
                    "nu": 5,
                    "nv": 4,
                    "boost": 1,
                },
            },
        },
        "likelihood": {
            "response_consumer": {
                "external": consume_response,
                "requires": {"baryon_response": None},
            },
        },
        "params": {
            "As": {"prior": {"min": 1.9e-9, "max": 2.3e-9}},
            "H0": {"prior": {"min": 66.0, "max": 69.0}},
            "mnu": {"prior": {"min": 0.0, "max": 0.15}},
            "ns": 0.965,
            "ombh2": 0.0224,
            "omch2": 0.12,
            "tau": 0.054,
            "M_c_baccoemu": {"prior": {"min": 13.5, "max": 14.5}},
            "eta_baccoemu": {"prior": {"min": -0.5, "max": 0.0}},
            "beta_baccoemu": {"prior": {"min": -0.5, "max": 0.0}},
            "M1_z0_cen_baccoemu": {"prior": {"min": 10.0, "max": 11.0}},
            "theta_inn_baccoemu": {"prior": {"min": -1.0, "max": -0.5}},
        },
        "stop_at_error": True,
    }


@pytest.fixture(scope="module")
def response_model():
    """Reuse the solved test graph, retaining Cobaya's ordinary cache behavior."""
    with get_model(info_or_yaml_or_file=response_configuration()) as model:
        yield model


def reference_point():
    """Choose interior cosmology and feedback coordinates for all displacements."""
    return {
        "As": 2.1e-9,
        "H0": 67.5,
        "mnu": 0.06,
        "M_c_baccoemu": 14.0,
        "eta_baccoemu": -0.3,
        "beta_baccoemu": -0.22,
        "M1_z0_cen_baccoemu": 10.5,
        "theta_inn_baccoemu": -0.86,
    }


@pytest.mark.parametrize("parameter,step", [
    ("M_c_baccoemu", 0.1),
    ("eta_baccoemu", 0.05),
    ("beta_baccoemu", 0.05),
    ("M1_z0_cen_baccoemu", 0.1),
    ("theta_inn_baccoemu", 0.05),
    ("As", 0.03e-9),
    ("H0", 0.5),
    ("mnu", 0.02),
])
def test_response_dependencies_and_restoration(response_model, parameter, step):
    """Every physical input changes P and B; A/B/A preserves retained arrays."""
    model = response_model
    point = reference_point()
    model.loglike(params_values=point, return_derived=False)
    first = model.provider.get_result("baryon_response")
    saved = deepcopy(first)
    camb_state = model.theory["camb"].current_state
    model.loglike(params_values=point, return_derived=False)
    assert model.provider.get_result("baryon_response") is first
    changed = dict(point)
    changed[parameter] += step
    model.loglike(params_values=changed, return_derived=False)
    second = model.provider.get_result("baryon_response")
    for name in ("log_power", "log_bispectrum"):
        assert not np.array_equal(first["native"][name], second["native"][name])
        np.testing.assert_array_equal(first["native"][name], saved["native"][name])
    if parameter.endswith("_baccoemu"):
        assert model.theory["camb"].current_state is camb_state
    model.loglike(params_values=point, return_derived=False, cached=False)
    restored = model.provider.get_result("baryon_response")
    for name in ("log_power", "log_bispectrum"):
        np.testing.assert_array_equal(restored["native"][name], saved["native"][name])


def test_response_matches_offline_preparation(response_model):
    """Independently convert the provider's spectra, cold sigma8 and mass fraction."""
    model = response_model
    model.loglike(params_values=reference_point(), return_derived=False)
    component = model.theory["bacco_bispectrum"]
    actual = model.provider.get_result("baryon_response")
    results = model.provider.get_CAMBdata()
    little_h = results.Params.H0 / 100.0
    parameters = {
        "omega_m": (results.Params.ombh2 + results.Params.omch2
                    + results.Params.omnuh2) / little_h**2,
        "omega_b": results.Params.ombh2 / little_h**2,
        "h": little_h,
        "sigma8": results.get_sigma8_0(),
        "ns": results.Params.InitPower.ns,
        "w0": results.Params.DarkEnergy.w,
        "wa": results.Params.DarkEnergy.wa,
    }
    for name, value in reference_point().items():
        if name.endswith("_baccoemu"):
            parameters[name] = value
    sigma_cb = results.get_sigmaR(
        R=8.0,
        z_indices=[len(results.Params.Transfer.PK_redshifts) - 1],
        var1="delta_nonu",
        var2="delta_nonu",
        hubble_units=True,
    )
    metadata = {
        "parameters": dict(parameters),
        "omega_nu_h2": results.Params.omnuh2,
        "sigma8_cb": float(np.asarray(sigma_cb).ravel()[0]),
    }
    assert metadata["sigma8_cb"] > parameters["sigma8"]
    tables = {"k": component.k, "z": component.z}
    for name, field, nonlinear in (
        ("p_nonlinear", "delta_tot", True),
        ("p_linear_nu", "delta_nu", False),
    ):
        interpolator = model.provider.get_Pk_interpolator(
            var_pair=(field, field),
            nonlinear=nonlinear,
            extrap_kmin=little_h * component.k[0],
            extrap_kmax=1.01 * little_h * component.k[-1],
        )
        tables[name] = interpolator.P(
            z=component.z,
            k=little_h * component.k,
        ) * little_h**3
    expected = BaccoFeedback(settings=component.response_settings).prepare(
        parameters=parameters,
        tables=tables,
        metadata=metadata,
    )
    for name in ("log_power", "log_bispectrum"):
        np.testing.assert_allclose(
            actual=actual["native"][name],
            desired=expected.native[name],
            rtol=1.0e-12,
            atol=1.0e-13,
        )


def test_massless_neutrinos(response_model):
    """The massless branch must not take the logarithm of zero neutrino power."""
    point = reference_point()
    point["mnu"] = 0.0
    response_model.loglike(params_values=point, return_derived=False)
    response = response_model.provider.get_result("baryon_response")
    assert response["metadata"]["neutrino_fraction"] == 0.0
    assert np.all(np.isfinite(response["native"]["log_power"]))


@pytest.mark.parametrize("number", [1, 2, 3, 4])
def test_numbered_bacco_instructions(number):
    """Uncommenting the taught options preserves matched sampler science."""
    enabled_cases = []
    for family in ("EVALUATE", "MCMC", "NAUTILUS"):
        path = PROJECT / f"EXAMPLE_{family}{number}.yaml"
        lines = path.read_text(encoding="utf-8").splitlines()
        enabled = []
        theory_block = False
        for line in lines:
            if line.startswith("  # bacco_bispectrum:"):
                theory_block = True
            if theory_block and not line.startswith("  #"):
                theory_block = False
            parameter = any(line.startswith("  # " + name + ":") for name in (
                "M_c_baccoemu", "eta_baccoemu", "beta_baccoemu",
                "M1_z0_cen_baccoemu", "theta_inn_baccoemu",
            ))
            if theory_block or parameter or "# external_baryon_response:" in line:
                line = line.replace("# ", "", 1)
            enabled.append(line)
        info = yaml.safe_load("\n".join(enabled))
        assert set(info["theory"]) == {"camb", "bihalofit", "bacco_bispectrum"}
        likelihood = info["likelihood"]["des_3pcf.joint_shear"]
        assert likelihood["external_baryon_response"]
        assert likelihood["data_file"] == f"des_y3_joint{number}.dataset"
        assert info["params"]["M_c_baccoemu"] == 14.0
        enabled_cases.append(info)
    for case in enabled_cases[1:]:
        for section in ("theory", "likelihood", "params", "prior"):
            assert case.get(section) == enabled_cases[0].get(section)


@pytest.mark.parametrize("disabled", [False, True])
def test_cocoa_registration_and_cleanup(tmp_path, disabled):
    """Source the actual new start/stop blocks in an isolated Cocoa tree."""
    theory = tmp_path / "cobaya/cobaya/theories"
    theory.mkdir(parents=True)
    target = tmp_path / "external_modules/code/cosmolike_core/theory/bacco_bispectrum"
    target.mkdir(parents=True)
    blocks = []
    for name in ("start_cocoa.sh", "stop_cocoa.sh"):
        text = (COCOA / name).read_text(encoding="utf-8")
        start = text.index("# --------------------------- BACCO Bispectrum")
        start = text.index("if [[", start)
        end = text.index("\nfi", start) + len("\nfi")
        blocks.append(text[start:end])
    flag = "1" if disabled else ""
    script = (
        f"ROOTDIR={shlex.quote(str(tmp_path))}\n"
        "OUT1=/dev/null\nOUT2=/dev/null\nEC34=link_failure\n"
        f"IGNORE_BACCO_BISPECTRUM_CODE={shlex.quote(flag)}\n"
        + blocks[0] + "\n" + blocks[0] + "\n"
    )
    link = shlex.quote(str(theory / "bacco_bispectrum"))
    script += f"test {'!' if disabled else ''} -L {link} || exit 1\n"
    script += blocks[1] + "\n" + blocks[1] + "\n"
    script += f"test ! -L {link} || exit 1\n"
    cleanup = shlex.quote(str(COCOA / "installation_scripts/flags_impl_unset_keys.sh"))
    script += f"source {cleanup}\n"
    script += 'test -z "${IGNORE_BACCO_BISPECTRUM_CODE+x}"\n'
    result = subprocess.run(
        args=["bash", "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
