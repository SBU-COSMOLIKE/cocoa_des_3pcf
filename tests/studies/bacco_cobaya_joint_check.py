"""Validate the BACCO example through the actual DES joint likelihood.

Run from activated Cocoa: python projects/des_3pcf/tests/studies/bacco_cobaya_joint_check.py
The YAML's production sampling is retained. This is an integration/state
check, not a forecast, sampler run or numerical convergence measurement.
The plain-text record is written under chains/; notebooks remain untouched.
"""

from pathlib import Path
import sys

import numpy as np
import yaml
from cobaya.model import get_model

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "interface"))
import cosmolike_des_3pcf_interface as ci


def main():
    """Check both probes, parameter restoration and the native power input."""
    # Enable the same commented options a user enables in numbered case 2.
    lines = (PROJECT / "EXAMPLE_EVALUATE3.yaml").read_text(encoding="utf-8").splitlines()
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
    # The evaluate override is the example's fiducial point.
    point = dict(info["sampler"]["evaluate"]["override"])
    info.pop("sampler")
    info.pop("output")
    info["likelihood"]["des_3pcf.joint_shear"]["omp_threads"] = 6
    # Sample one feedback coordinate only to exercise invalidation. Its
    # small interior range is a test choice, not a recommended survey prior.
    # The amplitude is already sampled (As_1e9; sigma8 is derived).
    info["params"]["eta_baccoemu"] = {"prior": {"min": -0.5, "max": 0.0}}
    point["eta_baccoemu"] = -0.3
    records = []
    with get_model(info_or_yaml_or_file=info) as model:
        likelihood = model.likelihood["des_3pcf.joint_shear"]
        for mode in ("NLA", "TATT"):
            point["DES_A2_1"] = 0.0 if mode == "NLA" else 0.65
            point["DES_BTA_1"] = 0.0 if mode == "NLA" else 0.91
            score = model.logposterior(params_values=point)
            assert np.isfinite(score.logpost)
            first = likelihood.get_datavector(**point)
            np.testing.assert_array_equal(first, likelihood.get_datavector(**point))
            records.append(f"{mode} loglike = {score.loglikes[0]:.17g}")
            baseline = model.provider.get_result("matter_bispectrum")
            saved_baseline = baseline["native"]["log_bispectrum"].copy()
            for parameter, step in (
                ("eta_baccoemu", 0.05),
                ("As_1e9", 0.01),
                ("DES_DZ_S1", 0.001),
            ):
                displaced = dict(point)
                displaced[parameter] += step
                changed_score = model.logposterior(params_values=displaced)
                assert np.isfinite(changed_score.logpost)
                changed = likelihood.get_datavector(**displaced)
                assert not np.array_equal(
                    first[likelihood.xi_indices], changed[likelihood.xi_indices],
                )
                assert not np.array_equal(
                    first[likelihood.map_indices], changed[likelihood.map_indices],
                )
                if parameter == "eta_baccoemu":
                    assert model.provider.get_result("matter_bispectrum") is baseline
                    np.testing.assert_array_equal(
                        baseline["native"]["log_bispectrum"], saved_baseline,
                    )
                model.logposterior(params_values=point)
                restored = likelihood.get_datavector(**point)
                np.testing.assert_array_equal(restored, first)
                records.append(f"{mode} {parameter}: both probes change; restore bitwise")

            # Reinstall the independently constructed single-response P.
            # Its xi must equal the likelihood's xi. A deliberately squared
            # response must differ, proving this check can detect the error.
            wave, redshift = np.meshgrid(
                10.0**likelihood.log10k_interp_2D,
                likelihood.z_interp_2D,
            )
            ratio = likelihood.model.power_response(
                k=wave.ravel(),
                z=redshift.ravel(),
                amplitude=1.0,
            ).reshape(likelihood.p_nonlinear.shape)
            correct, odd = ci.shear_power_rows(
                ell=likelihood.xi_ell,
                bins=likelihood.xi_rows.bins[:, :2],
            )
            for exponent in (1, 2):
                power = likelihood.p_nonlinear * ratio**exponent
                ci.set_non_linear_power_spectrum(
                    log10k=likelihood.log10k_interp_2D,
                    z=likelihood.z_interp_2D,
                    lnP=np.log(power).flatten(order="F"),
                )
                even, _ = ci.shear_power_rows(
                    ell=likelihood.xi_ell,
                    bins=likelihood.xi_rows.bins[:, :2],
                )
                if exponent == 1:
                    np.testing.assert_array_equal(even, correct)
                else:
                    assert not np.array_equal(even, correct)
            np.testing.assert_array_equal(first, likelihood.get_datavector(**point))
            records.append(f"{mode} power response: applied once; squared response differs")
            print("\n".join(records[-5:]), flush=True)
    output = PROJECT / "chains/bacco_cobaya_joint_check.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(records) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
