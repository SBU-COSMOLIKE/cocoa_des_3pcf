"""Check the real YAML theory graph and restored NLA/TATT predictions.

Run from activated Cocoa. Both examples retain their production sampling.
This check starts no sampler and writes its record under chains/. It reads
fixed synthetic inputs prepared before the provider refactor; they are a
regression reference, not a convergence or survey-validation criterion.
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import yaml
from cobaya.model import get_model

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "interface"))
import cosmolike_des_3pcf_interface as ci


def main():
    """Compare both IA choices and vary cosmology, IA and source shifts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "chains/cobaya_joint_check.txt")
    args = parser.parse_args()
    records = []
    for number, mode in ((2, "NLA"), (3, "TATT")):
        info = yaml.safe_load(
            (PROJECT / f"EXAMPLE_NAUTILUS{number}.yaml").read_text(),
        )
        # The evaluate override is the fiducial the synthetic data were
        # drawn at; it holds exactly the sampled parameters.
        point = dict(info["sampler"]["evaluate"]["override"])
        info.pop("sampler", None)
        info["likelihood"]["des_3pcf.joint_shear"]["omp_threads"] = args.threads
        with get_model(info_or_yaml_or_file=info) as model:
            likelihood = model.likelihood["des_3pcf.joint_shear"]
            score = model.logposterior(params_values=point)
            first = likelihood.get_datavector(**point)
            chi2 = ci.compute_joint_chi2(prediction=first)
            if not np.isfinite(score.logpost) or chi2 > 1.0e-6:
                raise AssertionError(f"{mode}: reference discrepancy chi2={chi2}.")
            records.append(f"{mode} reference delta_chi2 = {chi2:.17e}")
            for parameter, step in (("DES_A1_1", 0.01), ("DES_DZ_S1", 0.001),
                                    ("omegam", 0.003)):
                displaced = dict(point)
                displaced[parameter] += step
                changed_score = model.logposterior(params_values=displaced)
                changed = likelihood.get_datavector(**displaced)
                model.logposterior(params_values=point)
                restored = likelihood.get_datavector(**point)
                if (not np.isfinite(changed_score.logpost)
                        or np.array_equal(first, changed)
                        or not np.array_equal(first, restored)):
                    raise AssertionError(f"{mode}: failed {parameter} A/B/A sequence.")
                records.append(f"{mode} {parameter}: changed; restored bitwise")
        print("\n".join(records[-4:]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(records) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
