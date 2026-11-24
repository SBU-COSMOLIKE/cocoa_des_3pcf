"""Check that the synthetic data vectors are the current model's mean.

Cases 3 (NLA) and 4 (TATT) score synthetic vectors written by
scripts/make_synthetic_datavectors.py: the model's mean at each example
yaml's fiducial point. If the theory, the row geometry or the likelihood
defaults change without rerunning that script, the vectors go out of date
and the fiducial chi2 is no longer zero (it was 0.076 after the side-bin
change). This test catches that.
"""

from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
COCOA = PROJECT.parents[1]

# chi2 of the vector against itself. Case 3 reproduces it to 1e-27. The
# TATT case differs between processes at the 1e-11 level: CAMB's linear
# P(k, z) changes in the last digits (about 1e-14) from one process to the
# next, and the FAST-PT one-loop IA tables magnify that change where their
# values are far below their peak (high and low k). The thread count plays
# no part. The bound sits far above that and far below any size that
# matters for an analysis.
CHI2_BOUND = 1.0e-8


@pytest.mark.parametrize("case", [3, 4])
def test_synthetic_vector_gives_zero_chi2_at_its_fiducial(case):
    """Evaluate the case at its fiducial point; chi2 must vanish."""
    from cobaya.model import get_model
    from cobaya.yaml import yaml_load_file

    import cosmolike_des_3pcf_interface as ci

    # Step 1: the example's model and fiducial point.
    info = yaml_load_file(str(PROJECT / f"EXAMPLE_EVALUATE{case}.yaml"))
    point = dict(info["sampler"]["evaluate"]["override"])
    info.pop("sampler")
    info.pop("output")
    info["theory"]["camb"]["path"] = str(COCOA / "external_modules/code/CAMB")

    # Step 2: chi2 of the model mean against the loaded synthetic vector.
    with get_model(info_or_yaml_or_file=info) as model:
        model.logposterior(params_values=point, cached=False)
        mean = model.likelihood["des_3pcf.joint_shear"].last_datavector
        chi2 = float(ci.compute_joint_chi2(prediction=mean))
    assert np.isfinite(chi2) and chi2 < CHI2_BOUND, chi2
