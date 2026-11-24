"""Write the synthetic data vectors of cases 3, 4 and 5 from the current model.

Cases 3 (NLA) and 4 and 5 (TATT) score synthetic vectors: the model's own
mean at the example yaml's fiducial point (its sampler.evaluate.override).
At that point a correct installation then gives chi2 = 0. The vectors must
follow the model: rerun this after any change to the theory, the row
geometry or the likelihood defaults (a few minutes):

    python scripts/make_synthetic_datavectors.py

It writes data/des_y3_nla_synthetic.datavector and
data/des_y3_tatt_synthetic.datavector ("index value" per line, all 480
rows, as the converter writes the measured vectors).
"""

from pathlib import Path
import sys

from cobaya.model import get_model
from cobaya.yaml import yaml_load_file

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from convert_desy3_fits import write_vector  # noqa: E402


def main():
    """Evaluate each fiducial and write its full mean as the synthetic vector."""
    for label, example in (("nla", "EXAMPLE_EVALUATE3.yaml"), ("tatt", "EXAMPLE_EVALUATE4.yaml")):
        # Step 1: the example's model and fiducial point; CAMB by absolute path.
        info = yaml_load_file(str(PROJECT / example))
        point = dict(info["sampler"]["evaluate"]["override"])
        info.pop("sampler")
        info.pop("output", None)
        info["theory"]["camb"]["path"] = str(PROJECT.parents[1] / "external_modules/code/CAMB")

        # Step 2: the full mean at that point (cached=False: always computed).
        with get_model(info_or_yaml_or_file=info) as model:
            model.logposterior(params_values=point, cached=False)
            mean = model.likelihood["des_3pcf.joint_shear"].last_datavector.copy()
        write_vector(PROJECT / f"data/des_y3_{label}_synthetic.datavector", mean)


if __name__ == "__main__":
    main()
