"""Generate a small reference using the unmodified pinned fastnc sources.

Run this explicitly with --source-dir pointing to the pinned fastnc folder.
It never imports the C port. The saved numbers test source agreement, while
the analytic tests in test_physics_model.py provide separate physical checks.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from test_physics_model import physical_inputs


def read_source(path, name):
    """Load an explicit external file without changing its source text."""
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main():
    """Record matter and 19-component IA outputs for six triangles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    arguments = parser.parse_args()
    halofit_path = arguments.source_dir / "halofit.py"
    alignment_path = arguments.source_dir / "ia_bispectra.py"
    halofit = read_source(path=halofit_path, name="fastnc_reference_halofit")
    alignment = read_source(path=alignment_path, name="fastnc_reference_ia")
    inputs = physical_inputs()
    cosmology = {
        "Om0": inputs["omega_m"],
        "Ode0": inputs["omega_de"],
        "w0": inputs["w0"],
        "wa": inputs["wa"],
        "ns": inputs["ns"],
        "sigma8": inputs["sigma8"],
        "fnu0": 0.0,
    }
    matter = halofit.Halofit()
    matter.set_cosmology(cosmo=cosmology)
    matter.set_lgr(z=inputs["z"].copy(), lgr=inputs["growth"].copy())
    matter.set_pklin(k=inputs["k"].copy(), pklin=inputs["p_linear0"].copy())
    intrinsic = alignment.BispectraIA(
        k=inputs["k"].copy(),
        pklin=inputs["p_linear0"].copy(),
        z=inputs["z"].copy(),
        lgr=inputs["growth"].copy(),
        cosmo=cosmology,
    )
    intrinsic.set_pknl(k=inputs["k"], pknl=inputs["p_nonlinear"][0].copy())
    # BispectrumTATT.set_lgr installs this public coefficient reader in
    # its calculator. Do the same here without importing the outer class.
    intrinsic.z2lgr = InterpolatedUnivariateSpline(
        x=inputs["z"], y=inputs["growth"], ext=1
    )
    sides = np.asarray(
        [
            [0.03, 0.04, 0.05],
            [0.17, 0.11, 0.23],
            [0.8, 1.3, 1.8],
            [2.0, 2.3, 4.1],
            [0.02, 1.0, 1.01],
            [6.0, 9.0, 11.0],
        ],
        dtype=np.float64,
    )
    redshifts = np.asarray([0.0, 0.15, 0.4, 0.8, 1.2, 1.9])
    parameters = {
        "a1": 0.8,
        "eta1": 0.35,
        "a2": -1.1,
        "eta2": -0.65,
        "bta": 0.9,
        "ct": 0.25,
        "zpivot": 0.62,
    }
    record = {
        "source_commit": "7b2336ce37a909bed51f07f32a2a4f096e71b6ce",
        "halofit_sha256": hashlib.sha256(halofit_path.read_bytes()).hexdigest(),
        "ia_sha256": hashlib.sha256(alignment_path.read_bytes()).hexdigest(),
        "input_definition": "physical_inputs() in test_physics_model.py",
        "sides": sides.tolist(),
        "redshifts": redshifts.tolist(),
        "parameters": parameters,
        "matter": [],
        "alignment": {},
    }
    for selection in (["Bh1", "Bh3"], ["BT"], ["Bh1"], ["Bh3"]):
        values = matter.get_bihalofit(
            k1=sides[:, 0],
            k2=sides[:, 1],
            k3=sides[:, 2],
            z=redshifts,
            which=selection,
        )
        record["matter"].append(values.tolist())
    for mode in (0, 1):
        values = intrinsic.get_ia_bispectra(
            k1_mag=sides[:, 0],
            k2_mag=sides[:, 1],
            k3_mag=sides[:, 2],
            z=redshifts,
            z_piv=parameters["zpivot"],
            A1=parameters["a1"],
            alphaIA=parameters["eta1"],
            A2=parameters["a2"],
            alphaIA_2=parameters["eta2"],
            bias_ta=parameters["bta"],
            Ct=parameters["ct"],
            remove_alignment=False,
            do_non_linear=bool(mode),
        )
        record["alignment"][str(mode)] = np.asarray(values).T.tolist()
    destination = Path(__file__).with_name("reference_physics_fastnc.json")
    destination.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(f"Saved independent fastnc reference to {destination}")


if __name__ == "__main__":
    main()
