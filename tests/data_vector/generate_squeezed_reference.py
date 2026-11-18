"""Record BiHalofit effective powers and a high-precision cyclic reference.

The explicit --source-dir must contain the pinned fastnc halofit.py.
That unmodified module supplies the fitted coefficients and power values.
Only the final squared-side kernels and cyclic sum use 80-digit Decimal
arithmetic. This separates the source fit's root tolerance from the loss
of precision in its ordinary squeezed-triangle formula.
"""

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

import numpy as np

from generate_physics_reference import read_source
from test_physics_model import physical_inputs
from test_physics_squeezed import decimal_triangle


def main():
    """Save a small reproducible reference; do not import the C port."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("reference_squeezed_bihalofit.json"),
    )
    arguments = parser.parse_args()
    source_file = arguments.source_dir / "halofit.py"
    source = read_source(path=source_file, name="squeezed_reference_halofit")
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
    matter = source.Halofit()
    matter.set_cosmology(cosmo=cosmology)
    matter.set_lgr(z=inputs["z"].copy(), lgr=inputs["growth"].copy())
    matter.set_pklin(k=inputs["k"].copy(), pklin=inputs["p_linear0"].copy())
    triangles = []
    for opening in (0.0, 0.25, 0.875):
        base = np.asarray([1.0e-9, 1.0, 1.0 + opening * 1.0e-9])
        for cyclic in range(3):
            triangles.append(np.roll(base, shift=cyclic))
    sides = np.asarray(triangles, dtype=np.float64)
    redshift = np.zeros(shape=sides.shape[0], dtype=np.float64)
    ordinary = matter.get_bihalofit(
        k1=sides[:, 0],
        k2=sides[:, 1],
        k3=sides[:, 2],
        z=redshift,
        which=["Bh3"],
    )
    coefficients = matter.get_bihalofit_coeffs(z=redshift)
    scaled = sides * coefficients["r_sigma"][:, None]
    effective = np.empty_like(sides)
    damping = np.ones(shape=redshift.size, dtype=np.float64)
    for leg in range(3):
        linear = matter.get_interpolated_pklin(sides[:, leg], redshift)
        wave = scaled[:, leg]
        fitted = (1 + coefficients["fn"] * wave**2)
        fitted /= 1 + coefficients["gn"] * wave + coefficients["hn"] * wave**2
        fitted *= linear
        correction = coefficients["mn"] * wave ** coefficients["mun"]
        correction += coefficients["nn"] * wave ** coefficients["nun"]
        correction *= 1 + (coefficients["pn"] * wave) ** -3
        effective[:, leg] = fitted + 1 / correction
        damping /= 1 + coefficients["en"] * wave
    expected = []
    with localcontext() as context:
        context.prec = 80
        for index in range(redshift.size):
            _, powers, _, tree = decimal_triangle(
                sides=sides[index], power=effective[index]
            )
            value = sum(tree)
            fitted_d = Decimal.from_float(float(coefficients["dn"][index]))
            for cyclic in range(3):
                opposite = (cyclic + 2) % 3
                scaled_value = float(scaled[index, opposite])
                scaled_side = Decimal.from_float(scaled_value)
                correction = 2 * fitted_d * scaled_side
                correction *= powers[cyclic] * powers[(cyclic + 1) % 3]
                value += correction
            value *= Decimal.from_float(float(damping[index]))
            expected.append(float(value))
    record = {
        "source_commit": "7b2336ce37a909bed51f07f32a2a4f096e71b6ce",
        "halofit_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "reference": "Decimal F2/cyclic sum; source fitted effective powers",
        "sides": sides.tolist(),
        "redshifts": redshift.tolist(),
        "effective_power": effective.tolist(),
        "scaled_sides": scaled.tolist(),
        "fitted_d": coefficients["dn"].tolist(),
        "damping": damping.tolist(),
        "high_precision_bh3": expected,
        "ordinary_source_bh3": ordinary.tolist(),
    }
    arguments.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
