"""Save aperture weights from the pinned external Python kernel.

The production C implementation is checked against independently executed
Python equations. This script extracts only the inspected _calculateT
method from the exact fastnc source identified below. It does not import
fastnc, its cosmology providers or its transform machinery. The source hash
prevents a different upstream file from silently becoming the reference.

Run with --source pointing to fastnc/fastnc.py from the pinned commit.
The output records every coordinate, bin width, radius and source identity.
Regenerating it is a deliberate reference change, not a test repair.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_COMMIT = "7b2336ce37a909bed51f07f32a2a4f096e71b6ce"
SOURCE_SHA256 = "1d72ba0a735a1b899034d58ccf3589d93def05afcb6d82eb92076274c9d459b8"
OUTPUT = Path(__file__).parent / "data_vector" / "reference_aperture_weights.json"


def load_kernel(source):
    """Return the verified external method without importing its package.

    Arguments:
        source = path to the pinned fastnc/fastnc.py file.

    Returns:
        callable = original _calculateT function; its unused cls argument is None.

    Raises:
        ValueError if the bytes differ or the expected function is absent.

    Parsing produces a tree of Python statements. Only the selected
    function definition enters the compiled module; top-level package
    imports and other code are excluded. Its classmethod decorator is
    removed because the extracted function has no containing class.
    """
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(
            f"source SHA256 is {digest}; expected {SOURCE_SHA256}. "
            f"Use fastnc commit {SOURCE_COMMIT}."
        )
    tree = ast.parse(content.decode("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_calculateT":
            node.decorator_list = []
            selected = ast.Module(body=[node], type_ignores=[])
            namespace = {"np": np}
            exec(compile(selected, filename=str(source), mode="exec"), namespace)
            return namespace["_calculateT"]
    raise ValueError("The pinned source has no _calculateT method.")


def make_record(kernel):
    """Evaluate equal and unequal filters on a small finite triangle grid.

    Arguments:
        kernel = verified external complex aperture kernel.

    Returns:
        dict = geometry, provenance and weights with shape [3, 4, 24].

    The kernel supplies only the geometric functions. The integration
    measure and mirrored-radius sum implement the finite SAS convention.
    The factor one quarter is stored in the weights, so the C contraction
    must not apply it again.
    """
    side2 = np.array([0.0008, 0.0014, 0.00245])
    side3 = np.array([0.001, 0.00175])
    angle = np.array([0.35, 0.90, 1.45, 2.00])
    filters = np.array([
        [0.0011, 0.0011, 0.0011],
        [0.0015, 0.0020, 0.0028],
        [0.0022, 0.0013, 0.0018],
    ])
    dlnd = float(np.log(1.75))
    dphi = 0.55
    grid2, grid3, grid_angle = np.meshgrid(side2, side3, angle, indexing="ij")
    grid2 = grid2.ravel()
    grid3 = grid3.ravel()
    grid_angle = grid_angle.ravel()
    weights = np.empty((3, 4, grid2.size), dtype=np.complex128)

    # Each row has three filter radii but uses the same measured triangles.
    # Dividing by R1 makes the kernel arguments dimensionless.
    for index in range(3):
        radius1, radius2, radius3 = filters[index]
        side = grid2 / radius1
        other_radius = grid3 / radius1
        other = other_radius * np.exp(1j * grid_angle)
        direct = kernel(None, side, other, 1.0, radius2 / radius1, radius3 / radius1)
        mirror = kernel(None, side, other, 1.0, radius3 / radius1, radius2 / radius1)
        measure = side**2 * other_radius**2 * dlnd**2 * dphi / (2.0 * np.pi)
        weights[index] = 0.25 * (np.asarray(direct) + np.asarray(mirror)) * measure

    return {
        "source_url": (
            f"https://github.com/git-sunao/fastnc/blob/{SOURCE_COMMIT}/fastnc/fastnc.py"
        ),
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "method": "_calculateT, with finite SAS measure, mirror sum, and 1/4",
        "units": "d2, d3, phi and filters in radians; dlnd dimensionless",
        "numpy_version": np.__version__,
        "d2": side2.tolist(),
        "d3": side3.tolist(),
        "phi": angle.tolist(),
        "filters": filters.tolist(),
        "dlnd": dlnd,
        "dphi": dphi,
        "weights_real": weights.real.tolist(),
        "weights_imag": weights.imag.tolist(),
    }


def main():
    """Validate the pinned source, then write the fully specified reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    arguments = parser.parse_args()
    kernel = load_kernel(source=arguments.source)
    record = make_record(kernel=kernel)
    OUTPUT.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Wrote {OUTPUT}; source SHA256 {SOURCE_SHA256}")


if __name__ == "__main__":
    main()
