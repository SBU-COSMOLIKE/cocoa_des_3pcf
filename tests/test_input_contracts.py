"""Check that invalid scientific inputs stop with a named diagnostic.

Each child process imports the built module and performs one small invalid
call. Its failure must come from the component's input contract, rather
than a missing import or a numerical result containing NaN or infinity.
"""

from pathlib import Path
import subprocess
import sys

import pytest


WORKER = Path(__file__).with_name("refusal_worker.py")


@pytest.mark.parametrize("adapter", ["production", "notebook"])
@pytest.mark.parametrize("case", [
    "collinear",
    "underflow",
    "radius_overflow",
    "negative_width",
    "negative_gram_weight",
    "nonfinite",
    "result_overflow",
    "field_index",
    "triangle_shape",
    "asymmetric_pair",
])
def test_invalid_inputs_stop(case, adapter):
    """Inspect the expected process exit without intercepting core behavior."""
    result = subprocess.run(
        args=[sys.executable, "-B", str(WORKER), case, adapter],
        capture_output=True, text=True, check=False, timeout=20
    )
    message = result.stdout + result.stderr
    expected = {
        "collinear": "phi must lie strictly between zero and pi",
        "underflow": "geometry is not representable",
        "radius_overflow": "geometry is not representable",
        "negative_width": "bin widths must be",
        "negative_gram_weight": "weights violates its lower bound",
        "nonfinite": "must be finite",
        "result_overflow": "Gram result must be finite",
        "field_index": "field index out of range",
        "triangle_shape": "aperture component and triangle axes disagree",
        "asymmetric_pair": "pair must be symmetric",
        "unaligned": "weights has an unaligned buffer",
    }
    if adapter == "notebook":
        expected["negative_gram_weight"] = "Gram weights must be nonnegative"
        expected["triangle_shape"] = (
            "real/imaginary cube shapes and triangle counts must agree"
        )
        expected["result_overflow"] = "Gram result is not finite"
    assert result.returncode == 1
    assert expected[case] in message
    assert "Invalid input returned" not in message
    assert "ModuleNotFoundError" not in message
    assert "Traceback" not in message
    assert "runtime error:" not in message  # The explicit guard precedes UBSan.


def test_unaligned_production_buffer_stops():
    """Borrowed double storage must be aligned before C can read its values."""
    test_invalid_inputs_stop(case="unaligned", adapter="production")
