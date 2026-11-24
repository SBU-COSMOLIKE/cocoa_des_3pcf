"""Check integer dtypes on the two routes of xi_from_power.

The notebook route copies its inputs, so it accepts integer arrays of any
width (NumPy's default int64, for example) after a range check and
converts them to the C int. The production route borrows the buffer and
keeps requiring int32. A dtype refusal ends the process, so the refusals
run in child interpreters and the test reads their diagnostics.
"""

import subprocess
import sys

import numpy as np
import pytest


# Small valid inputs: two rows (one xi+, one xi-), three separation nodes
# per row with weights that sum to one, and smooth power spectra.
INPUTS = """
import numpy as np
ell = np.geomspace(2.0, 3000.0, 64)
dell = np.gradient(ell)
theta = np.radians(np.array([[10.0, 12.0, 14.0], [20.0, 24.0, 28.0]]) / 60.0)
weight = np.full((2, 3), 1.0 / 3.0)
spectra_e = np.tile(1.0e-9 / (1.0 + ell / 100.0)**2, (2, 1))
spectra_b = 1.0e-3 * spectra_e
calibration = np.array([0.01, -0.02])
"""


def xi(route, sign, bins):
    """Evaluate xi_from_power on one route with the shared inputs."""
    namespace = {}
    exec(INPUTS, namespace)
    return route.xi_from_power(
        ell=namespace["ell"], dell=namespace["dell"], theta=namespace["theta"],
        theta_weight=namespace["weight"], spectra_e=namespace["spectra_e"],
        spectra_b=namespace["spectra_b"], sign=sign, bins=bins,
        calibration=namespace["calibration"])


def test_notebook_route_converts_int64(compiled):
    """int64 sign and bins give the int32 result bit for bit."""
    sign = np.array([1, -1])
    bins = np.array([[0, 1], [1, 1]])
    reference = xi(compiled.notebook, sign.astype(np.int32), bins.astype(np.int32))
    converted = xi(compiled.notebook, sign.astype(np.int64), bins.astype(np.int64))
    assert np.array_equal(reference.view(np.uint64), converted.view(np.uint64))
    # The production route gives the same numbers for int32 inputs.
    production = xi(compiled.production, sign.astype(np.int32), bins.astype(np.int32))
    assert np.allclose(production, reference, rtol=1.0e-14, atol=0.0)


CHILD = """
import numpy as np
import cosmolike_des_3pcf_interface as compiled
{inputs}
route = getattr(compiled, {route!r})
sign = np.array({sign!r}, dtype={dtype!r})
route.xi_from_power(ell=ell, dell=dell, theta=theta, theta_weight=weight,
                    spectra_e=spectra_e, spectra_b=spectra_b, sign=sign)
print("evaluated")
"""


@pytest.mark.parametrize("route,sign,dtype,message", [
    # Production borrows the buffer: int64 stays refused.
    ("production", [1, -1], "int64", "wrong dtype"),
    # Notebook converts only values the C int can hold.
    ("notebook", [1, 2**40], "int64", "exceeds the range"),
])
def test_integer_refusals(route, sign, dtype, message):
    """The refusal names the dtype or the range problem."""
    program = CHILD.format(inputs=INPUTS, route=route, sign=sign, dtype=dtype)
    result = subprocess.run(args=[sys.executable, "-B", "-c", program],
                            capture_output=True, text=True, check=False, timeout=120)
    output = result.stdout + result.stderr
    assert 0 != result.returncode, output
    assert "evaluated" not in output
    assert message in output, output
