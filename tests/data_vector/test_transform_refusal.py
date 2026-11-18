"""Invalid transform inputs terminate before unaligned reads or FFT calls."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def invalid_call(case):
    """Perform one invalid call in this file's isolated child process."""
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project / "interface"))
    import cosmolike_des_3pcf_interface as compiled

    compiled.set_omp_threads(nthreads=1)
    ell = np.geomspace(start=0.1, stop=10.0, num=12)
    phi = np.array([0.4, 1.2])
    indices = np.array([3, 5], dtype=np.int32)
    if case == "unaligned":
        storage = bytearray(ell.nbytes + 1)
        shifted = np.ndarray(shape=ell.shape, dtype=np.float64,
                             buffer=storage, offset=1)
        shifted[:] = ell
        ell = shifted
    elif case == "nonuniform":
        ell[4] *= 1.02
    elif case == "negative_phi":
        phi[0] = -0.1
    elif case == "bad_index":
        indices[1] = 12
    elif case == "nonfinite":
        ell[4] = np.nan
    plan = compiled.production.TransformPlan(
        ell=ell, phi=phi, indices=indices, lmax=2, mmax=3,
    )
    if case == "wrong_multipoles":
        plan.natural_components(b_l=np.zeros((2, 12, 12)))
    print("Invalid transform returned")


@pytest.mark.parametrize("case,diagnostic", [
    ("unaligned", "unaligned array"),
    ("nonuniform", "uniformly spaced in logarithm"),
    ("negative_phi", "phi must lie in (0,pi)"),
    ("bad_index", "radial index outside Fourier grid"),
    ("nonfinite", "finite values"),
    ("wrong_multipoles", "multipoles mismatch plan shape"),
])
def test_invalid_transform_stops(case, diagnostic):
    """The contract check, rather than a sanitizer, names the failure."""
    result = subprocess.run(
        args=[sys.executable, "-B", str(Path(__file__)), case],
        capture_output=True, text=True, check=False, timeout=20,
    )
    message = result.stdout + result.stderr
    assert result.returncode == 1
    assert diagnostic in message
    assert "runtime error:" not in message
    assert "Traceback" not in message
    assert "Invalid transform returned" not in message


if __name__ == "__main__":
    invalid_call(case=sys.argv[1])
