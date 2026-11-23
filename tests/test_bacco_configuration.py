"""Refuse invalid BACCO targets before a forecast or sampler starts.

These checks use the settings class of the BACCO theory block and a
separate process with a small temporary package for file-identity
checks. They run no neural network, Boltzmann solve or survey
projection, and change no installed weight file.
"""

from pathlib import Path
import subprocess
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / "external_modules/code/cosmolike_core"))
from theory.bacco_bispectrum.response import BaccoSettings


@pytest.mark.parametrize("settings", [
    {"nz": "17"},
    {"high_k_anchor": [10.]},
    {"logu_min": -800.},
    {"high_z_width": 0.},
])
def test_bad_response_controls_fail_with_named_error(settings):
    """Unsupported types and unrepresentable triangle domains fail before allocation."""
    with pytest.raises(ValueError, match="BACCO"):
        BaccoSettings(**settings)


def test_replaced_weight_is_detected_without_touching_installation(tmp_path):
    """Select an isolated package before import, then replace its data file.

    Python's ordinary module search path chooses this temporary package in
    the child process. No imported method or module is replaced at runtime.
    The weight identity check does not need a real emulator pickle.
    """
    package = tmp_path / "baccoemu"
    weights = package / "bispectrum_baryonic_boost"
    weights.mkdir(parents=True)
    (package / "__init__.py").write_text("# Isolated file-identity check.\n")
    filenames = (
        "powerspectra_ratio_bcm_emu_folded.pkl",
        "bispectra_ratio_bcm_emu_folded.pkl",
    )
    for name in filenames:
        (weights / name).write_bytes(b"initial fake weight bytes")
    program = '''
from pathlib import Path
from theory.bacco_bispectrum.response import BaccoFeedback, BaccoSettings
producer = BaccoFeedback(settings=BaccoSettings())
producer.verify_weights()
path = Path(producer.weights["B"]["path"])
path.write_bytes(b"different fake weight bytes")
try:
    producer.verify_weights()
except ValueError as error:
    assert "weights changed" in str(error)
else:
    raise AssertionError("Changed weights were accepted")
'''
    # Prepend the temporary directory before importing the package in a fresh
    # interpreter. The parent's installation and imported modules stay intact.
    startup = f"import sys; sys.path[:0] = {[str(tmp_path)]!r}\n"
    result = subprocess.run(
        args=[sys.executable, "-B", "-c", startup + program],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
