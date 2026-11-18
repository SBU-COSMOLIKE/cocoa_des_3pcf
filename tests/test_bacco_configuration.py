"""Refuse invalid BACCO targets before a forecast or sampler starts.

These checks use actual YAML validation and a separate process with a small
temporary package for file-identity checks. They run no neural network,
Boltzmann solve or survey projection, and change no installed weight file.
"""

from pathlib import Path
import subprocess
import sys

import pytest
import yaml


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import run_3pcf as driver
from des3pcf.bacco_feedback import BaccoSettings
from des3pcf.inference import analysis_configuration, load_inference


@pytest.mark.parametrize("command", ["prepare", "evaluate", "fisher"])
def test_bacco_schema_retains_every_resolved_setting(command):
    """A saved target records all continuation choices and separate sampling axes."""
    config, _, _ = driver.load_configuration(
        path=PROJECT / "configs/des_y3_bacco_fisher.yaml", command=command,
    )
    assert config["feedback"]["settings"]["neutrino_model"] == "coherent_linear_neutrinos"
    assert config["feedback"]["settings"]["logk_min"] == -15.
    assert config["ia"]["feedback_amplitude"] == 1.
    assert len(config["fisher"]["steps"]) == 25
    assert config["accuracy"]["feedback_v_boost"] == 1.


@pytest.mark.parametrize("sample_amplitude", [False, True])
def test_fisher_cannot_change_converted_response_amplitude(tmp_path, sample_amplitude):
    """The cold-response conversion cannot be replaced by a total-response power."""
    path = PROJECT / "configs/des_y3_bacco_fisher.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if sample_amplitude:
        config["fisher"]["steps"]["feedback_amplitude"] = .1
    else:
        config["ia"]["feedback_amplitude"] = .5
    changed = tmp_path / "invalid.yaml"
    changed.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="feedback_amplitude"):
        driver.load_configuration(path=changed, command="fisher")


@pytest.mark.parametrize("sample_amplitude", [False, True])
def test_inference_rejects_amplitude_before_likelihood_construction(sample_amplitude):
    """A sampler must vary physical BCM coordinates, with the converted amplitude fixed."""
    specification, location = load_inference(path=PROJECT / "EXAMPLE_NAUTILUS1.yaml")
    specification["prediction"] = "configs/des_y3_bacco_fisher.yaml"
    specification["initial"] = {}
    if sample_amplitude:
        specification["priors"]["feedback_amplitude"] = {
            "distribution": "uniform",
            "minimum": .5,
            "maximum": 1.5,
        }
    else:
        specification["initial"]["feedback_amplitude"] = .5
    with pytest.raises(ValueError, match="feedback_amplitude"):
        analysis_configuration(specification=specification, location=location,
                               driver=driver)


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
from des3pcf.bacco_feedback import BaccoFeedback, BaccoSettings
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
    # Prepend both directories before importing the package in a fresh
    # interpreter. The parent's installation and imported modules stay intact.
    startup = f"import sys; sys.path[:0] = {[str(tmp_path), str(PROJECT / 'python')]!r}\n"
    result = subprocess.run(
        args=[sys.executable, "-B", "-c", startup + program],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
