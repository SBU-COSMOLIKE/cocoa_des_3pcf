"""Check source identity, isolated execution and comparison reach.

Live-source checks require the explicit checkouts named by the example
YAML. They skip with a named reason when those external files are absent;
the configuration and provenance checks remain runnable independently.
No test downloads source code or installs a dependency.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from des3pcf.comparison import BLAS_VARIABLES, FORMAT
from des3pcf.comparison import _worker_environment
from des3pcf.comparison import comparison_status, load_comparison_config
from des3pcf.comparison import load_comparison_case, run_code_comparison


PROJECT = Path(__file__).resolve().parents[1]


def small_configuration(tmp_path):
    """Copy settings, retaining explicit absolute source locations."""
    original = PROJECT / "EXAMPLE_COMPARE_CODES.yaml"
    settings = load_comparison_config(path=original)
    settings.pop("configuration")
    settings["output"] = str(tmp_path / "results")
    settings["nfft"] = 16
    settings["lmax"] = 2
    settings["mmax"] = 3
    settings["phi_count"] = 5
    path = tmp_path / "compare.yaml"
    path.write_text(yaml.safe_dump(settings, sort_keys=False))
    return path


def test_missing_sources_are_named_without_execution(tmp_path):
    """Missing source locations cannot silently select an installed version."""
    path = small_configuration(tmp_path=tmp_path)
    settings = yaml.safe_load(path.read_text())
    for name in settings["sources"]:
        settings["sources"][name] = None
    path.write_text(yaml.safe_dump(settings, sort_keys=False))
    report = comparison_status(path=path)
    assert report["format"] == FORMAT
    for record in report["sources"].values():
        assert record["status"] == "unavailable"
        assert record["problems"] == ["No source checkout was configured."]
    assert not (tmp_path / "results").exists()


def test_modified_source_is_not_accepted(tmp_path):
    """A declared pin is insufficient when the source bytes differ."""
    path = small_configuration(tmp_path=tmp_path)
    settings = yaml.safe_load(path.read_text())
    checkout = tmp_path / "modified"
    source = checkout / "python/fast_map3.py"
    source.parent.mkdir(parents=True)
    source.write_text("raise RuntimeError('must not execute')\n")
    settings["sources"]["integrator"] = str(checkout)
    path.write_text(yaml.safe_dump(settings, sort_keys=False))
    report = comparison_status(path=path)
    record = report["sources"]["integrator"]
    assert record["status"] == "unavailable"
    assert record["problems"] == ["SHA256 differs: python/fast_map3.py"]


def test_comparison_limits_and_child_environment(tmp_path):
    """Guard small grids and set all BLAS vendors before child imports."""
    path = small_configuration(tmp_path=tmp_path)
    settings = yaml.safe_load(path.read_text())
    settings["threads"] = 5
    path.write_text(yaml.safe_dump(settings, sort_keys=False))
    with pytest.raises(ValueError, match="threads"):
        load_comparison_config(path=path)
    environment = _worker_environment(threads=4)
    for name in BLAS_VARIABLES:
        assert environment[name] == "1"
    assert environment["OMP_NUM_THREADS"] == "4"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_live_original_codes_and_frozen_labels(tmp_path):
    """Check live outputs, matching FFT modes and the filter discrepancy."""
    path = small_configuration(tmp_path=tmp_path)
    preflight = comparison_status(path=path)
    for name, source in preflight["sources"].items():
        if source["status"] != "verified":
            pytest.skip(
                f"Explicit {name} checkout is unavailable: {source['problems']}"
            )
    for name in ("numpy", "scipy", "astropy", "pandas", "mpi4py"):
        if not preflight["dependencies"][name]:
            pytest.skip(f"The original TATT package requires {name}.")
    report = run_code_comparison(path=path)
    cases = {}
    for case in report["cases"]:
        assert case["status"] == "completed", case
        cases[case["name"]] = case
        saved = load_comparison_case(case=case)
        assert saved["reference"].shape == saved["ported"].shape
        assert np.all(np.isfinite(saved["ported"]))
        assert np.any(saved["reference"] != 0)
    for name in (
        "bessel_tatt_untapered", "bessel_main_taper", "aperture_equal"
    ):
        assert cases[name]["kind"] == "live_original_code"
        assert cases[name]["max_scaled_error"] < 3.0e-11
    assert cases["aperture_unequal"]["expected_discrepancy"]
    assert cases["aperture_unequal"]["max_scaled_error"] > 1.0e-3
    assert cases["natural_tatt_public"]["kind"] == "live_original_code"
    # The live source's capped left-rectangle integral has an O(1/4096)
    # bias, independently measured below. This 1e-3 bound checks the full
    # pipeline against that finite source rule, not continuum accuracy.
    assert cases["natural_tatt_public"]["max_scaled_error"] < 1.0e-3
    assert cases["natural_exact_coupling"]["expected_discrepancy"]
    coupling = cases["coupling_tatt_rectangle_bias"]
    np.testing.assert_allclose(
        coupling["source_left_endpoint_bias"],
        coupling["rectangle_bias_prediction"],
        rtol=3.0e-11, atol=0.0,
    )
    assert coupling["max_scaled_error"] < 3.0e-4
    for step in coupling["rectangle_bias_ladder"]:
        assert not step["converged"]
        np.testing.assert_allclose(
            step["source_integral"], step["predicted_bias"],
            rtol=3.0e-10, atol=0.0,
        )
    for name in ("physics_matter_frozen", "physics_ia_mode0_frozen",
                 "physics_ia_mode1_frozen"):
        assert cases[name]["kind"] == "frozen_reference"
        assert len(cases[name]["fixture_sha256"]) == 64
    assert cases["physics_matter_frozen"]["max_scaled_error"] < 2.0e-4
    assert cases["physics_ia_mode0_frozen"]["max_scaled_error"] < 1.0e-12
    assert cases["physics_ia_mode1_frozen"]["max_scaled_error"] < 1.0e-12
    stored = yaml.safe_load(Path(report["report_file"]).read_text())
    assert len(stored["cases"]) == 10
