"""Check workload preparation without CAMB, C predictions or timing calls."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from des3pcf.scaling_threepcf import preflight_threepcf, threepcf_point_bank
from des3pcf.scaling_threepcf import _clip_timing_sources, TIMING_SOURCE_POLICY


PROJECT = Path(__file__).resolve().parents[1]
CONFIGURATION = PROJECT / "configs" / "des_y3_tatt_fisher.yaml"


def test_seeded_points_are_random_reproducible_and_worker_independent():
    """Every thread count receives exactly the same non-fiducial physical inputs."""
    bank = threepcf_point_bank(config_path=CONFIGURATION)
    repeated = threepcf_point_bank(config_path=CONFIGURATION)
    assert bank == repeated
    assert len(bank["points"]) == 10
    assert bank["draw_settings"]["thread_counts"] == [1, 4, 8]
    assert bank["draw_settings"]["blas_threads"] == 1
    assert bank["warmup_point"]["seed"] == 147
    assert bank["warmup_point"]["index"] == -1
    bounds = bank["draw_settings"]["bounds"]
    for index, point in enumerate(bank["points"]):
        assert point["seed"] == 137 + index
        assert point["index"] == index
        parameters = point["parameters"]
        for name, interval in bounds.items():
            assert interval[0] <= parameters[name] <= interval[1]
        assert parameters["w0"] + parameters["wa"] < -0.01
        assert parameters["mnu"] == bank["draw_settings"]["fixed_parameters"]["mnu"]
    for name in bounds:
        distinct = {point["parameters"][name] for point in bank["points"]}
        assert len(distinct) == 10
    short = threepcf_point_bank(config_path=CONFIGURATION, npoints=3)
    assert short["points"] == bank["points"][:3]
    # The manifest owns every dictionary; editing one requested point
    # cannot change a later point or a separately generated manifest.
    bank["points"][0]["parameters"]["a1"] = 99.0
    assert repeated["points"][0]["parameters"]["a1"] != 99.0
    json.dumps(repeated, allow_nan=False)


@pytest.mark.parametrize("npoints,seed", [(0, 137), (2.5, 137), (10, -1), (10, True)])
def test_invalid_bank_count_or_seed_is_refused(npoints, seed):
    """A malformed draw request fails before opening a survey source file."""
    with pytest.raises(ValueError, match="integer"):
        threepcf_point_bank(config_path=CONFIGURATION, npoints=npoints, seed=seed)


def test_preflight_imports_neither_camb_nor_compiled_interface():
    """A fresh child inspects known inputs without activating either model engine."""
    script = """
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from des3pcf.scaling_threepcf import preflight_threepcf
report = preflight_threepcf(config_path=Path(sys.argv[2]))
assert 'camb' not in sys.modules
assert 'cosmolike_des_3pcf_interface' not in sys.modules
print(json.dumps(report, allow_nan=False))
"""
    execution = subprocess.run(
        args=[sys.executable, "-B", "-c", script,
              str(PROJECT / "python"), str(CONFIGURATION)],
        capture_output=True, text=True, check=False, timeout=15,
    )
    assert execution.returncode == 0, execution.stderr
    report = json.loads(execution.stdout)
    assert report["numerical_work_executed"] is False
    assert report["thread_counts"] == [1, 4, 8]
    assert report["aligned_fft_size"] >= 512
    assert report["aligned_ell_bounds"][0] < report["aligned_ell_bounds"][1]
    # The pinned source file contains negative samples. If the inputs are
    # absent, preflight reports that instead; neither state starts CAMB.
    assert report["ready"] is False
    assert any(message.startswith("Source input:") for message in report["blockers"])


def test_preflight_reports_projection_support_before_geometry_creation(tmp_path):
    """The aligned FFT support can extend beyond a nominal requested range."""
    configuration = yaml.safe_load(CONFIGURATION.read_text())
    changed = deepcopy(configuration)
    for section, name in (("source", "path"), ("rows", "aperture")):
        changed[section][name] = str(
            (CONFIGURATION.parent / changed[section][name]).resolve()
        )
    changed["mean"]["aperture_settings"]["projection_settings"]["ell_min"] = 2.0e5
    path = tmp_path / "inconsistent_geometry.yaml"
    path.write_text(yaml.safe_dump(changed, sort_keys=False))
    report = preflight_threepcf(config_path=path)
    assert report["ready"] is False
    diagnostic = "Projection bounds do not cover the aligned FFT ell grid."
    assert diagnostic in report["blockers"]


def test_timing_clipping_records_removed_area_and_preserves_inputs():
    """A signed source curve has a known removed area and normalization."""
    redshift = np.array([0.0, 1.0, 2.0, 3.0])
    raw = np.array([[0.0, 2.0, -1.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
    original = raw.copy()
    clipped, record = _clip_timing_sources(redshift=redshift, densities=raw)
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(clipped, [[0, 1, 0, 0], [0, 0.5, 0.5, 0]])
    assert record["policy"] == TIMING_SOURCE_POLICY
    assert record["negative_sample_count"] == [1, 0]
    assert record["negative_area_magnitude"] == [1.0, 0.0]
    assert record["raw_integrals"] == [1.0, 2.0]
    assert record["clipped_integrals_before_normalization"] == [2.0, 2.0]
    assert record["normalization_factors"] == [0.5, 0.5]
    np.testing.assert_array_equal(np.trapz(clipped, x=redshift, axis=1), 1.0)
    with pytest.raises(ValueError, match="positive probability"):
        _clip_timing_sources(redshift=redshift, densities=-np.ones((1, 4)))


def test_timing_configuration_records_its_policy_and_fits_shifted_support():
    """Only the separate benchmark file activates the authorized treatment."""
    if not (PROJECT / "data" / "upstream_y3").is_dir():
        pytest.skip("Pinned source files are not available for input inspection.")
    report = preflight_threepcf(
        config_path=PROJECT / "configs" / "des_y3_tatt_scaling.yaml",
    )
    assert report["ready"] is True, report["blockers"]
    assert report["row_count"] == 80
    assert report["source"]["table_zmax"] == 3.1
    processing = report["source"]["processing"]
    assert processing["timing_only"] is True
    assert processing["negative_sample_count"] == [12, 0, 0, 0]
    assert processing["negative_area_magnitude"][0] > 0.0
    assert len(processing["raw_input_sha256"]) == 64
    np.testing.assert_allclose(processing["final_shifted_integrals"], 1.0)
    ordinary = preflight_threepcf(config_path=CONFIGURATION)
    assert ordinary["ready"] is False
    assert any("Source input:" in message for message in ordinary["blockers"])
