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
from des3pcf.scaling_threepcf import _ia_model, _reuse_tables, _table_request
from des3pcf.scaling_threepcf import _point_parameters, _configuration
from des3pcf.inputs import save_tables


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


def test_nla_and_tatt_preserve_every_shared_random_parameter():
    """Discarding NLA-inactive draws must not move later calibration/source draws."""
    nla_path = PROJECT / "configs" / "des_y3_nla_scaling.yaml"
    tatt_path = PROJECT / "configs" / "des_y3_tatt_scaling.yaml"
    nla = threepcf_point_bank(config_path=nla_path)
    tatt = threepcf_point_bank(config_path=tatt_path)
    assert nla["ia_model"] == "NLA"
    assert tatt["ia_model"] == "TATT"
    assert nla["draw_settings"]["random_draw_bounds"] == (
        tatt["draw_settings"]["random_draw_bounds"]
    )
    inactive = ("a2", "eta2", "bta", "ct")
    nla_points = [nla["warmup_point"]] + nla["points"]
    tatt_points = [tatt["warmup_point"]] + tatt["points"]
    for nla_point, tatt_point in zip(nla_points, tatt_points):
        assert nla_point["seed"] == tatt_point["seed"]
        assert nla_point["parameters"].keys() == tatt_point["parameters"].keys()
        for name, value in nla_point["parameters"].items():
            if name in inactive:
                assert value == 0.0
            else:
                assert value == tatt_point["parameters"][name]
    nla_config = yaml.safe_load(nla_path.read_text())
    tatt_config = yaml.safe_load(tatt_path.read_text())
    for name in ("source", "rows", "mean"):
        assert nla_config[name] == tatt_config[name]
    for name, value in nla_config["physics"].items():
        if name != "tables":
            assert value == tatt_config["physics"][name]
    _, config, _, digest = _configuration(path=nla_path)
    changed = deepcopy(nla["points"][0])
    changed["parameters"]["bta"] = 0.1
    with pytest.raises(ValueError, match="holds bta fixed"):
        _point_parameters(point=changed, config=config, digest=digest)


def test_nla_declaration_refuses_active_quadratic_parameters():
    """An NLA label cannot hide a nonzero parameter override or another label."""
    path = PROJECT / "configs" / "des_y3_nla_scaling.yaml"
    config = yaml.safe_load(path.read_text())
    config["parameters"]["a2"] = 0.1
    with pytest.raises(ValueError, match="a2=0"):
        _ia_model(config=config)
    config["parameters"].pop("a2")
    config["notes"].append("Timing IA model: TATT.")
    with pytest.raises(ValueError, match="only one"):
        _ia_model(config=config)


def _write_reuse_fixture(tmp_path):
    """Write positive test arrays for identity checks, without a CAMB calculation.

    The archive follows the physical-table schema, but its powers are constant
    arrays. They are never passed to C or interpreted as a scientific model.
    """
    axes = {
        "k": np.array([0.01, 0.1, 1.0]),
        "z": np.array([0.0, 0.5, 1.0]),
        "los_z": np.array([0.1, 0.8]),
        "sources": np.tile([0.0, 1.0, 0.0], (4, 1)),
    }
    cosmology = {
        "omega_m": 0.3,
        "omega_b": 0.05,
        "h": 0.7,
        "ns": 0.96,
        "sigma8": 0.8,
        "w0": -1.0,
        "wa": 0.0,
        "mnu": 0.06,
        "num_massive_neutrinos": 3,
        "nnu": 3.046,
    }
    request = _table_request(
        cosmology=cosmology, halofit_version="takahashi", axes=axes,
        processing={"raw_input_sha256": "test-source", "policy": TIMING_SOURCE_POLICY},
    )
    tables = dict(axes)
    tables.update({
        "p_linear0": np.ones(3),
        "p_nonlinear": np.ones((3, 3)),
        "chi": np.array([0.0, 1000.0, 2000.0]),
        "dz_dchi": np.full(3, 0.0005),
        "growth": np.array([1.0, 0.8, 0.6]),
        "omega_m": 0.3,
        "omega_de": 0.7,
        "w0": -1.0,
        "wa": 0.0,
        "ns": 0.96,
        "sigma8": 0.8,
    })
    metadata = {
        "scaling_format": "cosmolike-threepcf-scaling-v1",
        "scaling_table_request": request,
        "producer": "CAMB",
        "parameters": cosmology,
        "halofit_version": "takahashi",
        "flat_background": True,
        "growth_definition": "linear sigma8(z)/sigma8(0)",
        "massive_neutrino_effective_growth": True,
        "configuration_sha256": "test-nla-config",
        "scaling_ia_model": "NLA",
        "camb_version": "synthetic-test-array; CAMB-was-not-run",
    }
    path = tmp_path / "test_nla_tables.npz"
    save_tables(path=path, tables=tables, metadata=metadata)
    return path, request, axes


def test_reuse_requires_the_same_physical_request_and_records_origin(tmp_path):
    """An IA change can retain physical tables; a cosmology or producer change cannot."""
    path, request, axes = _write_reuse_fixture(tmp_path=tmp_path)
    tables, metadata, origin = _reuse_tables(path=path, request=request, axes=axes)
    assert origin["ia_model"] == "NLA"
    assert origin["configuration_sha256"] == "test-nla-config"
    assert len(origin["sha256"]) == 64
    assert metadata["camb_version"] == "synthetic-test-array; CAMB-was-not-run"
    for name, expected in axes.items():
        np.testing.assert_array_equal(tables[name], expected)
    for key in ("halofit_version", "producer_source_sha256", "source_fits_sha256"):
        changed = deepcopy(request)
        changed[key] = "different-value"
        with pytest.raises(ValueError, match="physical preparation inputs"):
            _reuse_tables(path=path, request=changed, axes=axes)
    changed = deepcopy(request)
    changed["parameters"]["sigma8"] += 0.01
    with pytest.raises(ValueError, match="physical preparation inputs"):
        _reuse_tables(path=path, request=changed, axes=axes)


@pytest.mark.parametrize("name", ["k", "z", "los_z", "sources"])
def test_reuse_compares_actual_arrays_beyond_saved_request(tmp_path, name):
    """Matching metadata alone cannot conceal a different requested array value."""
    path, request, axes = _write_reuse_fixture(tmp_path=tmp_path)
    changed = deepcopy(axes)
    changed[name].flat[1] += 0.001
    with pytest.raises(ValueError, match=f"table {name} differs"):
        _reuse_tables(path=path, request=request, axes=changed)
