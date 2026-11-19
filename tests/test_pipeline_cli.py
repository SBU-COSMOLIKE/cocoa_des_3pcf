"""Exercise the command-line workflow on small declared physical tables.

No test runs CAMB or a survey forecast. The end-to-end mean uses eight
full-filter aperture rows from two smooth toy source distributions; its
only varied parameters are multiplicative calibration factors. The fixed
covariance is a labeled test matrix, not a DES noise model.
"""

import json
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest
import yaml

from des3pcf.inputs import save_tables
from des3pcf.survey import forecast_rows, ARCMIN_TO_RADIAN


PROJECT = Path(__file__).resolve().parents[1]
DRIVER = PROJECT / "scripts" / "run_3pcf.py"


def run_driver(command, config, validate=False):
    """Run the CLI in an isolated process and preserve its diagnostics."""
    arguments = [sys.executable, "-B", str(DRIVER), command, str(config)]
    if validate:
        arguments.append("--validate-config")
    return subprocess.run(args=arguments, capture_output=True, text=True,
                          check=False, timeout=30)


@pytest.fixture
def toy_configuration(tmp_path):
    """Write deterministic tables and eight labeled rows beside their config."""
    wave = np.geomspace(start=1.0e-5, stop=1.0e3, num=96)
    redshift = np.linspace(start=0, stop=2, num=41)
    linear = 2.0e4 * wave / (1 + (wave / 0.15) ** 2) ** 1.8
    nonlinear0 = linear * (1 + (wave / 0.35) ** 2) ** 0.22
    sources = np.empty((2, redshift.size))
    sources[0] = redshift ** 2 * np.exp(-(redshift / 0.6) ** 1.5)
    sources[1] = redshift ** 2 * np.exp(-(redshift / 0.95) ** 1.5)
    tables = {
        "k": wave,
        "p_linear0": linear,
        "z": redshift,
        "p_nonlinear": nonlinear0[None, :] / (1 + redshift[:, None]) ** 2.4,
        "chi": 3000 * redshift / (1 + 0.35 * redshift),
        "dz_dchi": (1 + 0.35 * redshift) ** 2 / 3000,
        "growth": 1 / (1 + redshift),
        "sources": sources,
        "los_z": np.array([0.15, 0.5, 1.0, 1.8]),
        "omega_m": 0.3,
        "omega_de": 0.7,
        "w0": -1.0,
        "wa": 0.0,
        "ns": 0.965,
        "sigma8": 0.8,
        "omega_m_z": 0.3 * (1 + redshift)**3 / (0.3 * (1 + redshift)**3 + 0.7),
    }
    save_tables(path=tmp_path / "physics.npz", tables=tables,
                metadata={"producer": "controlled CLI test"})
    rows = forecast_rows(nbin=2,
                         aperture_radii=np.array([7, 14]) * ARCMIN_TO_RADIAN)
    covariance = np.eye(rows.values.size) * 1.0e-14
    np.savez(file=tmp_path / "covariance.npz", covariance=covariance,
             row_labels=rows.labels)
    np.savez(file=tmp_path / "sample.npz", covariance=covariance,
             row_labels=rows.labels)
    config = {
        "format": "cosmolike-3pcf-run-v1",
        "description": "Controlled small CLI test; not DES data.",
        "threads": 1,
        "source": {"path": None, "extension": "nz_source", "nbin": 2},
        "physics": {"tables": "physics.npz",
                    "cosmology_mode": "fixed_tables", "kernel_nz": 101},
        "parameters": {
            "omega_m": 0.3,
            "omega_b": 0.05,
            "h": 0.7,
            "ns": 0.965,
            "sigma8": 0.8,
            "w0": -1.0,
            "wa": 0.0,
            "m1": 0.0,
            "m2": 0.0,
        },
        "ia": {"a1": 0.0, "a2": 0.0, "bta": 0.0, "ct": 0.0},
        "rows": {"mode": "forecast", "aperture_radii_arcmin": [7, 14]},
        "mean": {
            "aperture_estimator": "full_filter",
            "aperture_settings": {
                "projection_mode": "direct",
                "ell": {"lower": 20, "upper": 3000,
                        "count": 8, "logarithmic": True},
                "phi": {"lower": 0, "upper": 2 * np.pi, "count": 6},
            },
            "two_point": {"mode": "isolated_nla"},
        },
        "covariance": {"mode": "file", "path": "covariance.npz",
                       "estimator": "full_filter"},
        "fisher": {
            "steps": {"m1": 0.002, "m2": 0.002},
            "priors": {},
            "fom_pairs": [["m1", "m2"]],
            "precision": {"mode": "inverse"},
        },
        "outputs": {"evaluate": "mean.npz", "fisher": "fisher.npz"},
    }
    path = tmp_path / "example.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, config, rows


def test_public_examples_dry_check_without_physics():
    """Y6 missing source input is reported without generating a substitute."""
    for name in ("y3_full_filter_example.json", "y3_finite_sas_example.json",
                 "y6_full_filter_example.json", "des_y3_tatt_fisher.yaml",
                 "des_y3_nla_fisher.yaml"):
        config_path = PROJECT / "configs" / name
        result = run_driver(command="prepare", config=config_path,
                            validate=True)
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["schema_valid"] is True
        if name.startswith("y6"):
            assert report["missing_inputs"] == [{"input": "source FITS",
                                                  "path": None}]


@pytest.mark.parametrize("padding", (None, "zero_at_observer_and_background_edge",
                                    "zero_at_measured_edges"))
def test_pchip_requires_declared_source_count_and_padding(toy_configuration, padding):
    """Selecting a reconstruction cannot silently invent endpoint probabilities."""
    path, config, _ = toy_configuration
    config["physics"]["source_interpolation"] = "pchip_midpoint"
    if padding is not None:
        config["physics"]["source_nz"] = 65
        config["physics"]["source_padding"] = padding
        config["accuracy"] = {"source_boost": 2.0}
    path.write_text(json.dumps(config), encoding="utf-8")
    result = run_driver(command="fisher", config=path, validate=True)
    if padding is not None:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["schema_valid"] is True
    else:
        assert result.returncode != 0
        assert "PCHIP needs" in result.stderr


@pytest.mark.parametrize("bispectrum_source", ["internal", "supplied"])
@pytest.mark.parametrize("variance_source", ["local", "shared"])
def test_evaluate_and_fixed_covariance_fisher(toy_configuration, bispectrum_source,
                                            variance_source):
    """Keep all rows and check calibration derivatives against their algebra."""
    path, config, rows = toy_configuration
    config["physics"]["variance_source"] = variance_source
    if bispectrum_source == "supplied":
        config["physics"]["bispectrum_source"] = "supplied"
        config["physics"]["bispectrum_table"] = supplied_grid()
    path.write_text(json.dumps(config), encoding="utf-8")
    result = run_driver(command="evaluate", config=path)
    assert result.returncode == 0, result.stderr
    result = run_driver(command="fisher", config=path)
    assert result.returncode == 0, result.stderr
    with np.load(path.parent / "mean.npz", allow_pickle=False) as mean_file:
        mean = mean_file["mean"]
        mean_metadata = json.loads(str(mean_file["metadata_json"].item()))
        np.testing.assert_array_equal(mean_file["row_labels"], rows.labels)
    with np.load(path.parent / "fisher.npz", allow_pickle=False) as result_file:
        metadata = json.loads(str(result_file["metadata_json"].item()))
        assert metadata["compression"] == "none"
        assert metadata["covariance_is_fixed"] is True
        assert metadata["parameter_rank_status"] == "resolved"
        if bispectrum_source == "supplied":
            recorded = metadata["baseline_bispectrum"]
            assert recorded == mean_metadata["baseline_bispectrum"]
            assert recorded["native_input_sha256"]
            assert recorded["input_sha256"]
            assert recorded["controls"]["fit_nz"] == 33
        else:
            assert "baseline_bispectrum" not in metadata
        np.testing.assert_array_equal(result_file["mean"], mean)
        assert result_file["derivatives_h2"].shape == (8, 2)
        assert metadata["stencil_offsets"] == [-2.0, -1.0, 0.0, 1.0, 2.0]
        for label in ("h", "h2"):
            samples = result_file[f"stencil_values_{label}"]
            assert samples.shape == (2, 5, 8)
            np.testing.assert_array_equal(samples[:, 2], np.tile(mean, (2, 1)))
        for source in range(2):
            # Each occurrence of source i contributes (1+m_i), hence
            # dMap3/dm_i at m=0 is occurrence_count * Map3.
            count = np.sum(rows.bins == source, axis=1)
            expected = mean * count
            step = 0.5 * config["fisher"]["steps"][f"m{source + 1}"]
            for node, offset in enumerate((-2.0, -1.0, 0.0, 1.0, 2.0)):
                calibrated = mean * (1.0 + offset * step)**count
                np.testing.assert_allclose(
                    actual=result_file["stencil_values_h2"][source, node],
                    desired=calibrated, rtol=3.0e-15, atol=0.0,
                )
            # A zero physical derivative still subtracts nearly equal
            # floating-point means. Its absolute error scales as eps/h.
            cancellation = 4 * np.finfo(float).eps * np.max(np.abs(mean))
            cancellation /= step
            np.testing.assert_allclose(result_file["derivatives_h2"][:, source],
                                       expected, rtol=2.0e-11,
                                       atol=cancellation)
    again = run_driver(command="evaluate", config=path)
    assert again.returncode == 2
    assert "--overwrite" in again.stderr


def supplied_grid():
    """Return broad, coarse axes for dispatch and archive checks.

    These counts keep the toy CLI calculation short; they do not establish
    bispectrum interpolation accuracy. logk and logu contain logarithms
    already, so their stored coordinates are spaced linearly.
    """
    return {
        "z": {"lower": 0.0, "upper": 2.0, "count": 7},
        "logk": {"lower": -12.0, "upper": 12.0, "count": 17},
        "logu": {"lower": -30.0, "upper": 0.0, "count": 17},
        "v": {"lower": 0.0, "upper": 1.0, "count": 7},
        "boost": 1,
    }


def test_pme_keeps_joint_labels_and_saves_precision(toy_configuration):
    """The optional statistical route uses the whole supplied sample matrix."""
    path, config, rows = toy_configuration
    config["fisher"]["precision"] = {
        "mode": "pme",
        "sample_covariance": {"path": "sample.npz", "key": "covariance"},
        "independent_count": 100,
        "order": 2,
        "sample_assumption": "independent_gaussian_vectors",
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    result = run_driver(command="fisher", config=path)
    assert result.returncode == 0, result.stderr
    with np.load(path.parent / "fisher.npz", allow_pickle=False) as archive:
        assert archive["precision_pme"].shape == (8, 8)
        np.testing.assert_array_equal(archive["row_labels"], rows.labels)
        metadata = json.loads(str(archive["metadata_json"].item()))
        diagnostics = metadata["precision_expansion_diagnostics"]
        assert diagnostics["independent_count"] == 100


def test_estimator_mismatch_and_unmatched_covariance_refused(toy_configuration):
    """Equal matrix size does not justify an estimator or row permutation."""
    path, config, rows = toy_configuration
    config["covariance"]["estimator"] = "finite_sas"
    path.write_text(json.dumps(config), encoding="utf-8")
    result = run_driver(command="fisher", config=path, validate=True)
    assert result.returncode == 2
    assert "same aperture estimator" in result.stderr
    config["covariance"]["estimator"] = "full_filter"
    path.write_text(json.dumps(config), encoding="utf-8")
    np.savez(file=path.parent / "covariance.npz",
             covariance=np.eye(8), row_labels=rows.labels[::-1])
    result = run_driver(command="fisher", config=path)
    assert result.returncode == 2
    assert "row labels" in result.stderr


def test_unknown_config_field_is_not_ignored(toy_configuration):
    """A mistyped scientific choice must fail before any prediction."""
    path, config, _ = toy_configuration
    config["mean"]["aperture_estimatr"] = "finite_sas"
    path.write_text(json.dumps(config), encoding="utf-8")
    result = run_driver(command="evaluate", config=path, validate=True)
    assert result.returncode == 2
    assert "aperture_estimatr" in result.stderr


def test_profile_nodes_include_zero_only_for_positive_logarithmic_grid():
    """The explicit zero does not replace a requested positive radial node."""
    grid = runpy.run_path(str(DRIVER))["grid_values"]
    settings = {"lower": 1.0e-7, "upper": 0.5, "count": 8,
                "logarithmic": True, "include_zero": True}
    nodes = grid(settings=settings)
    assert nodes.size == 9
    assert nodes[0] == 0
    np.testing.assert_array_equal(
        nodes[1:], np.geomspace(start=1.0e-7, stop=0.5, num=8),
    )
    settings["logarithmic"] = False
    with pytest.raises(ValueError, match="positive logarithmic"):
        grid(settings=settings)


def test_analytic_covariance_requires_declared_physical_limits(
    toy_configuration,
):
    """Reject omitted calibration or neutrino physics before C is imported."""
    path, config, _ = toy_configuration
    config["covariance"] = {"mode": "analytic",
                            "ia_covariance": "gravitational_only"}
    config["parameters"]["m1"] = 0.01
    path.write_text(json.dumps(config), encoding="utf-8")
    result = run_driver(command="covariance", config=path, validate=True)
    assert result.returncode == 2
    assert "fiducial m_i=0" in result.stderr
    config["parameters"]["m1"] = 0.0
    config["parameters"]["mnu"] = 0.06
    path.write_text(json.dumps(config), encoding="utf-8")
    result = run_driver(command="covariance", config=path, validate=True)
    assert result.returncode == 2
    assert "single_fluid_effective_growth" in result.stderr
    config["covariance"]["neutrino_model"] = "single_fluid_effective_growth"
    config["physics"]["variance_source"] = "shared"
    config["physics"]["variance_growth"] = "evolving"
    path.write_text(json.dumps(config), encoding="utf-8")
    result = run_driver(command="covariance", config=path, validate=True)
    assert result.returncode == 2
    assert "requires factorized variance" in result.stderr


@pytest.mark.parametrize("save_components", [False, True])
@pytest.mark.parametrize("bispectrum_source", ["internal", "supplied"])
@pytest.mark.parametrize("variance_source", ["local", "shared"])
def test_covariance_command_preserves_each_term(
    toy_configuration, save_components, bispectrum_source, variance_source,
):
    """The saved diagnostic choice leaves the same final physical covariance."""
    path, config, rows = toy_configuration
    config["physics"]["variance_source"] = variance_source
    if bispectrum_source == "supplied":
        config["physics"]["bispectrum_source"] = "supplied"
        config["physics"]["bispectrum_table"] = supplied_grid()
    config["covariance"] = {
        "mode": "analytic",
        "save_components": save_components,
        "path": "analytic_covariance.npz",
        "ia_covariance": "gravitational_only",
        "noise": {"number_density": [1.5, 1.4],
                  "sigma_component": [0.25, 0.27]},
        "physical": {
            "area": 0.2,
            "center_window": 1,
            "large_area": 1,
            "estimator": 0,
            "ia_model": 0,
            "st_a": 0.707,
            "st_p": 0.3,
            "st_norm": 0.3222,
            "delta_c": 1.686,
            "collapse_model": 0,
            "concentration_a": 10.14,
            "concentration_b": -0.081,
            "concentration_c": -1.01,
            "concentration_pivot": 2.0e12,
            "baryon_strength": 0.0,
        },
        "grids": {
            "ell": {"lower": 20, "upper": 3000,
                    "count": 8, "logarithmic": True},
            "window_ell": {"lower": 0.02, "upper": 300,
                           "count": 16, "logarithmic": True},
            "phi": {"lower": 0, "upper": 2 * np.pi, "count": 6},
            "radius": {"lower": 1.0e-6, "upper": 0.02,
                       "count": 12, "logarithmic": True},
            "profile_radius": {"lower": 1.0e-7, "upper": 0.03,
                               "count": 32, "logarithmic": True,
                               "include_zero": True},
            "k": {"lower": 1.0e-5, "upper": 500,
                  "count": 64, "logarithmic": True},
            "mass": {"lower": 1.0e11, "upper": 2.0e14,
                     "count": 2, "logarithmic": True},
        },
        "theta_count": 2,
        "pair_weight": "uniform_area",
    }
    if not save_components:
        del config["covariance"]["save_components"]
    path.write_text(json.dumps(config), encoding="utf-8")
    result = run_driver(command="covariance", config=path)
    assert result.returncode == 0, result.stderr
    output = path.parent / "analytic_covariance.npz"
    with np.load(output, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["row_labels"], rows.labels)
        metadata = json.loads(str(archive["metadata_json"].item()))
        assert metadata["covariance_components_saved"] is save_components
        if variance_source == "shared":
            assert metadata["halo_variance"]["matter_field"] == "total"
            assert metadata["variance"]["growth_mode"] == "factorized"
        input_files = metadata["prediction_identity"]["input_files"]
        assert "physical_tables" in input_files
        assert "joint_covariance" not in input_files
        if bispectrum_source == "supplied":
            assert metadata["baseline_bispectrum"]["native_input_sha256"]
        assert archive["covariance"].shape == (8, 8)
        assert archive["covariance_model_mean"].shape == (8,)
        if save_components:
            assert archive["terms"].shape == (6, 8, 8)
            np.testing.assert_array_equal(archive["covariance"],
                                          np.sum(archive["terms"], axis=0))
            for index, name in enumerate(metadata["covariance_term_names"]):
                expected = archive["terms"][index]
                np.testing.assert_array_equal(archive[name], expected)
        else:
            assert "terms" not in archive.files
            for name in metadata["covariance_term_names"]:
                assert name not in archive.files


def test_yaml_and_json_resolve_the_same_scientific_configuration(toy_configuration):
    """YAML changes input readability without changing parsed physical choices."""
    path, config, _ = toy_configuration
    yaml_path = path.with_suffix(".yaml")
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    load = runpy.run_path(str(DRIVER))["load_configuration"]
    from_json, _, json_digest = load(path=path, command="fisher")
    from_yaml, _, yaml_digest = load(path=yaml_path, command="fisher")
    assert from_json == from_yaml
    assert json_digest != yaml_digest
    result = run_driver(command="evaluate", config=yaml_path)
    assert result.returncode == 0, result.stderr


def test_core_controls_are_checked_without_importing_a_compiled_module(
    toy_configuration,
):
    """A mistyped CFASTPT control fails before physical preparation or C state."""
    path, config, _ = toy_configuration
    config["mean"]["two_point"] = {
        "mode": "core_tatt",
        "directory": "missing/interface",
        "module": "missing_core",
        "accuracy_boost": 1.0,
        "integration_accuracy": 8,
        "fpt_internal_boost": 0.5,
        "ell_internal": 192,
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    result = run_driver(command="evaluate", config=path, validate=True)
    assert result.returncode == 2
    assert "integration_accuracy" in result.stderr


def test_joint_cli_saves_conditional_information_and_checks_result_identity(
    toy_configuration,
):
    """Exercise the actual small joint path, retaining a nonzero cross block."""
    import cosmolike_des_3pcf_fisher_wrappers as notebook

    path, config, _ = toy_configuration
    config["rows"]["xi_edges_arcmin"] = [10.0, 20.0, 40.0]
    config["mean"]["xi_settings"] = {
        "ell": {"lower": 20.0, "upper": 3000.0,
                "count": 16, "logarithmic": True},
        "theta_count": 4,
        "pair_weight": "uniform_area",
    }
    config["fisher"]["priors"] = {"m1": 0.04}
    rows = forecast_rows(
        nbin=2, aperture_radii=np.array([7, 14]) * ARCMIN_TO_RADIAN,
        xi_edges=np.array([10, 20, 40]) * ARCMIN_TO_RADIAN,
    )
    scales = np.where(rows.kind == "map3", 1.0e-7, 1.0e-4)
    matrix = np.eye(rows.values.size) + 0.05
    matrix *= scales[:, None] * scales[None, :]
    np.savez(file=path.parent / "covariance.npz",
             covariance=matrix, row_labels=rows.labels)
    path.write_text(json.dumps(config), encoding="utf-8")
    execution = run_driver(command="fisher", config=path)
    assert execution.returncode == 0, execution.stderr
    workflow = notebook.FisherWorkflow(configuration=path)
    result, metadata = workflow.load_result()
    assert result["mean"].shape == (20,)
    assert metadata["comparison_rank_status"] == {
        "two_point": "resolved", "joint": "resolved",
    }
    np.testing.assert_allclose(result["fisher_h2"], result["fisher_joint"])
    np.testing.assert_allclose(
        result["fisher_two_point_likelihood"]
        + result["fisher_conditional_three_point"],
        result["fisher_joint_likelihood"],
    )
    np.testing.assert_allclose(result["fisher_prior"], [[625.0, 0.0], [0.0, 0.0]])
    config["description"] = "A changed experiment must not relabel the old file."
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="different configuration"):
        workflow.load_result()
