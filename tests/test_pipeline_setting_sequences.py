"""Compare notebook prediction caches with independently initialized processes.

The ordinary data-vector tests distinguish allocation sizes, refill values
and reset behavior. Here the C model owns its arrays; the additional retained
state is the Python provider and the mean used by Fisher differences. Every
changed setting below is evaluated after earlier settings in one process,
then compared as uint64 bits with a fresh interpreter at the final settings.

Small toy tables make this a state test. They are not a survey accuracy test
or a performance measurement. Each numerical change has a reach check on
the effective radial count, provider key or evaluated vector.
"""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "interface"))
sys.path.insert(0, str(PROJECT / "python"))

from des3pcf.accuracy import Accuracy
from des3pcf.forecast import ExactMeanCache
from des3pcf.mean import physics_model, XiEngine
from des3pcf.pipeline import PhysicalProvider, JointPrediction
from des3pcf.survey import forecast_rows
from test_pipeline import toy_tables


def make_prediction(compiled):
    """Prepare a small joint vector and its unboosted physical baseline.

    Returns:
      provider, prediction, original parameter dictionary and caller tables.
      The original tables remain writable so the ownership test can edit them.
    """
    tables, parameters = toy_tables()
    provider = PhysicalProvider(
        compiled=compiled, reference_tables=tables, reference_metadata={},
        reference_parameters=parameters, kernel_nz=9, table_boost=2, fit_nz=9,
    )
    rows = forecast_rows(
        nbin=2, aperture_radii=[0.003], xi_edges=[0.001, 0.002],
    )
    prediction = JointPrediction(
        compiled=compiled, rows=rows, provider=provider,
        ia_defaults={"a1": 0.2}, aperture_estimator="full_filter",
        aperture_settings={
            "projection_mode": "direct",
            "ell": {"lower": 30.0, "upper": 2000.0, "count": 5},
            "phi": {"lower": 0.0, "upper": 2.0 * np.pi, "count": 5},
        },
        xi_settings={
            "ell": {"lower": 30.0, "upper": 2000.0, "count": 7},
            "theta_count": 3,
            "pair_weight": "uniform_area",
        },
    )
    return provider, prediction, parameters, tables


def apply_case(provider, parameters, case):
    """Set a complete final state, never multiply a previous resolution.

    Arguments:
      provider = retained physical-table provider.
      parameters = original physical parameter dictionary, left unchanged.
      case = named single change or reset to baseline.
    Returns:
      parameter copy and expected number of line-of-sight nodes.
    """
    provider.kernel_nz = 9
    provider.table_boost = 2
    provider.fit_nz = 9
    provider.source_readers = False
    provider.accuracy = Accuracy()
    point = dict(parameters)
    expected_los = 4
    if case == "kernel_grow":
        provider.kernel_nz = 17
    elif case == "kernel_shrink":
        provider.kernel_nz = 5
    elif case == "reader_density":
        provider.table_boost = 3
    elif case == "fit_sampling":
        provider.fit_nz = 13
    elif case == "upstream_readers":
        provider.source_readers = True
    elif case == "global_sampling":
        provider.accuracy = Accuracy(accuracy_boost=2.0)
        expected_los = 7
    elif case == "global_integration":
        provider.accuracy = Accuracy(integration_accuracy_boost=2.0)
        expected_los = 7
    elif case == "chi_only":
        provider.accuracy = Accuracy(chi_boost=3.0)
        expected_los = 13
    elif case == "kernel_only":
        provider.accuracy = Accuracy(kernel_boost=2.0)
    elif case == "power_only":
        provider.accuracy = Accuracy(power_table_boost=2.0)
    elif case == "fit_only":
        provider.accuracy = Accuracy(fit_z_boost=2.0)
    elif case == "photoz_positive":
        point["dz1"] = 0.035
    elif case == "photoz_negative":
        point["dz2"] = -0.025
    elif case == "ia_amplitude":
        point["a1"] = 0.7
    elif case == "ia_slope":
        point["eta1"] = -1.3
    elif case == "calibration":
        point["m1"] = 0.03
    elif case != "baseline":
        raise ValueError(f"Unknown pipeline sequence case {case}.")
    return point, expected_los


def snapshot(provider, prediction, cache, parameters, case):
    """Evaluate through both prediction caches and assert the changed path is used."""
    point, expected_los = apply_case(provider=provider, parameters=parameters, case=case)
    first = cache(parameters=point)
    direct = prediction(parameters=point)
    np.testing.assert_array_equal(first.view(np.uint64), direct.view(np.uint64))
    model, tables, _ = provider.get(parameters=point)
    radial = model.radial()
    assert radial["z"].size == expected_los
    # A repeated request at the exact point must reuse its physical owner.
    assert provider.get(parameters=point)[0] is model
    return {
        "mean": first.copy(),
        "radial_z": radial["z"].copy(),
        "radial_lens": radial["lens"].copy(),
        "sources": tables["sources"].copy(),
    }


def fresh_snapshot(case, destination):
    """Run one setting in a process that has never evaluated another setting."""
    environment = dict(os.environ)
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    process = subprocess.run(
        args=[sys.executable, "-B", str(Path(__file__).resolve()),
              case, str(destination)],
        env=environment, capture_output=True, text=True, check=False, timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "runtime error:" not in process.stderr
    with np.load(destination, allow_pickle=False) as archive:
        result = {}
        for name in archive.files:
            result[name] = archive[name].copy()
    return result


def test_provider_and_fisher_cache_setting_sequence(compiled, tmp_path):
    """Grow/shrink, switch values, and restore an old state without stale means."""
    provider, prediction, parameters, _ = make_prediction(compiled=compiled)
    cache = ExactMeanCache(predict=prediction)
    cases = (
        "baseline", "kernel_grow", "kernel_shrink", "reader_density",
        "fit_sampling", "upstream_readers", "global_sampling", "global_integration",
        "chi_only", "kernel_only", "power_only", "fit_only", "photoz_positive",
        "photoz_negative", "ia_amplitude", "ia_slope", "calibration", "baseline",
    )
    original = None
    retained_model = None
    retained_power = None
    ell = np.array([45.0, 170.0, 1100.0])
    bins = np.array([0, 1], dtype=np.int32)
    for index, case in enumerate(cases):
        actual = snapshot(provider=provider, prediction=prediction, cache=cache,
                          parameters=parameters, case=case)
        reference = fresh_snapshot(case=case, destination=tmp_path / f"{index}.npz")
        for name, values in actual.items():
            np.testing.assert_array_equal(
                values.view(np.uint64), reference[name].view(np.uint64),
                err_msg=f"{case}: stale {name}",
            )
        if index == 0:
            original = actual["mean"].copy()
            retained_model = provider.model
            retained_power = retained_model.nla_power(ell=ell, bins=bins)
        elif case != "baseline":
            assert not np.array_equal(actual["mean"], original), case
        else:
            np.testing.assert_array_equal(actual["mean"].view(np.uint64),
                                          original.view(np.uint64))
        old_power = retained_model.nla_power(ell=ell, bins=bins)
        np.testing.assert_array_equal(old_power.view(np.uint64),
                                      retained_power.view(np.uint64))


def test_provider_and_mean_ownership_and_refusals(compiled):
    """Caller edits cannot alter only one of the physical table and cached mean."""
    provider, prediction, parameters, tables = make_prediction(compiled=compiled)
    cache = ExactMeanCache(predict=prediction)
    baseline = cache(parameters=parameters).copy()
    tables["sources"][:] = 0.0
    tables["p_nonlinear"] *= 3.0
    np.testing.assert_array_equal(cache(parameters=parameters), baseline)
    for array in (provider.base["k"], provider.model_tables["sources"],
                  prediction.rows.angles, prediction.ell, cache(parameters=parameters)):
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.flat[0] = 0.0
    with pytest.raises(TypeError):
        provider.model_tables["sources"] = tables["sources"]
    with pytest.raises(AttributeError, match="construct a new provider"):
        provider.base = tables
    with pytest.raises(AttributeError, match="construct a new engine"):
        prediction.ell = prediction.ell.copy()
    with pytest.raises(TypeError):
        prediction.projection_settings["nr"] = 12
    with pytest.raises(TypeError):
        prediction.ia_defaults["a1"] = 0.8
    # A previously cached valid point cannot hide an invalid new control.
    provider.table_boost = True
    with pytest.raises(ValueError, match="table_boost"):
        cache(parameters=parameters)
    provider.table_boost = 2
    provider.halofit_version = "different_prescription"
    with pytest.raises(ValueError, match="select camb or supply new tables"):
        cache(parameters=parameters)
    provider.halofit_version = "takahashi"
    np.testing.assert_array_equal(cache(parameters=parameters), baseline)
    returned = prediction(parameters=parameters)
    returned[:] = -3.0
    np.testing.assert_array_equal(prediction(parameters=parameters), baseline)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_boosted_model_factory_resolves_both_bindings(compiled, adapter):
    """The notebook constructor receives the same resolved physical arrays."""
    tables, _ = toy_tables()
    controls = Accuracy(chi_boost=2.0, kernel_boost=2.0)
    model = physics_model(compiled=compiled, tables=tables, kernel_nz=9,
                          table_boost=2, fit_nz=9, accuracy=controls, adapter=adapter)
    np.testing.assert_array_equal(model.radial()["z"][::2], tables["los_z"])
    assert model.radial()["z"].size == 7


def test_xi_geometry_copies_inputs_and_refuses_partial_changes(compiled):
    """An ell edit must not separate a Bessel grid from its retained quadrature."""
    rows = forecast_rows(nbin=2, aperture_radii=[], xi_edges=[0.001, 0.002])
    ell = np.array([30.0, 100.0, 500.0])
    weights = np.array([20.0, 80.0, 300.0])
    engine = XiEngine(compiled=compiled, rows=rows, ell=ell, dell=weights,
                      theta_count=3, pair_weight="uniform_area")
    spectra = np.ones((rows.values.size, ell.size)) * 1.0e-8
    baseline = engine.evaluate_spectra(spectra_e=spectra, spectra_b=spectra * 0.0)
    ell *= 2.0
    weights *= 0.5
    np.testing.assert_array_equal(
        engine.evaluate_spectra(spectra_e=spectra, spectra_b=spectra * 0.0), baseline,
    )
    with pytest.raises(ValueError):
        engine.ell.setflags(write=True)
    with pytest.raises(AttributeError, match="construct a new engine"):
        engine.theta = engine.theta.copy()


if __name__ == "__main__":
    import cosmolike_des_3pcf_interface as extension

    extension.set_omp_threads(nthreads=1)
    owner, predictor, point, _ = make_prediction(compiled=extension)
    retained = ExactMeanCache(predict=predictor)
    arrays = snapshot(provider=owner, prediction=predictor, cache=retained,
                      parameters=point, case=sys.argv[1])
    np.savez(sys.argv[2], **arrays)
