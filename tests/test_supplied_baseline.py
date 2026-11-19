"""Exercise supplied absolute matter B separately from a feedback response.

The analytic B tables below are numerical probes, not cosmological fits.
Their logarithms are multilinear in the four stored coordinates, so the
expected off-grid answer is known without invoking BiHalofit. Other checks
retain P and IA while replacing B, test ownership after input mutation,
and follow the core bridge through replacement, clearing and resetting.
"""

from des3pcf.variance import reference_physics_model

import gc
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from data_vector.test_physics_model import compact_grid_settings, physical_inputs


def baseline_input(*, amplitude=7.0, boost=2, varying=True, clamp=0, zmax=2.0):
    """Return a fresh analytic ln B table in the native coordinate convention.

    Arguments:
        amplitude: Positive B normalization in (Mpc/h)^6.
        boost: Dense intervals per input interval.
        varying: Include known multilinear coordinate dependence when true.
        clamp: Explicit boundary policy; zero rejects queries outside the table.

    Returns:
        Dictionary of owned arrays and integer interpolation controls.
    """
    redshift = np.linspace(start=0.0, stop=zmax, num=4)
    logk = np.linspace(start=-16.0, stop=16.0, num=5)
    logu = np.linspace(start=-40.0, stop=0.0, num=6)
    shape = np.linspace(start=0.0, stop=1.0, num=4)
    coordinates = np.meshgrid(redshift, logk, logu, shape, indexing="ij")
    values = np.full(shape=coordinates[0].shape, fill_value=np.log(amplitude))
    if varying:
        values += 0.03 * coordinates[0] + 0.02 * coordinates[1]
        values += 0.01 * coordinates[2] + 0.04 * coordinates[3]
        values += 0.005 * coordinates[0] * coordinates[1]
        values += 0.003 * coordinates[2] * coordinates[3]
    return {
        "z": redshift,
        "logk": logk,
        "logu": logu,
        "v": shape,
        "log_bispectrum": values.ravel(order="C"),
        "boost": boost,
        "clamp": clamp,
    }


def supplied_inputs():
    """Return physical arrays with an explicit background, without fit scalars.

    Returns:
        Fresh model inputs. Omega_m(z) is a controlled constant test table,
        deliberately distinct from the internal w0-wa background formula.
    """
    inputs = physical_inputs()
    for name in ("omega_de", "w0", "wa", "ns"):
        del inputs[name]
    inputs["omega_m_z"] = np.full(shape=inputs["z"].shape, fill_value=0.3)
    return inputs


def analytic_bispectrum(*, sides, redshifts, amplitude):
    """Evaluate the known multilinear ln B function at physical triangles.

    Arguments:
        sides: Positive triangle magnitudes [n,3], in h/Mpc, any side order.
        redshifts: One redshift per triangle [n].
        amplitude: B normalization in (Mpc/h)^6.

    Returns:
        Absolute B values [n], independent of native interpolation code.
    """
    sorted_sides = np.sort(a=sides, axis=1)
    logk = np.log(sorted_sides[:, 1])
    logu = np.log(sorted_sides[:, 0] / sorted_sides[:, 1])
    shape = (sorted_sides[:, 2] - sorted_sides[:, 1]) / sorted_sides[:, 0]
    logarithm = np.log(amplitude) + 0.03 * redshifts + 0.02 * logk
    logarithm += 0.01 * logu + 0.04 * shape
    logarithm += 0.005 * redshifts * logk + 0.003 * logu * shape
    return np.exp(logarithm)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
@pytest.mark.parametrize("boost", [1, 3])
def test_off_grid_baseline_and_input_ownership(compiled, adapter, boost):
    """Off-grid values, side permutations and copied storage share one answer."""
    baseline = baseline_input(boost=boost)
    inputs = supplied_inputs()
    model = reference_physics_model(factory=getattr(compiled, adapter).PhysicsModelWithBispectrum,
        **inputs, baseline=baseline
    )
    baseline["log_bispectrum"][:] = 100.0
    inputs["omega_m_z"][:] = 0.9
    sides = np.array([
        [0.3, 0.4, 0.5],
        [1.2, 0.8, 1.0],
        [1.0e-12, 1.0, 1.0 + 5.0e-13],
    ])
    redshifts = np.array([0.25, 1.4, 0.7])
    expected = analytic_bispectrum(sides=sides, redshifts=redshifts, amplitude=7.0)
    fractions = model.background_matter_fraction(redshifts=redshifts)
    np.testing.assert_allclose(actual=fractions, desired=0.3, rtol=4.0e-15)
    actual = model.matter(sides=sides, redshifts=redshifts)
    copied = model.matter_notebook(sides=sides[:, ::-1], redshifts=redshifts)
    np.testing.assert_allclose(actual=actual, desired=expected, rtol=2.0e-14)
    np.testing.assert_allclose(actual=copied, desired=expected, rtol=2.0e-14)


@pytest.mark.parametrize("source_readers", [False, True])
def test_tree_power_and_raw_alignment_unchanged(compiled, source_readers):
    """Replacing full B leaves tree, P and every raw IA component unchanged."""
    ordinary_inputs = physical_inputs()
    ordinary_inputs["source_readers"] = source_readers
    supplied = supplied_inputs()
    supplied["source_readers"] = source_readers
    original = reference_physics_model(factory=compiled.PhysicsModel, **ordinary_inputs)
    replacement = reference_physics_model(factory=compiled.production.PhysicsModelWithBispectrum,
        **supplied, baseline=baseline_input()
    )
    sides = np.array([[0.03, 0.04, 0.05], [0.2, 0.3, 0.4]])
    redshifts = np.array([0.2, 1.3])
    np.testing.assert_array_equal(
        x=replacement.matter(sides=sides, redshifts=redshifts, component=1),
        y=original.matter(sides=sides, redshifts=redshifts, component=1),
    )
    parameters = {
        "a1": 0.4,
        "a2": 0.7,
        "bta": 0.9,
        "ct": 0.2,
    }
    for mode in range(3):
        parameters["power_mode"] = mode
        np.testing.assert_array_equal(
            x=replacement.ia(
                sides=sides, redshifts=redshifts, parameters=parameters
            ),
            y=original.ia(
                sides=sides, redshifts=redshifts, parameters=parameters
            ),
        )
        np.testing.assert_array_equal(
            x=replacement.power(k=sides[:, 0].copy(), z=redshifts, mode=mode),
            y=original.power(k=sides[:, 0].copy(), z=redshifts, mode=mode),
        )


def test_constant_baseline_projection_and_retained_grid(compiled):
    """A constant B gives its exact radial-weight sum and survives model deletion."""
    model = reference_physics_model(factory=compiled.production.PhysicsModelWithBispectrum,
        **supplied_inputs(), baseline=baseline_input(varying=False)
    )
    bins = np.array([0, 1, 0], dtype=np.int32)
    sides = np.array([[120.0, 180.0, 250.0], [200.0, 270.0, 340.0]])
    radial = model.radial()
    kernel_product = radial["lens"][0] * radial["lens"][1] * radial["lens"][0]
    # radial.lens stores W/[chi(1+z)], the same finite kernel used by GGG.
    measure = radial["dchi"] * (1.0 + radial["z"])**3 / radial["chi"]
    expected = 7.0 * np.sum(measure * kernel_product)
    actual = model.projected(sides=sides, bins=bins, parameters={"a1": 0.0})
    np.testing.assert_allclose(actual=actual, desired=expected, rtol=4.0e-14)
    settings = compact_grid_settings(reference_mode=0)
    grid = model.projection_grid(bins=bins, parameters={"a1": 0.0}, settings=settings)
    before = grid.projected(sides=sides)
    del model
    gc.collect()
    np.testing.assert_array_equal(x=grid.projected(sides=sides), y=before)


def test_supplied_baseline_and_feedback_remain_distinct(compiled):
    """R_B multiplies the selected B baseline while quadratic IA stays fixed."""
    baseline = baseline_input(varying=False)
    response = {}
    for name in ("z", "logk", "logu", "v"):
        response[name] = baseline[name].copy()
    response["log_power"] = np.full(
        shape=response["z"].size * response["logk"].size, fill_value=np.log(0.8)
    )
    response["log_bispectrum"] = np.full(
        shape=baseline["log_bispectrum"].size, fill_value=np.log(0.6)
    )
    response["boost"] = 2
    response["clamp"] = 0
    model = reference_physics_model(factory=compiled.production.PhysicsModelWithBispectrum,
        **supplied_inputs(), baseline=baseline, feedback=response
    )
    parameters = {
        "a1": 0.4,
        "a2": 0.7,
        "bta": 0.9,
        "ct": 0.2,
    }
    bins = np.array([0, 1, 0], dtype=np.int32)
    sides = np.array([[120.0, 180.0, 250.0], [200.0, 270.0, 340.0]])
    reference_baseline = model.projected(
        sides=sides, bins=bins, parameters=parameters, component=1
    )
    reference_residual = model.projected(
        sides=sides, bins=bins, parameters=parameters, component=2
    )
    parameters["feedback_amplitude"] = 1.0
    changed_baseline = model.projected(
        sides=sides, bins=bins, parameters=parameters, component=1
    )
    changed_residual = model.projected(
        sides=sides, bins=bins, parameters=parameters, component=2
    )
    np.testing.assert_allclose(
        actual=changed_baseline, desired=0.6 * reference_baseline, rtol=4.0e-14
    )
    np.testing.assert_array_equal(x=changed_residual, y=reference_residual)


def test_covariance_uses_supplied_b_in_each_b_family(compiled):
    """Doubling B doubles the cubic mean/PB and quadruples BB, with fixed P."""
    from covariance.test_physical_covariance import covariance_arguments

    inputs = supplied_inputs()
    inputs["los_z"] = np.array([0.15, 0.5, 1.0, 1.8])
    first = reference_physics_model(factory=compiled.production.PhysicsModelWithBispectrum,
        **inputs, baseline=baseline_input(amplitude=7.0, varying=False)
    )
    second = reference_physics_model(factory=compiled.production.PhysicsModelWithBispectrum,
        **inputs, baseline=baseline_input(amplitude=14.0, varying=False)
    )
    request = covariance_arguments()
    mean_first, terms_first = compiled.production.physical_covariance(
        model=first, inputs=request
    )
    mean_second, terms_second = compiled.production.physical_covariance(
        model=second, inputs=request
    )
    cubic = request["row_kind"] == 1
    np.testing.assert_allclose(
        actual=mean_second[cubic], desired=2.0 * mean_first[cubic], rtol=4.0e-14
    )
    np.testing.assert_array_equal(x=mean_second[~cubic], y=mean_first[~cubic])
    # The six families are Gaussian, BB, P*T, P*B, connected, and SSC.
    # Only BB and P*B contain the selected matter bispectrum. Other halo
    # moments keep their separate model; no fitted one-halo B is requested.
    for term in (0, 2, 4, 5):
        np.testing.assert_array_equal(x=terms_second[term], y=terms_first[term])
    assert np.any(terms_first[1] != 0.0)
    assert np.any(terms_first[3] != 0.0)
    np.testing.assert_allclose(
        actual=terms_second[1], desired=4.0 * terms_first[1], rtol=8.0e-14
    )
    np.testing.assert_allclose(
        actual=terms_second[3], desired=2.0 * terms_first[3], rtol=8.0e-14
    )


def install_bridge(*, compiled, inputs, install_scalars):
    """Install small controlled arrays in core structs, without building a model.

    Arguments:
        compiled: Current compiled extension.
        inputs: physical_inputs dictionary.
        install_scalars: Install legacy fit parameters only when requested.

    Returns:
        Nothing. Replaces the process-global core state after initial_setup.
    """
    core = compiled.core
    core.initial_setup()
    core.set_cosmological_parameters(
        omega_matter=0.3, omega_baryon=0.05, hubble=70.0, omega_nu_h2=0.0
    )
    logarithmic_wave = np.log10(inputs["k"])
    linear = inputs["growth"][:, None]**2 * inputs["p_linear0"][None, :]
    core.set_linear_power_spectrum(
        log10k=logarithmic_wave, z=inputs["z"], lnP=np.log(linear).ravel(order="F")
    )
    core.set_non_linear_power_spectrum(
        log10k=logarithmic_wave,
        z=inputs["z"],
        lnP=np.log(inputs["p_nonlinear"]).ravel(order="F"),
    )
    core.set_distances(z=inputs["z"], chi=inputs["chi"])
    core.set_growth(z=inputs["z"], G=(1.0 + inputs["z"]) * inputs["growth"])
    core.set_source_sample_size(ntomo=2)
    source_columns = np.column_stack((inputs["z"], inputs["sources"].T))
    core.set_source_sample(nofz=np.ascontiguousarray(source_columns))
    core.set_nuisance_shear_photoz(bias=np.zeros(shape=2))
    core.set_los(z=inputs["los_z"])
    core.set_resolution(kernel_nz=101, source_readers=1, table_boost=8, fit_nz=33)
    if install_scalars:
        core.set_scalars(ns=0.965, sigma8=0.8, w0=-1.0, wa=0.0)


def test_bridge_replacement_clear_and_owned_lifetime(compiled):
    """A/B/A rebuilds, identical resends reuse, and clearing restores the fit."""
    inputs = physical_inputs()
    install_bridge(compiled=compiled, inputs=inputs, install_scalars=False)
    core = compiled.core
    matter_fraction = np.full(shape=inputs["z"].shape, fill_value=0.3)
    first_table = baseline_input(varying=False)
    core.set_baseline_bispectrum(
        baseline=first_table, omega_m_z=matter_fraction, sigma8=0.8
    )
    first = core.model_from_core()
    core.set_baseline_bispectrum(
        baseline=first_table, omega_m_z=matter_fraction, sigma8=0.8
    )
    assert core.model_from_core() is first
    sides = np.array([[0.3, 0.4, 0.5]])
    redshifts = np.array([0.4])
    expected = first.matter(sides=sides, redshifts=redshifts)
    second_table = baseline_input(amplitude=13.0, varying=False, boost=3)
    core.set_baseline_bispectrum(
        baseline=second_table, omega_m_z=matter_fraction, sigma8=0.8
    )
    second = core.model_from_core()
    assert second is not first
    second_table["log_bispectrum"][:] = 100.0
    np.testing.assert_allclose(
        actual=second.matter(sides=sides, redshifts=redshifts), desired=13.0
    )
    np.testing.assert_array_equal(
        x=first.matter(sides=sides, redshifts=redshifts), y=expected
    )
    core.set_baseline_bispectrum(
        baseline=first_table, omega_m_z=matter_fraction, sigma8=0.8
    )
    restored = core.model_from_core()
    np.testing.assert_array_equal(
        x=restored.matter(sides=sides, redshifts=redshifts), y=expected
    )
    core.set_baseline_bispectrum(baseline=None)
    # Clearing removes the input; it never chooses a second matter model.
    # Refusal is tested in a child below because native errors are fatal.
    core.initial_setup()
    np.testing.assert_array_equal(
        x=restored.matter(sides=sides, redshifts=redshifts), y=expected
    )
    install_bridge(compiled=compiled, inputs=inputs, install_scalars=True)
    core.set_baseline_bispectrum(
        baseline=first_table, omega_m_z=matter_fraction, sigma8=0.8,
    )
    fresh = core.model_from_core()
    np.testing.assert_array_equal(
        x=fresh.matter(sides=sides, redshifts=redshifts), y=expected
    )
    core.initial_setup()


@pytest.mark.parametrize("field", [
    "z",
    "logk",
    "logu",
    "v",
    "log_bispectrum",
    "boost",
    "clamp",
    "omega_m_z",
    "sigma8",
    "signed_zero",
    "z_count",
    "k_count",
    "u_count",
    "v_count",
])
def test_every_supplied_bridge_value_invalidates(compiled, field):
    """A same-size value change builds once; its identical resend reuses."""
    inputs = physical_inputs()
    install_bridge(compiled=compiled, inputs=inputs, install_scalars=False)
    baseline = baseline_input(varying=False)
    if field == "signed_zero":
        baseline["log_bispectrum"][0] = 0.0
    matter_fraction = np.full(shape=inputs["z"].shape, fill_value=0.3)
    sigma8 = 0.8
    core = compiled.core
    core.set_baseline_bispectrum(
        baseline=baseline, omega_m_z=matter_fraction, sigma8=sigma8
    )
    original = core.model_from_core()
    if field == "z":
        baseline["z"] *= 1.01
    elif field == "logk":
        baseline["logk"] += 0.01
    elif field == "logu":
        baseline["logu"] *= 1.01
    elif field == "v":
        baseline["v"] *= 0.99
    elif field == "log_bispectrum":
        baseline["log_bispectrum"][0] += 0.01
    elif field == "boost":
        baseline["boost"] = 3
    elif field == "clamp":
        baseline["clamp"] = 1
    elif field == "omega_m_z":
        matter_fraction[3] += 0.01
    elif field == "sigma8":
        sigma8 = 0.81
    elif field == "signed_zero":
        baseline["log_bispectrum"][0] = -0.0
    else:
        resized_axis = {
            "z_count": "z",
            "k_count": "logk",
            "u_count": "logu",
            "v_count": "v",
        }[field]
        original_axis = baseline[resized_axis]
        baseline[resized_axis] = np.linspace(
            start=original_axis[0],
            stop=original_axis[-1],
            num=original_axis.size + 1,
        )
        size = 1
        for name in ("z", "logk", "logu", "v"):
            size *= baseline[name].size
        baseline["log_bispectrum"] = np.full(shape=size, fill_value=np.log(7.0))
    core.set_baseline_bispectrum(
        baseline=baseline, omega_m_z=matter_fraction, sigma8=sigma8
    )
    changed = core.model_from_core()
    assert changed is not original
    core.set_baseline_bispectrum(
        baseline=baseline, omega_m_z=matter_fraction, sigma8=sigma8
    )
    assert core.model_from_core() is changed
    core.initial_setup()


@pytest.mark.parametrize("case", [
    "one_halo",
    "three_halo",
    "missing_baseline",
    "cleared_baseline",
    "stale_baseline",
    "domain",
    "nonfinite",
    "background_size",
])
def test_supplied_refusals_in_fresh_process(compiled, case):
    """Unsupported pieces and out-of-domain/nonfinite tables exit explicitly."""
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["THREEPCF_TEST_INTERFACE"] = str(Path(compiled.__file__).parent)
    run = subprocess.run(
        args=[sys.executable, str(Path(__file__).resolve()), case],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert run.returncode == 1, run.stdout + run.stderr
    expected = {
        "missing_baseline": "theory-supplied matter bispectrum",
        "cleared_baseline": "theory-supplied matter bispectrum",
        "stale_baseline": "supplied matter bispectrum is stale",
        "one_halo": "no fitted one-/three-halo",
        "three_halo": "no fitted one-/three-halo",
        "domain": "exceeds its declared table domain",
        "nonfinite": "nonfinite",
        "background_size": "Omega_m(z) must match its redshift axis",
    }
    assert expected[case] in run.stdout + run.stderr


def refusal_child(*, case):
    """Run one deliberate fatal query in a child, protecting pytest's process.

    Arguments:
        case: Refusal scenario named by test_supplied_refusals_in_fresh_process.

    Returns:
        Nothing; C is expected to terminate the child with exit code one.
    """
    project = Path(__file__).resolve().parents[1]
    interface = os.environ.get("THREEPCF_TEST_INTERFACE", str(project / "interface"))
    sys.path.insert(0, interface)
    import cosmolike_des_3pcf_interface as compiled

    compiled.set_omp_threads(nthreads=1)
    if case in ("missing_baseline", "cleared_baseline", "stale_baseline"):
        inputs = physical_inputs()
        install_bridge(compiled=compiled, inputs=inputs, install_scalars=False)
        if case != "missing_baseline":
            compiled.core.set_baseline_bispectrum(
                baseline=baseline_input(),
                omega_m_z=np.full(inputs["z"].shape, 0.3), sigma8=0.8,
            )
            compiled.core.model_from_core()
        if case == "cleared_baseline":
            compiled.core.set_baseline_bispectrum(baseline=None)
        if case == "stale_baseline":
            compiled.core.set_cosmological_parameters(
                omega_matter=0.31, omega_baryon=0.05, hubble=70.0,
                omega_nu_h2=0.0,
            )
        compiled.core.model_from_core()
        raise AssertionError("missing/stale B was accepted")
    baseline = baseline_input()
    if case == "nonfinite":
        baseline["log_bispectrum"][0] = np.nan
    inputs = supplied_inputs()
    if case == "background_size":
        inputs["omega_m_z"] = inputs["omega_m_z"][:-1].copy()
    model = reference_physics_model(factory=compiled.production.PhysicsModelWithBispectrum,
        **inputs, baseline=baseline
    )
    component = 0
    if case == "one_halo":
        component = 2
    if case == "three_halo":
        component = 3
    scale = 0.3
    if case == "domain":
        scale = 1.0e10
    sides = np.array([[scale, scale, scale]])
    model.matter(sides=sides, redshifts=np.array([0.4]), component=component)
    raise AssertionError("the requested refusal did not stop the native query")


if __name__ == "__main__":
    refusal_child(case=sys.argv[1])
