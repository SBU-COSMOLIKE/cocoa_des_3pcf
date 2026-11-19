"""Check the Python BiHalofit equation, supplied inputs and immutable tables.

The C comparison uses a saved copy of every input array and scalar. The
background-density samples in that comparison deliberately reproduce the
legacy C formula; a separate test varies the supplied background. Thus a
provider-background change cannot be mistaken for a port error. No survey
data vector, covariance or mask is read by these equation-level tests.
"""

from des3pcf.variance import reference_physics_model

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import mpmath
import numpy as np
import pytest
from scipy.integrate import quad, simpson

from des3pcf.bihalofit import (
    BaselineBispectrum,
    BiHalofit,
    BiHalofitProducer,
    _simpson_uniform,
    _upper_gamma,
    fit_redshifts,
    input_hashes,
)
from des3pcf.bihalofit_theory import BiHalofitTheory
from test_physics_model import physical_inputs


def comparison_inputs(source_readers=False, fit_nz=33):
    """Supply legacy Omega_m samples only for an identified equation test."""
    native = physical_inputs()
    native["source_readers"] = source_readers
    native["fit_nz"] = fit_nz
    anchor = fit_redshifts(zmax=native["z"][-1], fit_nz=fit_nz,
                           source_readers=source_readers)
    # This is a comparison fixture, not the provider implementation. Choosing
    # identical old background values isolates the numerical equation port.
    matter = native["omega_m"] * (1.0 + anchor)**3
    omega = matter / (matter + native["omega_de"])
    supplied = {}
    for name in ("k", "p_linear0", "z", "growth", "ns", "sigma8"):
        supplied[name] = native[name]
    supplied["omega_m_redshift"] = anchor
    supplied["omega_m_z"] = omega
    return native, supplied


def query_triangles():
    """Include unequal, flattened and strongly squeezed physical triangles."""
    sides = np.asarray([
        [0.03, 0.04, 0.05],
        [0.17, 0.11, 0.23],
        [0.8, 1.3, 1.8],
        [2.0, 2.3, 4.1],
        [0.02, 1.0, 1.01],
        [6.0, 9.0, 11.0],
        [1.0e-10, 1.0, 1.0],
        [0.1, 0.2, 0.3],
    ], dtype=np.float64)
    redshift = np.asarray([0.0, 0.15, 0.4, 0.8, 1.2, 1.9, 0.6, 2.0])
    return sides, redshift


def make_producer(block_size=65536, boost=2):
    """Use small explicit axes for ownership tests, not survey accuracy."""
    return BiHalofitProducer(
        z=np.linspace(start=0.0, stop=2.0, num=4),
        logk=np.linspace(start=-4.0, stop=2.0, num=5),
        logu=np.linspace(start=-5.0, stop=0.0, num=4),
        v=np.linspace(start=0.0, stop=1.0, num=4),
        block_size=block_size, boost=boost)


def evaluate(model, sides, redshift, component=0):
    """Map a [n,3] side array to the explicit broadcast query arguments."""
    return model.evaluate(k1=sides[:, 0], k2=sides[:, 1], k3=sides[:, 2],
                          z=redshift, component=component)


@pytest.mark.parametrize("source_readers", [False, True])
@pytest.mark.parametrize("fit_nz,table_boost", [(17, 4), (33, 8), (65, 16)])
def test_native_equation_with_pinned_inputs(compiled, tmp_path,
                                          source_readers, fit_nz, table_boost):
    """Compare all four diagnostics at three coefficient/read resolutions.

    The 2e-11 allowance is for library arithmetic, not interpolation error
    between baseline nodes. The comparison preserves the same quadrature,
    root tolerance and preparation controls in both languages. Measured
    errors and all input hashes are retained beside each private test result.
    """
    native, supplied = comparison_inputs(source_readers=source_readers,
                                         fit_nz=fit_nz)
    native["table_boost"] = table_boost
    sides, redshift = query_triangles()
    arrays = dict(native)
    arrays.update(supplied)
    arrays["query_sides"] = sides
    arrays["query_redshift"] = redshift
    input_path = tmp_path / "equation_inputs.npz"
    np.savez(file=input_path, **arrays)
    # Reload the actual bytes used by this calculation before either route
    # starts. A later reference regeneration elsewhere cannot alter them.
    with np.load(file=input_path, allow_pickle=False) as saved:
        for name in list(native):
            if isinstance(native[name], np.ndarray):
                native[name] = saved[name].copy()
        for name in list(supplied):
            if isinstance(supplied[name], np.ndarray):
                supplied[name] = saved[name].copy()
        sides = saved["query_sides"].copy()
        redshift = saved["query_redshift"].copy()
    model = BiHalofit(**supplied, fit_nz=fit_nz, table_boost=table_boost,
                      source_readers=source_readers)
    reference = reference_physics_model(factory=compiled.PhysicsModel, **native)
    report = {
        "inputs_npz_sha256": hashlib.sha256(
            input_path.read_bytes()).hexdigest(),
        "native_extension_sha256": hashlib.sha256(
            Path(compiled.__file__).read_bytes()).hexdigest(),
        "survey_data_covariance_mask": "not read by this equation test",
        "relative_errors": {},
    }
    for component in range(4):
        actual = evaluate(model=model, sides=sides, redshift=redshift,
                          component=component)
        expected = reference.matter(sides=sides, redshifts=redshift,
                                    component=component)
        relative = float(np.max(np.abs(actual / expected - 1.0)))
        report["relative_errors"][str(component)] = relative
        np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=0.0)
    (tmp_path / "comparison.json").write_text(json.dumps(report, indent=2))


def test_scalar_permutations_and_component_sum():
    """The full fit sums its pieces and is invariant under side relabeling."""
    unused, supplied = comparison_inputs()
    model = BiHalofit(**supplied)
    sides, redshift = query_triangles()
    full = evaluate(model=model, sides=sides, redshift=redshift)
    one = evaluate(model=model, sides=sides, redshift=redshift, component=2)
    three = evaluate(model=model, sides=sides, redshift=redshift, component=3)
    np.testing.assert_array_equal(full, one + three)
    for permutation in ((1, 2, 0), (2, 0, 1), (2, 1, 0)):
        permuted = evaluate(model=model, sides=sides[:, permutation],
                            redshift=redshift)
        np.testing.assert_array_equal(permuted, full)

    scalar = model.evaluate(k1=0.8, k2=1.3, k3=1.8, z=0.4)
    np.testing.assert_allclose(scalar, full[2], rtol=0.0, atol=0.0)
    equilateral = model.evaluate(k1=0.1, k2=0.1, k3=0.1, z=0.0, component=1)
    power = np.exp(model._linear.read(coordinates=np.log(0.1),
                                      endpoint_power=True))
    np.testing.assert_allclose(equilateral, (12.0 / 7.0) * power**2,
                               rtol=2.0e-15)


@pytest.mark.parametrize("source_readers", [False, True])
def test_shared_variance_python_and_c_equations(compiled, source_readers):
    """The same shared anchors must give the same fit in both languages.

    Supply the variance arrays explicitly, so differences cannot come from
    independent smoothing quadratures or a second sigma8 normalization.
    The existing equation tolerance covers library arithmetic only.
    """
    from des3pcf.variance import SharedVariance

    native, supplied = comparison_inputs(source_readers=source_readers)
    engine = SharedVariance(compiled=compiled)
    tables = dict(native)
    parameters = {"omega_b": 0.05, "h": 0.7}
    anchors = fit_redshifts(zmax=native["z"][-1], fit_nz=native["fit_nz"],
                            source_readers=source_readers)
    snapshot, unused = engine.prepare(
        tables=tables, parameters=parameters, metadata={}, z=anchors,
    )
    variance = {}
    for name in ("z", "rnl", "neff", "sigma8"):
        variance[name] = snapshot[name]
    native["variance"] = variance
    model = BiHalofit(**supplied, variance=variance, source_readers=source_readers)
    reference = reference_physics_model(factory=compiled.PhysicsModel, **native)
    sides, redshift = query_triangles()
    for component in range(4):
        actual = evaluate(model=model, sides=sides, redshift=redshift,
                          component=component)
        expected = reference.matter(sides=sides, redshifts=redshift,
                                    component=component)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=0.0)


@pytest.mark.parametrize("count", [7, 8, 1000, 2000])
def test_simpson_matches_independent_scipy(count):
    """The even-node closing interval must not be replaced by a trapezoid."""
    coordinates = np.linspace(start=-2.0, stop=3.0, num=count)
    values = np.stack((np.exp(0.3 * coordinates), coordinates**3), axis=0)
    spacing = (coordinates[-1] - coordinates[0]) / (count - 1)
    actual = _simpson_uniform(values=values, spacing=spacing)
    expected = simpson(y=values, dx=spacing, axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-15, atol=2.0e-15)


@pytest.mark.parametrize("shape", [-1.2, -1.0, -0.1, 0.0, 0.3, 1.5])
def test_gaussian_tail_without_positive_shape_assumption(shape):
    """Check upper gamma against its independently integrated definition."""
    lower = np.asarray([0.02, 0.5, 3.0])
    actual = _upper_gamma(shape=shape, lower=lower)
    expected = []
    for boundary in lower:
        def integrand(value):
            return value**(shape - 1.0) * np.exp(-value)
        value, error = quad(func=integrand, a=boundary, b=np.inf,
                            epsabs=2.0e-12, epsrel=2.0e-12)
        expected.append(value)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=2.0e-12)


@pytest.mark.parametrize("shape", [
    -1.000000000001,
    -1.0,
    -0.00100001,
    -0.001,
    -0.00099999,
    -1.0e-8,
    -1.0e-12,
    0.0,
    1.0e-12,
    1.0e-8,
    0.00099999,
    0.001,
    0.00100001,
])
def test_gaussian_tail_at_near_zero_shape(shape):
    """Compare special-function boundaries with an 80-digit independent value.

    The old recurrence loses significant digits when it divides a small
    subtraction by a shape near zero. These nodes cover both sides of the
    zero, the algorithm's shape boundary, a negative-integer recurrence
    crossing, the x=1 split, and the large-x underflow limit.
    """
    lower = np.asarray([
        1.0e-300,
        1.0e-12,
        1.0e-6,
        0.02,
        0.5,
        1.0,
        1.0 + 1.0e-12,
        3.0,
        10.0,
        100.0,
        700.0,
        1000.0,
    ])
    actual = _upper_gamma(shape=shape, lower=lower)
    expected = []
    with mpmath.workdps(80):
        for boundary in lower:
            value = mpmath.gammainc(mpmath.mpf(shape),
                                    mpmath.mpf(float(boundary)), mpmath.inf)
            expected.append(float(value))
    expected = np.asarray(expected)
    positive = expected > 0.0
    np.testing.assert_allclose(actual[positive], expected[positive],
                               rtol=2.0e-12, atol=0.0)
    np.testing.assert_array_equal(actual[~positive], expected[~positive])


def test_background_is_an_explicit_input():
    """Changing Omega_m changes d_n without reconstructing a w0-wa universe."""
    unused, supplied = comparison_inputs()
    original = BiHalofit(**supplied)
    supplied["omega_m_z"] = supplied["omega_m_z"] + 0.1
    changed = BiHalofit(**supplied)
    redshift = np.asarray([0.0, 0.5, 1.3, 2.0])
    old = original._coefficients(redshift=redshift)
    new = changed._coefficients(redshift=redshift)
    np.testing.assert_allclose(new["d"] / old["d"], 10.0**(-0.086 * 0.1),
                               rtol=3.0e-15)
    for name in ("radius", "loga", "b", "mu", "alpha_shape"):
        np.testing.assert_array_equal(new[name], old[name])


def test_table_matches_direct_nodes_and_block_sizes():
    """Bounded blocks retain every node and its C-order coordinate mapping."""
    unused, supplied = comparison_inputs()
    small_blocks = make_producer(block_size=7)
    one_block = make_producer(block_size=10000)
    first = small_blocks.prepare(tables=supplied)
    second = one_block.prepare(tables=supplied)
    np.testing.assert_array_equal(first.native["log_bispectrum"],
                                  second.native["log_bispectrum"])
    shape = (small_blocks.z.size, small_blocks.logk.size,
             small_blocks.logu.size, small_blocks.v.size)
    model = BiHalofit(**supplied)
    logarithm = first.native["log_bispectrum"].reshape(shape)
    for iz, ik, iu, iv in ((0, 0, 0, 0), (1, 2, 3, 1), (3, 4, 2, 3)):
        middle = np.exp(small_blocks.logk[ik])
        short = middle * np.exp(small_blocks.logu[iu])
        long = middle + small_blocks.v[iv] * short
        value = model.evaluate(k1=short, k2=middle, k3=long,
                               z=small_blocks.z[iz])
        np.testing.assert_allclose(logarithm[iz, ik, iu, iv], np.log(value),
                                   rtol=0.0, atol=0.0)
    assert set(first.native) == {
        "z", "logk", "logu", "v", "log_bispectrum", "boost", "clamp"}
    assert first.native["clamp"] == 0


@pytest.mark.parametrize("changed", [
    "k", "z", "omega_m_redshift",
    "p_linear0", "growth", "omega_m_z", "ns", "sigma8",
    "fit_nz", "table_boost", "source_readers", "variance_nodes",
    "slope_nodes", "radius_rtol",
])
def test_cache_keys_and_retained_old_outputs(changed):
    """Every consumed quantity rebuilds; A/B/A remains deterministic."""
    unused, supplied = comparison_inputs()
    producer = make_producer()
    original = producer.prepare(tables=supplied)
    saved = original.native["log_bispectrum"].copy()
    assert producer.prepare(tables=supplied) is original
    replacement = dict(supplied)
    controls = {}
    if changed == "k":
        replacement[changed] = supplied[changed] * 1.01
    elif changed == "z":
        replacement[changed] = supplied[changed].copy()
        replacement[changed][1:-1] += 0.001
    elif changed == "omega_m_redshift":
        replacement[changed] = supplied[changed].copy()
        replacement[changed][1:-1] += 0.001
    elif changed == "p_linear0":
        replacement[changed] = supplied[changed] * (supplied["k"] / 0.1)**0.01
    elif changed == "growth":
        replacement[changed] = supplied[changed] * np.exp(-0.01 * supplied["z"])
    elif changed == "omega_m_z":
        replacement[changed] = supplied[changed] + 0.02
    elif changed == "ns":
        replacement[changed] = 0.99
    elif changed == "sigma8":
        replacement[changed] = 0.82
    else:
        choices = {
            "fit_nz": 17,
            "table_boost": 4,
            "source_readers": True,
            "variance_nodes": 1001,
            "slope_nodes": 2001,
            "radius_rtol": 1.0e-6,
        }
        controls[changed] = choices[changed]
    updated = producer.prepare(tables=replacement, **controls)
    assert updated is not original
    assert updated.record()["input_sha256"] != original.record()["input_sha256"]
    restored = producer.prepare(tables=supplied)
    np.testing.assert_array_equal(restored.native["log_bispectrum"], saved)
    np.testing.assert_array_equal(original.native["log_bispectrum"], saved)
    with pytest.raises(ValueError):
        original.native["log_bispectrum"].setflags(write=True)
    with pytest.raises(AttributeError):
        producer.z = np.linspace(start=0.0, stop=3.0, num=4)


def test_failure_preserves_last_valid_cache():
    """An invalid replacement cannot become the cache's claimed input state."""
    unused, supplied = comparison_inputs()
    producer = make_producer()
    original = producer.prepare(tables=supplied)
    bad = dict(supplied)
    bad["p_linear0"] = -supplied["p_linear0"]
    with pytest.raises(ValueError, match="positive"):
        producer.prepare(tables=bad)
    assert producer.prepare(tables=supplied) is original
    record = original.record()
    record["model"] = "caller edit"
    assert original.record()["model"] != "caller edit"


def test_matching_background_preserves_supplied_bytes():
    """The default provider's model/background grids require no resampling."""
    unused, supplied = comparison_inputs()
    supplied["omega_m_redshift"] = supplied["z"].copy()
    supplied["omega_m_z"] = 0.3 + 0.1 * supplied["z"]
    baseline = make_producer().prepare(tables=supplied)
    assert baseline.background_omega_m_z.tobytes() == (
        supplied["omega_m_z"].tobytes())
    saved = baseline.background_omega_m_z.copy()
    supplied["omega_m_z"][:] = 0.8
    np.testing.assert_array_equal(baseline.background_omega_m_z, saved)
    with pytest.raises(ValueError):
        baseline.background_omega_m_z.setflags(write=True)


def test_shared_producer_never_pairs_one_key_with_another_result():
    """Independent Python callers may prepare together without mixed state."""
    unused, supplied = comparison_inputs()
    producer = make_producer()
    inputs = []
    expected = []
    for amplitude in (0.79, 0.83):
        tables = dict(supplied)
        tables["sigma8"] = amplitude
        inputs.append(tables)
        expected.append(producer.prepare(tables=tables))
    for repeat in range(3):
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = []
            for tables in inputs:
                futures.append(workers.submit(producer.prepare, tables=tables))
            for index, future in enumerate(futures):
                actual = future.result()
                np.testing.assert_array_equal(
                    actual.native["log_bispectrum"],
                    expected[index].native["log_bispectrum"])
        for index, tables in enumerate(inputs):
            actual = producer.prepare(tables=tables)
            np.testing.assert_array_equal(
                actual.native["log_bispectrum"],
                expected[index].native["log_bispectrum"])


@pytest.mark.parametrize("bad", ["shape", "irregular", "nonfinite"])
def test_invalid_baseline_is_rejected_before_native_construction(bad):
    """A finite uniform logarithmic table is part of the native contract."""
    axis = np.linspace(start=0.0, stop=1.0, num=4)
    values = np.zeros(shape=(4, 4, 4, 4))
    redshift = axis.copy()
    if bad == "shape":
        values = values[:, :, :, :3]
    elif bad == "irregular":
        redshift[1] += 0.01
    else:
        values[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        BaselineBispectrum(z=redshift, logk=axis, logu=axis - 1.0,
                           v=axis, log_bispectrum=values, metadata={})


def test_invalid_physical_queries_are_explicit():
    """The direct equation does not hide invalid triangles or extrapolate z."""
    unused, supplied = comparison_inputs()
    model = BiHalofit(**supplied)
    for sides in ((0.0, 1.0, 1.0), (0.1, 0.2, 1.0), (np.nan, 1.0, 1.0)):
        with pytest.raises(ValueError):
            model.evaluate(k1=sides[0], k2=sides[1], k3=sides[2], z=0.2)
    with pytest.raises(ValueError, match="outside"):
        model.evaluate(k1=1.0, k2=1.0, k3=1.0, z=2.1)
    with pytest.raises(ValueError, match="component"):
        model.evaluate(k1=1.0, k2=1.0, k3=1.0, z=0.2, component=4)


class ArrayPower:
    """Provide an analytic P in physical Mpc units for adapter unit checks."""

    def __init__(self, little_h):
        self.little_h = little_h

    def P(self, z, k):
        wave = np.asarray(k) / self.little_h
        power_h = 2.0e4 * wave / (1.0 + (wave / 0.15)**2)**1.8
        if np.ndim(z) == 1:
            return power_h[None, :] / (1.0 + np.asarray(z)[:, None])**2 / self.little_h**3
        return power_h / self.little_h**3


class ArrayProvider:
    """An explicitly supplied provider; no imported behavior is replaced."""

    def __init__(self):
        self.little_h = 0.7
        self.ns = 0.965
        self.sigma8 = 0.8
        self.pk_requests = []

    def get_Hubble(self, z, units):
        assert units == "km/s/Mpc"
        return np.full(np.asarray(z).shape, 100.0 * self.little_h)

    def get_sigma8_z(self, z):
        return self.sigma8 / (1.0 + np.asarray(z))

    def get_Omega_b(self, z):
        return 0.05 + 0.01 * np.asarray(z)

    def get_Omega_cdm(self, z):
        return 0.25 + 0.02 * np.asarray(z)

    def get_Omega_nu_massive(self, z):
        return 0.01 + 0.003 * np.asarray(z)

    def get_param(self, name):
        assert name == "ns"
        return self.ns

    def get_Pk_interpolator(self, **arguments):
        self.pk_requests.append(arguments)
        return ArrayPower(little_h=self.little_h)


@pytest.mark.parametrize("variance_source,variance_growth", [
    ("local", "factorized"),
    ("shared", "factorized"),
    ("shared", "evolving"),
])
def test_actual_theory_base_and_provider_contract(variance_source, variance_growth):
    """Exercise Cobaya's real Theory state management with explicit arrays."""
    background = np.linspace(start=0.0, stop=2.0, num=5)
    theory = BiHalofitTheory(info={
        "baseline_z": np.linspace(start=0.0, stop=2.0, num=4).tolist(),
        "baseline_logk": np.linspace(start=-4.0, stop=2.0, num=4).tolist(),
        "baseline_logu": np.linspace(start=-5.0, stop=0.0, num=4).tolist(),
        "baseline_v": np.linspace(start=0.0, stop=1.0, num=4).tolist(),
        "background_z": background.tolist(),
        "power_k": np.geomspace(start=1.0e-5, stop=1.0e3, num=96).tolist(),
        "stop_at_error": True,
        "variance_source": variance_source,
        "variance_growth": variance_growth,
    })
    provider = ArrayProvider()
    theory.initialize_with_provider(provider=provider)
    requirements = theory.get_requirements()
    assert requirements["Pk_interpolator"]["k_max"] == 5.0
    assert requirements["Pk_interpolator"]["nonlinear"] is False
    assert "Omega_nu_massive" in requirements
    assert theory.check_cache_and_compute(params_values_dict={},
                                          dependency_params=[provider.sigma8])
    baseline = theory.get_bihalofit_baseline()
    sampled = theory.get_bihalofit_background()
    expected = (provider.get_Omega_b(z=background)
                + provider.get_Omega_cdm(z=background)
                + provider.get_Omega_nu_massive(z=background))
    np.testing.assert_allclose(sampled["omega_m_z"], expected, rtol=2.0e-15)
    assert theory.check_cache_and_compute(params_values_dict={},
                                          dependency_params=[provider.sigma8])
    assert theory.get_bihalofit_baseline() is baseline
    provider.sigma8 = 0.82
    assert theory.check_cache_and_compute(params_values_dict={},
                                          dependency_params=[provider.sigma8])
    changed = theory.get_bihalofit_baseline()
    assert changed is not baseline
    if variance_source == "shared":
        # This explicit provider changes only its sigma8 scalar. Its P and
        # normalized growth stay fixed, so shared smoothing keeps B fixed.
        np.testing.assert_array_equal(changed.native["log_bispectrum"],
                                      baseline.native["log_bispectrum"])
        assert theory.current_state["bihalofit_variance_record"]["growth_mode"] == variance_growth
    assert provider.pk_requests[0]["extrap_kmax"] > 700.0
    assert provider.pk_requests[0]["var_pair"] == ("delta_tot", "delta_tot")
    # Check the raw input hash as well: sigma8 renormalization could hide a
    # missing h^3 amplitude factor in a comparison of final bispectra alone.
    raw_power = provider.get_Pk_interpolator(
        nonlinear=False).P(z=0.0, k=provider.little_h * theory._power_k)
    raw_power *= provider.little_h**3
    raw_hash = input_hashes(arrays={"p_linear0": raw_power}, settings={})
    actual_hash = changed.record()["input_sha256"]["p_linear0"]
    assert actual_hash == raw_hash["p_linear0"]
    table = {
        "k": theory._power_k,
        "p_linear0": ArrayPower(little_h=1.0).P(z=0.0, k=theory._power_k),
        "z": background,
        "growth": 1.0 / (1.0 + background),
        "omega_m_z": provider.get_Omega_b(z=theory._omega_z)
                       + provider.get_Omega_cdm(z=theory._omega_z)
                       + provider.get_Omega_nu_massive(z=theory._omega_z),
        "omega_m_redshift": theory._omega_z,
        "ns": provider.ns,
        "sigma8": provider.sigma8,
    }
    variance = None
    if variance_source == "shared":
        table["omega_m"] = float(table["omega_m_z"][0])
        table["p_linear"] = table["growth"][:, None]**2 * table["p_linear0"][None, :]
        snapshot, unused = theory._variance_engine.prepare(
            tables=table, parameters={"omega_b": 0.05, "h": provider.little_h},
            metadata={"omega_nu_h2": 0.01 * provider.little_h**2},
            z=theory._fit_z, growth_mode=variance_growth,
        )
        variance = {}
        for name in ("z", "rnl", "neff", "sigma8"):
            variance[name] = snapshot[name]
    expected_owner = theory._producer.prepare(tables=table, variance=variance)
    np.testing.assert_allclose(changed.native["log_bispectrum"],
                               expected_owner.native["log_bispectrum"],
                               rtol=2.0e-14, atol=2.0e-14)
