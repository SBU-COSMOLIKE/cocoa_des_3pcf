"""Physical checks of the halo, joint covariance and xi-bin calculations.

The small grids in the assembly checks test normalization, term selection,
noise rules and ownership. They are deliberately not a DES accuracy study.
Separate tests use resolved quadrature and analytic or independently
integrated references for the halo and xi transforms. No timing is measured.
"""

from theory.bihalofit.variance import reference_physics_model

from pathlib import Path
import subprocess
import sys

import mpmath
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.integrate import simpson
from scipy.special import hyp1f1, j0, j1, jv, spherical_jn


def logarithmic_rule(lower, upper, count):
    """Return positive nodes and weights for an integral over dx.

    A Gauss-Legendre rule is constructed in ln(x). Multiplication by x
    converts its weights into dx; the C routines add no such Jacobian.
    """
    nodes, weights = leggauss(deg=count)
    half = 0.5 * np.log(upper / lower)
    center = 0.5 * np.log(upper * lower)
    values = np.exp(center + half * nodes)
    return values, weights * half * values


def make_covariance_model(compiled):
    """Construct a smooth test cosmology with two different source samples."""
    wave = np.geomspace(start=1.0e-5, stop=1.0e3, num=96)
    redshift = np.linspace(start=0.0, stop=2.0, num=41)
    linear = 2.0e4 * wave / (1.0 + (wave / 0.15) ** 2) ** 1.8
    nonlinear0 = linear * (1.0 + (wave / 0.35) ** 2) ** 0.22
    nonlinear = nonlinear0[None, :] / (1.0 + redshift[:, None]) ** 2.4
    sources = np.empty(shape=(2, redshift.size), dtype=np.float64)
    sources[0] = redshift**2 * np.exp(-(redshift / 0.6) ** 1.5)
    sources[1] = redshift**2 * np.exp(-(redshift / 0.95) ** 1.5)
    return reference_physics_model(factory=compiled.PhysicsModel,
        k=wave,
        p_linear0=linear,
        z=redshift,
        p_nonlinear=nonlinear,
        chi=3000.0 * redshift / (1.0 + 0.35 * redshift),
        dz_dchi=(1.0 + 0.35 * redshift) ** 2 / 3000.0,
        growth=1.0 / (1.0 + redshift),
        sources=sources,
        omega_m=0.3,
        omega_de=0.7,
        w0=-1.0,
        wa=0.0,
        ns=0.965,
        sigma8=0.8,
        los_z=np.asarray([0.15, 0.5, 1.0, 1.8], dtype=np.float64),
        kernel_nz=101,
    )


@pytest.fixture(scope="module")
def covariance_model(compiled):
    """Reuse the controlled model within the in-process physical checks."""
    return make_covariance_model(compiled=compiled)


def covariance_arguments():
    """Return a small full-aperture/xi request, with each grid named.

    Fields zero and one are apertures in source bins zero and one. Fields
    two and three are unfiltered shear in those bins. The rows are Map3,
    xi+, xi-, and Map2. The wider profile grid encloses both displaced xi
    endpoints even at the largest halo-offset integration node.
    """
    ell, dell = logarithmic_rule(lower=10.0, upper=8000.0, count=8)
    window, dwindow = logarithmic_rule(lower=0.02, upper=300.0, count=28)
    radius, dr = logarithmic_rule(lower=1.0e-6, upper=0.03, count=32)
    wave, dwave = logarithmic_rule(lower=1.0e-5, upper=500.0, count=192)
    mass, dmass = logarithmic_rule(lower=1.0e11, upper=2.0e14, count=3)
    nangle = 8
    angle = (np.arange(nangle) + 0.5) * (2.0 * np.pi / nangle)
    theta = np.zeros(shape=(4, 2), dtype=np.float64)
    theta[1:3] = [0.002, 0.003]
    theta_weight = np.zeros_like(theta)
    theta_weight[1:3] = [0.25, 0.75]
    return {
        "ell": ell,
        "dell": dell,
        "window_ell": window,
        "window_dell": dwindow,
        "phi": angle,
        "dphi": np.full(shape=nangle, fill_value=2.0 * np.pi / nangle),
        "radius": radius,
        "dr": dr,
        "profile_radius": np.concatenate(([0.0], np.geomspace(1.0e-7, 0.04, 96))),
        "k": wave,
        "dk": dwave,
        "mass": mass,
        "dmass": dmass,
        "field_source": np.asarray([0, 1, 0, 1], dtype=np.int32),
        "field_radius": np.asarray([0.002, 0.003, 0.0, 0.0]),
        "row_kind": np.asarray([1, 2, 3, 0], dtype=np.int32),
        "row_fields": np.asarray(
            [[0, 0, 1], [2, 3, 0], [2, 3, 0], [0, 1, 0]], dtype=np.int32
        ),
        "theta": theta,
        "theta_weight": theta_weight,
        "noise": np.diag([2.0e-9, 3.0e-9]),
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
        "profile_upsample": 8,
        "concentration_a": 10.14,
        "concentration_b": -0.081,
        "concentration_c": -1.01,
        "concentration_pivot": 2.0e12,
        "baryon_strength": 0.0,
    }


@pytest.mark.parametrize("count", [255, 256])
def test_xi_bins_against_gaussian_spectrum(compiled, count):
    """A Gaussian spectrum has analytic J0 and J4 transforms.

    This checks the Fourier 1/(2*pi), separation weights, sign of B power,
    and independent even/odd row counts in both adapters. The J4 reference
    uses the confluent hypergeometric integral, not a Bessel quadrature.
    """
    ell, dell = logarithmic_rule(lower=1.0e-6, upper=50.0, count=count)
    scale = 0.2
    b_fraction = 0.3
    theta = np.asarray([[0.2, 0.4, 0.7], [0.3, 0.6, 1.0], [0.1, 0.5, 0.8]])
    weights = np.asarray([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.4, 0.2, 0.4]])
    signs = np.asarray([1, -1, 1], dtype=np.int32)
    power = np.tile(np.exp(-scale * ell**2), reps=(3, 1))
    bpower = b_fraction * power
    expected = np.empty(shape=3)
    for row in range(3):
        argument = theta[row] ** 2 / (4.0 * scale)
        if signs[row] == 1:
            values = (1.0 + b_fraction) * np.exp(-argument) / (4.0 * np.pi * scale)
        else:
            values = (1.0 - b_fraction) * theta[row] ** 4
            values *= hyp1f1(3.0, 5.0, -argument) / (768.0 * np.pi * scale**3)
        expected[row] = np.sum(weights[row] * values)
    for adapter in (compiled.production, compiled.notebook):
        actual = adapter.xi_from_power(
            ell=ell,
            dell=dell,
            theta=theta,
            theta_weight=weights,
            spectra_e=power,
            spectra_b=bpower,
            sign=signs,
        )
        np.testing.assert_allclose(actual, expected, rtol=3.0e-12, atol=1.0e-14)


def test_nfw_against_radial_mass_integral(compiled, covariance_model):
    """Integrate the real NFW density independently of the Si/Ci expression."""
    inputs = covariance_arguments()
    redshift = np.asarray([0.0, 0.7])
    wave = np.asarray([0.0, 1.0e-7, 0.01, 0.3, 3.0])
    result = compiled.production.halo_tables(
        model=covariance_model, inputs=inputs, z=redshift, k_profile=wave
    )
    sigma2, abundance, bias, concentration, actual = result
    nodes, weights = leggauss(deg=256)
    rho = 2.77536627e11 * 0.3
    expected = np.empty_like(actual)
    for iz in range(redshift.size):
        for im in range(inputs["mass"].size):
            mass = inputs["mass"][im]
            c = concentration[iz, im]
            radius = (3.0 * mass / (800.0 * np.pi * rho)) ** (1.0 / 3.0)
            radial = 0.5 * c * (nodes + 1.0)
            measure = 0.5 * c * weights * radial / (1.0 + radial) ** 2
            normalization = np.log1p(c) - c / (1.0 + c)
            for ik in range(wave.size):
                phase = wave[ik] * radius * radial / c
                expected[iz, im, ik] = np.sum(measure * np.sinc(phase / np.pi))
                expected[iz, im, ik] /= normalization
    np.testing.assert_allclose(actual, expected, rtol=3.0e-10, atol=2.0e-13)
    np.testing.assert_array_equal(actual[:, :, 0], np.ones_like(sigma2))
    assert np.all(sigma2 > 0.0)
    assert np.all(abundance > 0.0)
    assert np.all(bias > 0.0)


def nfw_reference(concentration, phase):
    """Return the normalized truncated profile with 70-digit Si/Ci values.

    Arguments:
        concentration = positive r200/r_s, dimensionless.
        phase = one-dimensional float64 array of nonnegative q=k*r200.

    Returns:
        array = u(c,q), with the shape of phase and exact value one at zero.

    Si(x) integrates sin(t)/t from zero to x; Ci(x) is minus the integral
    of cos(t)/t from x to infinity. Integrating the NFW radial mass weight
    gives their differences at x=q/c and (1+c)*x, plus the outer boundary
    term. Dividing by ln(1+c)-c/(1+c) normalizes the enclosed mass.

    High precision resolves the subtraction of oscillatory Si/Ci values
    when q is large. The separate radial-integral test checks this physical
    transform without using special functions. Neither reference reads the
    production f/G interpolation table.
    """
    expected = np.empty_like(phase)
    with mpmath.workdps(70):
        c = mpmath.mpf(float(concentration))
        normalization = mpmath.log1p(c) - c / (1 + c)
        for index, value in enumerate(phase):
            if value == 0.0:
                expected[index] = 1.0
                continue
            q = mpmath.mpf(float(value))
            x = q / c
            outer = (1 + c) * x
            numerator = mpmath.sin(x) * (mpmath.si(outer) - mpmath.si(x))
            numerator += mpmath.cos(x) * (mpmath.ci(outer) - mpmath.ci(x))
            numerator -= mpmath.sin(q) / outer
            expected[index] = float(numerator / normalization)
    return expected


@pytest.mark.parametrize("concentration", [1.0, 3.0, 10.0, 30.0])
def test_nfw_dimensionless_domain(compiled, covariance_model, concentration):
    """Check the production profile from zero through q=20000.

    The transform depends on concentration and q=k*r200, not on mass and
    k separately. A single mass therefore spans this numerical domain by
    varying k. Fixing the concentration coefficients isolates the profile
    reader from the Duffy mass/redshift law. This is not a covariance or
    concentration-prescription convergence test.

    Include both sides of q=1e-4 (the small-phase expansion), x=50 and
    (1+c)*x=50 (the two table/asymptotic transitions). The 2e-8 relative
    allowance encloses the measured 1.17e-8 maximum in the independent
    524288-node comparison over c=1..30, q<=10000. The 2e-13 absolute
    allowance also tests the small profile values in the extended tail.
    The stricter, resolved radial-integral test retains its own tolerance.
    """
    inputs = covariance_arguments()
    mass = 1.0e12
    inputs["mass"] = np.asarray([mass])
    inputs["dmass"] = np.asarray([mass])
    inputs["concentration_a"] = concentration
    inputs["concentration_b"] = 0.0
    inputs["concentration_c"] = 0.0
    rho = 2.77536627e11 * 0.3
    radius = np.cbrt(3.0 * mass / (4.0 * np.pi * rho)) / np.cbrt(200.0)
    phase = list(np.geomspace(start=1.0e-8, stop=2.0e4, num=41))
    phase.append(0.0)
    for boundary in (1.0e-4, 50.0 * concentration / (1.0 + concentration),
                     50.0 * concentration):
        for factor in (1.0 - 2.0e-8, 1.0, 1.0 + 2.0e-8):
            phase.append(boundary * factor)
    phase = np.unique(np.asarray(phase, dtype=np.float64))
    wave = phase / radius
    # Evaluate the reference at the rounded supplied k values, using the
    # halo_tables mass-radius expression. C and NumPy cube roots can differ
    # in their last bit; this test compares the physical transform within
    # the stated tolerance rather than demanding identical radius bits.
    expected = nfw_reference(concentration=concentration, phase=wave * radius)
    for adapter in (compiled.production, compiled.notebook):
        result = adapter.halo_tables(
            model=covariance_model, inputs=inputs, z=np.asarray([0.0]),
            k_profile=wave,
        )
        actual_concentration = result[3][0, 0]
        actual = result[4][0, 0]
        assert actual_concentration == concentration
        assert actual[0] == 1.0
        np.testing.assert_allclose(actual, expected, rtol=2.0e-8, atol=2.0e-13)


def test_halo_bias_as_abundance_response(compiled, covariance_model):
    """Universal mass-function bias obeys b=1-d ln(n)/d delta_c.

    The derivative changes the collapse threshold while retaining the
    actual linear power and mass grid. It checks the response coefficient
    used in the selected long-mode covariance independently of its formula.
    """
    inputs = covariance_arguments()
    redshift = np.asarray([0.2, 0.8])
    wave = np.asarray([0.0, 0.1])
    baseline = compiled.production.halo_tables(
        model=covariance_model, inputs=inputs, z=redshift, k_profile=wave
    )
    step = 1.0e-5
    low = dict(inputs)
    high = dict(inputs)
    low["delta_c"] -= step
    high["delta_c"] += step
    below = compiled.production.halo_tables(
        model=covariance_model, inputs=low, z=redshift, k_profile=wave
    )[1]
    above = compiled.production.halo_tables(
        model=covariance_model, inputs=high, z=redshift, k_profile=wave
    )[1]
    expected = 1.0 - (np.log(above) - np.log(below)) / (2.0 * step)
    np.testing.assert_allclose(baseline[2], expected, rtol=2.0e-9, atol=2.0e-9)


def test_halo_variance_against_independent_rule(compiled, covariance_model):
    """Check the top-hat variance with a different quadrature and Bessel form.

    The reference uses W(x)=3*j_1(x)/x and Simpson integration in ln(k).
    This can expose a missing k Jacobian, Fourier normalization, or wrong
    conversion from halo mass to the Lagrangian smoothing radius.
    """
    inputs = covariance_arguments()
    inputs["k"], inputs["dk"] = logarithmic_rule(
        lower=1.0e-5, upper=500.0, count=512
    )
    redshift = np.asarray([0.0, 0.7])
    actual = compiled.production.halo_tables(
        model=covariance_model,
        inputs=inputs,
        z=redshift,
        k_profile=np.asarray([0.0, 0.1]),
    )[0]
    wave = np.geomspace(start=1.0e-5, stop=500.0, num=6001)
    rho = 2.77536627e11 * 0.3
    expected = np.empty_like(actual)
    for iz in range(redshift.size):
        power = covariance_model.power(
            k=wave, z=np.full_like(wave, redshift[iz]), mode=0
        )
        for im in range(inputs["mass"].size):
            radius = (3.0 * inputs["mass"][im] / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
            argument = wave * radius
            window = 3.0 * spherical_jn(1, argument) / argument
            expected[iz, im] = simpson(
                y=wave**3 * power * window**2 / (2.0 * np.pi**2),
                x=np.log(wave),
            )
    np.testing.assert_allclose(actual, expected, rtol=3.0e-5, atol=0.0)


def test_joint_terms_noise_rules_and_area(compiled, covariance_model):
    """Check physical partition rules with the actual projected model.

    BB, connected one-halo and abundance-response terms contain no Gaussian
    shape noise. P*T and P*B are linear in its amplitude. Every short-mode
    large-area term scales as 1/area; the finite-window response need not.
    """
    inputs = covariance_arguments()
    results = []
    for noise_amplitude in (0.0, 1.0, 2.0):
        changed = dict(inputs)
        changed["noise"] = noise_amplitude * inputs["noise"]
        results.append(
            compiled.production.physical_covariance(
                model=covariance_model, inputs=changed
            )
        )
    mean, covariance = results[1]
    np.testing.assert_array_equal(covariance, covariance.transpose(0, 2, 1))
    np.testing.assert_array_equal(results[0][0], results[2][0])
    np.testing.assert_array_equal(covariance[0, 0, 1:], np.zeros(shape=3))
    for term in (1, 4, 5):
        np.testing.assert_array_equal(results[0][1][term], results[2][1][term])
    for term in (2, 3):
        expected_middle = 0.5 * (results[0][1][term] + results[2][1][term])
        np.testing.assert_allclose(covariance[term], expected_middle, rtol=2.0e-13)
    for term in (4, 5):
        eigenvalues = np.linalg.eigvalsh(covariance[term])
        assert eigenvalues[0] >= -1.0e-12 * np.max(np.abs(eigenvalues))
    larger = dict(inputs)
    larger["area"] = 2.0 * inputs["area"]
    larger_mean, larger_covariance = compiled.production.physical_covariance(
        model=covariance_model, inputs=larger
    )
    np.testing.assert_array_equal(larger_mean, mean)
    np.testing.assert_allclose(larger_covariance[:5], covariance[:5] / 2.0, rtol=3.0e-14)


@pytest.mark.parametrize("adapter_name", ["production", "notebook"])
@pytest.mark.parametrize("large_area", [0, 1])
def test_zero_measure_padding_preserves_bits(
    compiled, covariance_model, adapter_name, large_area
):
    """Move the same radial nodes between SIMD lanes and the scalar tail.

    A radial integral contains 2*pi*r*dr. Prepending r=0 with positive dr
    therefore adds exactly zero measure while preserving every original
    summand and its order. Seven nodes become eight, so the last physical
    node moves from the scalar tail into a SIMD lane. Different grouping
    of w*(a*b) and (w*a)*b can then change the halo covariance's last bits.
    This test exercises the public calculation rather than a copied dot
    product, including finite-window transforms of the halo response.
    """
    inputs = covariance_arguments()
    inputs["large_area"] = large_area
    inputs["radius"], inputs["dr"] = logarithmic_rule(
        lower=1.0e-6, upper=0.03, count=7
    )
    padded = dict(inputs)
    padded["radius"] = np.concatenate(([0.0], inputs["radius"]))
    padded["dr"] = np.concatenate(([1.0], inputs["dr"]))
    adapter = getattr(compiled, adapter_name)
    original = adapter.physical_covariance(model=covariance_model, inputs=inputs)
    actual = adapter.physical_covariance(model=covariance_model, inputs=padded)
    assert np.any(original[1][4] != 0.0), "connected halo response must be reached"
    for reference, result in zip(original, actual):
        assert reference.shape == result.shape
        assert reference.tobytes() == result.tobytes()


@pytest.mark.parametrize("adapter_name", ["production", "notebook"])
def test_zero_power_padding_preserves_xi_bits(compiled, adapter_name):
    """An added zero-power node changes SIMD placement, not xi's value."""
    ell, dell = logarithmic_rule(lower=0.7, upper=1200.0, count=7)
    arguments = {
        "ell": ell,
        "dell": dell,
        "theta": np.asarray([[0.001, 0.004], [0.003, 0.007]]),
        "theta_weight": np.asarray([[0.3, 0.7], [0.4, 0.6]]),
        "spectra_e": np.asarray([1.3, 0.7])[:, None] / (1.0 + ell / 170.0),
        "spectra_b": np.asarray([0.2, 0.4])[:, None] / (1.0 + ell / 350.0),
        "sign": np.asarray([1, -1], dtype=np.int32),
    }
    padded = dict(arguments)
    padded["ell"] = np.concatenate((ell, [2.0 * ell[-1]]))
    padded["dell"] = np.concatenate((dell, [1.0]))
    for name in ("spectra_e", "spectra_b"):
        padded[name] = np.column_stack((arguments[name], np.zeros(shape=2)))
    adapter = getattr(compiled, adapter_name)
    original = adapter.xi_from_power(**arguments)
    actual = adapter.xi_from_power(**padded)
    assert np.all(original != 0.0)
    assert original.tobytes() == actual.tobytes()


@pytest.mark.parametrize("adapter_name", ["production", "notebook"])
def test_zero_bin_weight_preserves_endpoint_bits(
    compiled, covariance_model, adapter_name
):
    """A zero-weight separation moves the old final endpoint into a lane.

    Three separation nodes times three orientations give nine endpoints.
    The fourth separation has exactly zero weight, giving twelve nodes
    with unchanged earlier nodes and sums. The finite-window connected
    term uses these xi endpoint profiles, including their interpolation.
    """
    inputs = covariance_arguments()
    inputs["large_area"] = 0
    inputs["phi"] = (np.arange(3) + 0.5) * (2.0 * np.pi / 3.0)
    inputs["dphi"] = np.full(shape=3, fill_value=2.0 * np.pi / 3.0)
    inputs["theta"] = np.zeros(shape=(4, 3), dtype=np.float64)
    inputs["theta"][1:3] = [0.001, 0.002, 0.003]
    inputs["theta_weight"] = np.zeros_like(inputs["theta"])
    inputs["theta_weight"][1:3] = [0.1, 0.3, 0.6]
    padded = dict(inputs)
    padded["theta"] = np.column_stack((inputs["theta"], inputs["theta"][:, -1]))
    padded["theta_weight"] = np.column_stack(
        (inputs["theta_weight"], np.zeros(shape=4))
    )
    adapter = getattr(compiled, adapter_name)
    original = adapter.physical_covariance(model=covariance_model, inputs=inputs)
    actual = adapter.physical_covariance(model=covariance_model, inputs=padded)
    assert np.all(np.diag(original[1][4])[1:3] != 0.0)
    for reference, result in zip(original, actual):
        assert reference.shape == result.shape
        assert reference.tobytes() == result.tobytes()


def test_joint_adapter_and_thread_ownership(compiled, covariance_model):
    """Both adapters call one physical implementation and own their outputs."""
    inputs = covariance_arguments()
    baseline = compiled.production.physical_covariance(
        model=covariance_model, inputs=inputs
    )
    copied = dict(inputs)
    copied["theta"] = np.asfortranarray(inputs["theta"])
    notebook = compiled.notebook.physical_covariance(
        model=covariance_model, inputs=copied
    )
    np.testing.assert_array_equal(notebook[0], baseline[0])
    np.testing.assert_array_equal(notebook[1], baseline[1])
    try:
        for threads in (2, 1):
            compiled.set_omp_threads(nthreads=threads)
            actual = compiled.production.physical_covariance(
                model=covariance_model, inputs=inputs
            )
            np.testing.assert_array_equal(actual[0], baseline[0])
            np.testing.assert_array_equal(actual[1], baseline[1])
    finally:
        compiled.set_omp_threads(nthreads=1)


def test_pb_and_bb_fourier_normalization(compiled, covariance_model):
    """Compare non-Gaussian contractions with direct labeled Fourier sums.

    A single-source Map2/Map3 pair has six large-area PB placements and
    nine BB placements. The reference evaluates the public projected
    bispectrum at every triangle, keeping its angular normalization
    visible. It does not use the covariance's filtered-bispectrum tables.
    """
    inputs = covariance_arguments()
    inputs["row_kind"] = np.asarray([0, 1], dtype=np.int32)
    inputs["row_fields"] = np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
    inputs["field_source"] = np.zeros(shape=4, dtype=np.int32)
    inputs["theta"] = np.zeros(shape=(2, 2))
    inputs["theta_weight"] = np.zeros(shape=(2, 2))
    _, terms = compiled.production.physical_covariance(
        model=covariance_model, inputs=inputs
    )
    first, second, angle = np.meshgrid(
        inputs["ell"], inputs["ell"], inputs["phi"], indexing="ij"
    )
    closure = np.hypot(first + second * np.cos(angle), second * np.sin(angle))
    triangles = np.column_stack((first.ravel(), second.ravel(), closure.ravel()))
    bispectrum = covariance_model.projected(
        sides=triangles,
        bins=np.zeros(shape=3, dtype=np.int32),
        parameters={"a1": 0.0, "a2": 0.0, "ct": 0.0},
        component=1,
    ).reshape(first.shape)
    power = covariance_model.nla_power(
        ell=inputs["ell"],
        bins=np.zeros(shape=2, dtype=np.int32),
        parameters={"a1": 0.0},
    )
    power += inputs["noise"][0, 0]
    small = inputs["field_radius"][0]
    large = inputs["field_radius"][1]
    filter2 = 0.5 * (first * small) ** 2 * np.exp(-0.5 * (first * small) ** 2)
    filter31 = 0.5 * (first * large) ** 2 * np.exp(-0.5 * (first * large) ** 2)
    filter32 = 0.5 * (second * large) ** 2 * np.exp(-0.5 * (second * large) ** 2)
    filter33 = 0.5 * (closure * large) ** 2 * np.exp(-0.5 * (closure * large) ** 2)
    radial = inputs["ell"] * inputs["dell"]
    measure = radial[:, None, None] * radial[None, :, None]
    measure = measure * inputs["dphi"][None, None, :] / (2.0 * np.pi) ** 3
    integrand = filter2**2 * filter31 * filter32 * filter33
    integrand = integrand * power[:, None, None] * bispectrum
    expected_pb = 6.0 * np.sum(measure * integrand) / inputs["area"]
    np.testing.assert_allclose(terms[3, 0, 1], expected_pb, rtol=5.0e-12)

    inner_measure = radial[None, :, None] * inputs["dphi"][None, None, :]
    inner_measure = inner_measure / (2.0 * np.pi) ** 2
    inner = np.sum(inner_measure * filter32 * filter33 * bispectrum, axis=(1, 2))
    filter_outer = filter31[:, 0, 0]
    expected_bb = 9.0 * np.sum(radial * filter_outer**2 * inner**2)
    expected_bb /= 2.0 * np.pi * inputs["area"]
    np.testing.assert_allclose(terms[1, 1, 1], expected_bb, rtol=5.0e-12)


def test_connected_four_field_projection(compiled, covariance_model):
    """Check the physical one-halo volume and source-window normalization.

    The independent reference builds the Map2 halo response directly in
    Fourier space, then integrates its square against chi^2 dchi dn/dM dM.
    This distinguishes angular halo profiles from comoving density profiles.
    """
    inputs = covariance_arguments()
    _, terms = compiled.production.physical_covariance(
        model=covariance_model, inputs=inputs
    )
    radial = covariance_model.radial()
    redshift = radial["z"]
    distance = radial["chi"]
    wave = np.unique((inputs["ell"][None, :] / distance[:, None]).ravel())
    tables = compiled.production.halo_tables(
        model=covariance_model, inputs=inputs, z=redshift, k_profile=wave
    )
    abundance = tables[1]
    profile = tables[4]
    rho = 2.77536627e11 * 0.3
    fourier_weight = inputs["ell"] * inputs["dell"] / (2.0 * np.pi)
    filters = []
    for radius in inputs["field_radius"][:2]:
        x = radius * inputs["ell"]
        filters.append(0.5 * x**2 * np.exp(-0.5 * x**2))
    reference = 0.0
    for iz in range(redshift.size):
        indices = np.searchsorted(wave, inputs["ell"] / distance[iz])
        windows = distance[iz] * (1.0 + redshift[iz]) * radial["lens"][:, iz]
        for im in range(inputs["mass"].size):
            halo = inputs["mass"][im] * profile[iz, im, indices]
            halo /= rho * distance[iz] ** 2
            response = np.sum(fourier_weight * filters[0] * filters[1] * halo**2)
            response *= windows[0] * windows[1]
            measure = radial["dchi"][iz] * distance[iz] ** 2
            measure *= inputs["dmass"][im] * abundance[iz, im]
            reference += measure * response**2 / inputs["area"]
    np.testing.assert_allclose(terms[4, 3, 3], reference, rtol=2.0e-12)


@pytest.mark.parametrize("window", [1, 2])
def test_finite_center_windows(compiled, covariance_model, window):
    """Exercise circle and square windows with all joint physical families.

    The chosen small survey keeps its Fourier window broad on this test
    grid. This checks finite-window execution and the two Gram structures;
    it does not test the angular resolution needed for a wide DES footprint.
    """
    inputs = covariance_arguments()
    inputs["large_area"] = 0
    inputs["center_window"] = window
    inputs["area"] = 1.0e-5
    inputs["window_ell"], inputs["window_dell"] = logarithmic_rule(
        lower=0.02, upper=8000.0, count=48
    )
    mean, terms = compiled.production.physical_covariance(
        model=covariance_model, inputs=inputs
    )
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(terms))
    np.testing.assert_array_equal(terms, terms.transpose(0, 2, 1))
    for term in (4, 5):
        eigenvalues = np.linalg.eigvalsh(terms[term])
        assert eigenvalues[0] >= -1.0e-12 * np.max(np.abs(eigenvalues))


@pytest.mark.parametrize("large_area", [0, 1])
def test_xi_only_matches_joint_submatrix(compiled, covariance_model, large_area):
    """Removing cubic rows must retain every two-point covariance family.

    The pure-xi call skips bispectrum and aperture-partition preparation.
    Comparing with the same rows inside a mixed request checks this smaller
    preparation path, including finite-window Gaussian spin kernels and
    the connected and selected-response halo contributions.
    """
    inputs = covariance_arguments()
    inputs["large_area"] = large_area
    inputs["area"] = 1.0e-5
    joint_mean, joint_terms = compiled.production.physical_covariance(
        model=covariance_model, inputs=inputs
    )
    inputs["field_source"] = np.asarray([0, 1], dtype=np.int32)
    inputs["field_radius"] = np.zeros(shape=2)
    inputs["row_kind"] = np.asarray([2, 3], dtype=np.int32)
    inputs["row_fields"] = np.asarray([[0, 1, 0], [0, 1, 0]], dtype=np.int32)
    inputs["theta"] = inputs["theta"][1:3].copy()
    inputs["theta_weight"] = inputs["theta_weight"][1:3].copy()
    mean, terms = compiled.production.physical_covariance(
        model=covariance_model, inputs=inputs
    )
    np.testing.assert_allclose(mean, joint_mean[1:3], rtol=2.0e-14)
    np.testing.assert_allclose(terms, joint_terms[:, 1:3, 1:3], rtol=2.0e-14)


def test_finite_xi_geometry_sharing_against_spin_sum(compiled, covariance_model):
    """Sum all E/B Wick components with no grouped geometry in the reference.

    Six rows repeat the same xi bins across different source pairs. A
    seventh changes only its bin weights, so it must get another kernel.
    The reference constructs each row separately and sums its four spin
    components and two Wick contractions at every Fourier triangle.
    """
    inputs = covariance_arguments()
    inputs["large_area"] = 0
    inputs["area"] = 1.0e-5
    inputs["field_source"] = np.asarray([0, 1], dtype=np.int32)
    inputs["field_radius"] = np.zeros(shape=2)
    inputs["row_kind"] = np.asarray([2, 2, 2, 3, 3, 3, 2], dtype=np.int32)
    pairs = np.asarray([[0, 0], [0, 1], [1, 1], [0, 0], [0, 1], [1, 1], [0, 1]])
    inputs["row_fields"] = np.column_stack((pairs, np.zeros(shape=7))).astype(np.int32)
    inputs["theta"] = np.tile(np.asarray([0.002, 0.003]), reps=(7, 1))
    inputs["theta_weight"] = np.tile(np.asarray([0.25, 0.75]), reps=(7, 1))
    inputs["theta_weight"][-1] = [0.6, 0.4]
    inputs["noise"] = np.asarray([[2.0e-9, 0.4e-9], [0.4e-9, 3.0e-9]])
    _, terms = compiled.production.physical_covariance(
        model=covariance_model, inputs=inputs
    )
    ell = inputs["ell"]
    first = ell[:, None, None]
    second = ell[None, :, None]
    angle = inputs["phi"][None, None, :]
    dx = first - second * np.cos(angle)
    dy = -second * np.sin(angle)
    separation_wave = 0.5 * np.hypot(dx, dy)
    closure = np.hypot(first + second * np.cos(angle), second * np.sin(angle))
    window_argument = closure * np.sqrt(inputs["area"] / np.pi)
    survey_window = (2.0 * j1(window_argument) / window_argument) ** 2
    radial_measure = ell * inputs["dell"]
    measure = radial_measure[:, None, None] * radial_measure[None, :, None]
    measure = measure * inputs["dphi"][None, None, :] / (2.0 * np.pi) ** 3
    measure = measure * survey_window
    power = np.empty(shape=(2, 2, 2, ell.size))
    for a in range(2):
        for b in range(2):
            power[0, a, b] = covariance_model.nla_power(
                ell=ell,
                bins=np.asarray([a, b], dtype=np.int32),
                parameters={"a1": 0.0},
            )
            power[0, a, b] += inputs["noise"][a, b]
            power[1, a, b] = inputs["noise"][a, b]
    kernels = np.empty(shape=(7, 2, 2) + separation_wave.shape)
    for row in range(7):
        order = 0 if inputs["row_kind"][row] == 2 else 4
        bessel = np.sum(
            inputs["theta_weight"][row] * jv(
                order, separation_wave[..., None] * inputs["theta"][row]
            ),
            axis=-1,
        )
        phase = 2.0 * angle
        if order == 4:
            phase = phase - 4.0 * np.arctan2(dy, dx)
        kernels[row, 0, 0] = np.cos(phase) * bessel
        kernels[row, 1, 1] = kernels[row, 0, 0] * (1.0 if order == 0 else -1.0)
        kernels[row, 0, 1] = -np.sin(phase) * bessel
        kernels[row, 1, 0] = kernels[row, 0, 1] * (-1.0 if order == 0 else 1.0)
    reference = np.zeros(shape=(7, 7))
    for i, (a, b) in enumerate(pairs):
        for j, (c, d) in enumerate(pairs):
            for alpha in range(2):
                for beta in range(2):
                    contractions = (
                        power[alpha, a, c, :, None, None]
                        * power[beta, b, d, None, :, None]
                        + power[alpha, a, d, :, None, None]
                        * power[beta, b, c, None, :, None]
                    )
                    reference[i, j] += np.sum(
                        measure * kernels[i, alpha, beta]
                        * kernels[j, alpha, beta] * contractions
                    )
    np.testing.assert_allclose(terms[0], reference, rtol=3.0e-12)


def analytic_noise_arguments():
    """Return uniform-area xi annuli with overlap, disjoint bins and parity."""
    inputs = covariance_arguments()
    inputs["noise_mode"] = 1
    inputs["field_source"] = np.asarray([0, 1], dtype=np.int32)
    inputs["field_radius"] = np.zeros(shape=2)
    inputs["row_kind"] = np.asarray([2, 2, 2, 3, 2, 3, 2], dtype=np.int32)
    inputs["row_fields"] = np.asarray(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
         [0, 1, 0], [1, 0, 0], [1, 1, 0]],
        dtype=np.int32,
    )
    edges = np.asarray(
        [[0.001, 0.002], [0.0015, 0.0025], [0.003, 0.004],
         [0.001, 0.002], [0.001, 0.002], [0.0015, 0.0025], [0.001, 0.002]]
    )
    inputs["theta_edges"] = edges
    nodes, weights = leggauss(deg=5)
    half = 0.5 * (edges[:, 1] - edges[:, 0])
    center = 0.5 * (edges[:, 1] + edges[:, 0])
    inputs["theta"] = center[:, None] + half[:, None] * nodes
    inputs["theta_weight"] = weights[None, :] * inputs["theta"]
    inputs["theta_weight"] /= np.sum(inputs["theta_weight"], axis=1)[:, None]
    return inputs


def test_analytic_xi_noise_against_galaxy_pair_counts(compiled, covariance_model):
    """Two independent ellipticity components give variance per shared pair.

    Count expected ordered galaxy pairs in each annulus and in the overlap.
    Their normalized shared-pair variance is a configuration-space reference,
    independent of Hankel transforms. A second difference in noise amplitude
    isolates NN while cancelling the signal and mixed signal-noise terms.
    """
    inputs = analytic_noise_arguments()
    density = np.asarray([1.5e7, 2.2e7])
    component_variance = np.asarray([0.26**2, 0.29**2])
    noise = np.diag(component_variance / density)
    edges = inputs["theta_edges"]
    pairs = inputs["row_fields"][:, :2]
    count = len(pairs)
    expected = np.zeros(shape=(count, count))
    annuli = np.pi * (edges[:, 1] - edges[:, 0]) * (edges[:, 1] + edges[:, 0])
    for i, (a, b) in enumerate(pairs):
        for j, (c, d) in enumerate(pairs):
            direct = int(a == c and b == d)
            exchanged = int(a == d and b == c)
            lower = max(edges[i, 0], edges[j, 0])
            upper = min(edges[i, 1], edges[j, 1])
            if inputs["row_kind"][i] != inputs["row_kind"][j] or upper <= lower:
                continue
            overlap = np.pi * (upper - lower) * (upper + lower)
            shared_pairs = density[a] * density[b] * inputs["area"] * overlap
            count_i = density[a] * density[b] * inputs["area"] * annuli[i]
            count_j = density[c] * density[d] * inputs["area"] * annuli[j]
            pair_variance = 2.0 * component_variance[a] * component_variance[b]
            expected[i, j] = (direct + exchanged) * pair_variance
            expected[i, j] *= shared_pairs / (count_i * count_j)
    power = np.empty(shape=(2, 2, inputs["ell"].size))
    for a in range(2):
        for b in range(2):
            power[a, b] = covariance_model.nla_power(
                ell=inputs["ell"],
                bins=np.asarray([a, b], dtype=np.int32),
                parameters={"a1": 0.0},
            )
    bessel = []
    for row in range(count):
        order = 0 if inputs["row_kind"][row] == 2 else 4
        bessel.append(np.sum(
            inputs["theta_weight"][row, :, None]
            * jv(order, inputs["theta"][row, :, None] * inputs["ell"]),
            axis=0,
        ))
    fourier_measure = inputs["ell"] * inputs["dell"] / (2.0 * np.pi * inputs["area"])
    signal = np.zeros_like(expected)
    mixed = np.zeros_like(expected)
    for i, (a, b) in enumerate(pairs):
        for j, (c, d) in enumerate(pairs):
            weight = fourier_measure * bessel[i] * bessel[j]
            signal[i, j] = np.sum(
                weight * (power[a, c] * power[b, d] + power[a, d] * power[b, c])
            )
            mixed[i, j] = np.sum(weight * (
                power[a, c] * noise[b, d] + noise[a, c] * power[b, d]
                + power[a, d] * noise[b, c] + noise[a, d] * power[b, c]
            ))
    for adapter in (compiled.production, compiled.notebook):
        gaussian = []
        all_terms = []
        for scale in (0.0, 1.0, 2.0):
            inputs["noise"] = scale * noise
            _, terms = adapter.physical_covariance(model=covariance_model, inputs=inputs)
            gaussian.append(terms[0])
            all_terms.append(terms)
            np.testing.assert_allclose(
                terms[0], signal + scale * mixed + scale**2 * expected,
                rtol=2.0e-12, atol=np.max(expected) * 2.0e-14,
            )
        actual = 0.5 * (gaussian[2] - 2.0 * gaussian[1] + gaussian[0])
        np.testing.assert_allclose(
            actual, expected, rtol=2.0e-12, atol=np.max(expected) * 2.0e-14
        )
        # Replacing NN does not alter connected or selected-response terms.
        np.testing.assert_array_equal(all_terms[0][4:], all_terms[2][4:])
        # The independent pair count includes the auto-correlation exchange.
        expected_auto = 4.0 * noise[0, 0] ** 2 / (inputs["area"] * annuli[0])
        np.testing.assert_allclose(actual[0, 0], expected_auto, rtol=2.0e-12)


@pytest.mark.parametrize("adapter_name", ["production", "notebook"])
@pytest.mark.parametrize("invalid", ["finite_window", "absent", "reversed", "outside"])
def test_analytic_xi_noise_refuses_incompatible_geometry(
    adapter_name, invalid
):
    """Isolate the core's fatal refusal and require its intended diagnostic."""
    result = subprocess.run(
        args=[sys.executable, "-B", str(Path(__file__)), adapter_name, invalid],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    message = result.stdout + result.stderr
    diagnostics = {
        "finite_window": (
            "analytic xi noise requires large_area=1 and theta_edges[nrow,2]"
        ),
        "absent": "analytic xi noise requires large_area=1 and theta_edges[nrow,2]",
        "reversed": "analytic xi noise needs positive ordered bin edges",
        "outside": "xi node lies outside its declared analytic-noise bin",
    }
    assert result.returncode == 1
    assert diagnostics[invalid] in message
    for unexpected in (
        "Traceback", "runtime error:", "AddressSanitizer",
        "UndefinedBehaviorSanitizer", "Library not loaded", "symbol not found",
        "Invalid analytic noise inputs returned",
    ):
        assert unexpected not in message


def invalid_analytic_noise_worker(adapter_name, invalid):
    """Construct valid physics before submitting the selected invalid request."""
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project / "interface"))
    import cosmolike_des_3pcf_interface as compiled

    compiled.set_omp_threads(nthreads=1)
    model = make_covariance_model(compiled=compiled)
    inputs = analytic_noise_arguments()
    if invalid == "finite_window":
        inputs["large_area"] = 0
    elif invalid == "absent":
        inputs.pop("theta_edges")
    elif invalid == "reversed":
        inputs["theta_edges"][0] = [0.002, 0.001]
    else:
        inputs["theta_edges"][0] = [0.0011, 0.0019]
    getattr(compiled, adapter_name).physical_covariance(model=model, inputs=inputs)
    print("Invalid analytic noise inputs returned")


@pytest.mark.parametrize("large_area", [0, 1])
def test_halo_radius_sharing_against_labeled_fields(
    compiled, covariance_model, large_area
):
    """An unfactored calculation retains every source/radius field.

    Two source bins share each aperture radius; two cubic rows share their
    radius geometry while keeping different source products. This reference
    transforms each labeled field and each labeled pair independently. It
    checks both the six-field connected halo term and all P*T placements,
    including the six internal-pair placements for a finite center window.
    The comparison concerns equality of fixed quadrature, not DES accuracy.
    """
    inputs = covariance_arguments()
    inputs["large_area"] = large_area
    inputs["area"] = 1.0e-5
    inputs["field_source"] = np.asarray([0, 1, 0, 1], dtype=np.int32)
    inputs["field_radius"] = np.asarray([0.002, 0.002, 0.003, 0.003])
    rows = np.asarray([[0, 1, 2], [1, 0, 3], [0, 2, 3]], dtype=np.int32)
    inputs["row_fields"] = rows
    inputs["row_kind"] = np.ones(shape=3, dtype=np.int32)
    inputs["theta"] = np.zeros(shape=(3, 1))
    inputs["theta_weight"] = np.zeros(shape=(3, 1))
    _, terms = compiled.production.physical_covariance(
        model=covariance_model, inputs=inputs
    )
    radial = covariance_model.radial()
    distance = radial["chi"]
    redshift = radial["z"]
    all_ell = np.concatenate((inputs["ell"], inputs["window_ell"]))
    wave = np.unique((all_ell[None, :] / distance[:, None]).ravel())
    tables = compiled.production.halo_tables(
        model=covariance_model, inputs=inputs, z=redshift, k_profile=wave
    )
    sources = inputs["field_source"]
    ell = inputs["ell"]
    window_ell = inputs["window_ell"]
    radius = inputs["radius"]
    fourier_measure = ell * inputs["dell"] / (2.0 * np.pi)
    radial_measure = 2.0 * np.pi * radius * inputs["dr"]
    argument = inputs["field_radius"][:, None] * ell[None, :]
    filters = 0.5 * argument**2 * np.exp(-0.5 * argument**2)
    argument_window = inputs["field_radius"][:, None] * window_ell[None, :]
    window_filters = 0.5 * argument_window**2 * np.exp(-0.5 * argument_window**2)
    transform = j0(radius[:, None] * ell[None, :])
    window_transform = j0(radius[:, None] * window_ell[None, :])
    survey_radius = np.sqrt(inputs["area"] / np.pi)
    window_argument = window_ell * survey_radius
    window_measure = window_ell * inputs["window_dell"] / (2.0 * np.pi)
    window_measure *= (2.0 * j1(window_argument) / window_argument) ** 2
    overlap_argument = radius / (2.0 * survey_radius)
    overlap = np.zeros_like(radius)
    inside = overlap_argument < 1.0
    x = overlap_argument[inside]
    overlap[inside] = 2.0 * (np.arccos(x) - x * np.sqrt(1.0 - x**2))
    overlap /= np.pi * inputs["area"]

    power = np.empty(shape=(4, 4, ell.size))
    pair_zero = np.empty(shape=(4, 4))
    pair_window = np.empty_like(power)
    for a in range(4):
        for b in range(4):
            power[a, b] = covariance_model.nla_power(
                ell=ell,
                bins=np.asarray([sources[a], sources[b]], dtype=np.int32),
                parameters={"a1": 0.0},
            )
            power[a, b] += inputs["noise"][sources[a], sources[b]]
            filtered_power = filters[a] * filters[b] * power[a, b]
            pair_zero[a, b] = np.sum(fourier_measure * filtered_power)
            if large_area:
                pair_window[a, b] = fourier_measure * filtered_power / inputs["area"]
            else:
                correlation = transform @ (fourier_measure * filtered_power)
                pair_window[a, b] = fourier_measure * (
                    (radial_measure * overlap * correlation) @ transform
                )
    connected = np.zeros(shape=(3, 3))
    pt = np.zeros_like(connected)
    rho = 2.77536627e11 * 0.3
    for iz in range(redshift.size):
        indices = np.searchsorted(wave, ell / distance[iz])
        window_indices = np.searchsorted(wave, window_ell / distance[iz])
        source_window = distance[iz] * (1.0 + redshift[iz]) * radial["lens"][:, iz]
        for im in range(inputs["mass"].size):
            amplitude = inputs["mass"][im] / (rho * distance[iz] ** 2)
            halo = amplitude * tables[4][iz, im, indices]
            halo_window = amplitude * tables[4][iz, im, window_indices]
            field = np.empty(shape=(4, radius.size))
            field_window = np.empty(shape=(4, window_ell.size))
            for a in range(4):
                field[a] = transform @ (
                    fourier_measure * source_window[sources[a]] * halo * filters[a]
                )
                field_window[a] = (
                    source_window[sources[a]] * halo_window * window_filters[a]
                )
            pair_transform = np.empty(shape=(4, 4, ell.size))
            for a in range(4):
                for b in range(4):
                    pair_transform[a, b] = (
                        radial_measure * field[a] * field[b]
                    ) @ transform
            response = np.asarray([np.prod(field[row], axis=0) for row in rows])
            response_zero = response @ radial_measure
            response_window = (response * radial_measure) @ window_transform
            measure = radial["dchi"][iz] * distance[iz] ** 2
            measure *= inputs["dmass"][im] * tables[1][iz, im]
            if large_area:
                connected += (
                    measure * np.outer(response_zero, response_zero) / inputs["area"]
                )
            else:
                connected += (
                    measure * (response_window * window_measure) @ response_window.T
                )
            for i, a in enumerate(rows):
                for j, b in enumerate(rows):
                    value = 0.0
                    for p in range(3):
                        remaining_a = np.delete(a, p)
                        for q in range(3):
                            remaining_b = np.delete(b, q)
                            value += np.sum(
                                pair_window[a[p], b[q]]
                                * pair_transform[tuple(remaining_a)]
                                * pair_transform[tuple(remaining_b)]
                            )
                        if not large_area:
                            remaining_b = np.delete(b, p)
                            value += pair_zero[tuple(remaining_a)] * np.sum(
                                window_measure * field_window[a[p]] * response_window[j]
                            )
                            value += pair_zero[tuple(remaining_b)] * np.sum(
                                window_measure * field_window[b[p]] * response_window[i]
                            )
                    pt[i, j] += measure * value
    np.testing.assert_allclose(terms[4], connected, rtol=3.0e-12)
    np.testing.assert_allclose(terms[2], pt, rtol=3.0e-12)


def test_xi_profile_preparation_against_direct_bessel(compiled, covariance_model):
    """Resolve the xi halo response without interpolating its shear profile.

    Direct J2 evaluation at both displaced endpoints supplies an independent
    reference for the finite-window connected xi covariance. Three fixed
    preparation grids test both the coarse spline and uniform-table error.
    Other quadrature grids remain fixed, isolating this one approximation.
    """
    inputs = covariance_arguments()
    inputs["ell"], inputs["dell"] = logarithmic_rule(
        lower=10.0, upper=1500.0, count=9
    )
    inputs["large_area"] = 0
    inputs["area"] = 1.0e-5
    inputs["field_source"] = np.asarray([0, 1], dtype=np.int32)
    inputs["field_radius"] = np.zeros(shape=2)
    inputs["row_kind"] = np.asarray([2, 3], dtype=np.int32)
    inputs["row_fields"] = np.asarray([[0, 1, 0], [0, 1, 0]], dtype=np.int32)
    inputs["theta"] = np.asarray([[0.0015, 0.002, 0.003], [0.0015, 0.002, 0.003]])
    inputs["theta_weight"] = np.asarray([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]])
    # An odd node count exercises the SIMD tail; a shifted periodic rule
    # avoids assigning a node to an exactly collapsed Fourier triangle.
    inputs["phi"] = (np.arange(9) + 0.37) * (2.0 * np.pi / 9.0)
    inputs["dphi"] = np.full(shape=9, fill_value=2.0 * np.pi / 9.0)
    radial = covariance_model.radial()
    distance = radial["chi"]
    redshift = radial["z"]
    wave = np.unique((inputs["ell"][None, :] / distance[:, None]).ravel())
    tables = compiled.production.halo_tables(
        model=covariance_model, inputs=inputs, z=redshift, k_profile=wave
    )
    u = inputs["radius"][:, None, None]
    half = 0.5 * inputs["theta"][0][None, :, None]
    angle = inputs["phi"][None, None, :]
    dx = half * np.cos(angle)
    dy = half * np.sin(angle)
    minus_radius = np.hypot(u - dx, dy)
    plus_radius = np.hypot(u + dx, dy)
    minus_phase = np.arctan2(-dy, u - dx)
    plus_phase = np.arctan2(dy, u + dx)
    geometry_weights = inputs["theta_weight"][0][None, :, None]
    geometry_weights = geometry_weights * inputs["dphi"][None, None, :] / (2.0 * np.pi)
    phases = [
        2.0 * (minus_phase - plus_phase),
        2.0 * (minus_phase + plus_phase) - 4.0 * angle,
    ]
    minus_bessel = jv(2, minus_radius[..., None] * inputs["ell"])
    plus_bessel = jv(2, plus_radius[..., None] * inputs["ell"])
    fourier_measure = inputs["ell"] * inputs["dell"] / (2.0 * np.pi)
    radial_measure = 2.0 * np.pi * inputs["radius"] * inputs["dr"]
    hankel = j0(inputs["radius"][:, None] * inputs["window_ell"][None, :])
    x = inputs["window_ell"] * np.sqrt(inputs["area"] / np.pi)
    window_measure = inputs["window_ell"] * inputs["window_dell"] / (2.0 * np.pi)
    window_measure *= (2.0 * j1(x) / x) ** 2
    rho = 2.77536627e11 * 0.3
    reference = np.zeros(shape=(2, 2))
    for iz in range(redshift.size):
        indices = np.searchsorted(wave, inputs["ell"] / distance[iz])
        source_window = distance[iz] * (1.0 + redshift[iz]) * radial["lens"][:, iz]
        for im in range(inputs["mass"].size):
            halo = inputs["mass"][im] * tables[4][iz, im, indices]
            halo /= rho * distance[iz] ** 2
            minus = minus_bessel @ (fourier_measure * halo)
            plus = plus_bessel @ (fourier_measure * halo)
            response = np.asarray([
                np.sum(geometry_weights * np.cos(phase) * minus * plus, axis=(1, 2))
                for phase in phases
            ])
            response *= np.prod(source_window)
            transformed = (response * radial_measure) @ hankel
            measure = radial["dchi"][iz] * distance[iz] ** 2
            measure *= inputs["dmass"][im] * tables[1][iz, im]
            reference += measure * (transformed * window_measure) @ transformed.T
    errors = []
    for coarse, upsample in ((96, 4), (192, 8), (384, 16)):
        inputs["profile_radius"] = np.concatenate((
            [0.0], np.geomspace(start=1.0e-7, stop=0.04, num=coarse)
        ))
        inputs["profile_upsample"] = upsample
        _, terms = compiled.production.physical_covariance(
            model=covariance_model, inputs=inputs
        )
        errors.append(np.linalg.norm(terms[4] - reference) / np.linalg.norm(reference))
    print("xi halo interpolation relative errors:", errors)
    assert errors[-1] < errors[0]
    assert errors[-1] < 5.0e-4


if __name__ == "__main__":
    invalid_analytic_noise_worker(adapter_name=sys.argv[1], invalid=sys.argv[2])
