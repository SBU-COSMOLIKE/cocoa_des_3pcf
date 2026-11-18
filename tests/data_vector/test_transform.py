"""Check the angular and FFTLog transforms independently of cosmology.

The small grids here exercise finite numerical operators. The two-Bessel
reference uses NumPy FFTs and SciPy complex gamma functions, whereas C uses
FFTW and GSL. Angular coupling is also checked against direct integration
of its defining trigonometric integral. Agreement does not establish DES
grid convergence or a particular matter/IA bispectrum model.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_legendre, loggamma


def numpy_two_bessel(
    ell,
    integrand,
    order1,
    order2,
    nu1=1.01,
    nu2=1.01,
    xy=1.0,
    dlnt=0.0,
    padding=0,
    window=False,
    window_width=0.25,
):
    """Evaluate the documented finite FFTLog operator with NumPy/SciPy.

    Args:
        ell = positive logarithmic Fourier grid, [n].
        integrand = real dln(ell1)dln(ell2) integrand, [n,n].
        order1, order2 = nonnegative cylindrical Bessel orders.
        nu1, nu2 = real Mellin biases.
        xy = theta*reversed(ell), fixing the output-grid phase.
        dlnt = radial-bin logarithmic width; zero means point values.
        padding = zero cells added at each Fourier endpoint.
        window = enable the corrected Fourier coefficient taper.
        window_width = fraction of positive frequencies tapered.

    Returns:
        values = real transform [n,n] at theta=xy/ell[::-1].

    The historical fastnc branch constructs a taper and does not apply it.
    window=False deliberately retains that reference discretization.
    """
    size = ell.size + 2 * padding
    log_step = np.log(ell[1] / ell[0])
    extended_ell = ell[0] * np.exp(
        (np.arange(size) - padding) * log_step
    )
    extended = np.zeros((size, size))
    extended[padding:padding + ell.size, padding:padding + ell.size] = (
        integrand
    )
    biased = extended * extended_ell[:, None] ** (0.5 - nu1)
    biased = biased * extended_ell[None, :] ** (0.5 - nu2)
    coefficients = np.fft.rfft2(biased)
    row_modes = np.arange(size)
    row_modes[row_modes > size // 2] -= size
    column_modes = np.arange(size // 2 + 1)
    phase_origin = np.log(xy) - (size - 1) * log_step
    kernels = []
    for order, bias, modes in (
        (order1, nu1, row_modes),
        (order2, nu2, column_modes),
    ):
        frequency = 2 * np.pi * modes / (size * log_step)
        exponent = bias + 1j * frequency
        kernel = np.exp(
            exponent * np.log(2)
            + loggamma((order - 0.5 + exponent) / 2)
            - loggamma((order + 2.5 - exponent) / 2)
            - 1j * frequency * phase_origin
        )
        if dlnt > 0:
            slope = 2.5 - exponent
            kernel *= np.expm1(slope * dlnt) / slope
            kernel /= np.expm1(2 * dlnt) / 2
        if window:
            cutoff = int(window_width * size / 2)
            taper = np.ones(modes.size)
            if cutoff > 0:
                selected = np.abs(modes) > size // 2 - cutoff
                if cutoff == 1:
                    taper[selected] = 0
                else:
                    position = (size // 2 - np.abs(modes[selected]))
                    position = position / (cutoff - 1)
                    taper[selected] = position - np.sin(
                        2 * np.pi * position
                    ) / (2 * np.pi)
            kernel *= taper
        kernels.append(kernel)
    inverse = np.fft.irfft2(
        np.conj(coefficients * kernels[0][:, None] * kernels[1][None, :]),
        s=(size, size),
    )
    theta = xy / extended_ell[::-1]
    inverse *= theta[:, None] ** (0.5 - nu1)
    inverse *= theta[None, :] ** (0.5 - nu2)
    selected = slice(padding, padding + ell.size)
    return inverse[selected, selected] / 8


def angular_integrand(delta, psi, degree, mode):
    """Return 4 pi P_L(cos delta) cos(2 beta + M delta)."""
    cosine = np.cos(delta) + np.sin(2 * psi)
    sine = np.cos(2 * psi) * np.sin(delta)
    norm = np.hypot(cosine, sine)
    phase = (cosine * np.cos(mode * delta)
             - sine * np.sin(mode * delta)) / norm
    return 4 * np.pi * eval_legendre(degree, np.cos(delta)) * phase


def test_coupling_defining_integral(compiled):
    """Test both angular branches and the exact equal-leg limit."""
    psi = np.array([0.0, 0.21, np.pi / 4, 1.2, np.pi / 2])
    actual = compiled.production.legendre_coupling(
        psi=psi, lmax=5, mmax=6, npsi=0
    )
    reference = np.zeros_like(actual)
    for mode in range(7):
        for degree in range(6):
            for query, angle in enumerate(psi):
                reference[mode, degree, query] = quad(
                    func=angular_integrand,
                    a=0,
                    b=np.pi,
                    args=(angle, degree, mode),
                    epsabs=2.0e-11,
                    epsrel=2.0e-11,
                )[0]
    np.testing.assert_allclose(actual, reference, rtol=1.0e-11, atol=2.0e-11)
    copied = compiled.notebook.legendre_coupling(
        psi=psi[::-1], lmax=5, mmax=6, npsi=0
    )
    np.testing.assert_array_equal(copied, actual[:, :, ::-1])


@pytest.mark.parametrize("orders", [(0, 0), (1, 3), (5, 2)])
@pytest.mark.parametrize("padding,dlnt,window", [
    (0, 0.0, False),
    (4, 0.13, False),
    (2, 0.05, True),
])
def test_two_bessel_numpy_reference(compiled, orders, padding, dlnt, window):
    """Compare unequal orders/biases, averaging, padding and tapering."""
    ell = np.geomspace(start=0.05, stop=30.0, num=24)
    integrand = np.exp(-0.2 * ell[:, None] - 0.4 * ell[None, :])
    integrand *= ell[:, None] ** 1.2 * ell[None, :] ** 0.8
    arguments = {
        "ell": ell,
        "integrand": integrand,
        "order1": orders[0],
        "order2": orders[1],
        "nu1": 1.13,
        "nu2": 0.89,
        "xy": 1.7,
        "dlnt": dlnt,
        "padding": padding,
        "window": window,
    }
    reference = numpy_two_bessel(**arguments)
    actual = compiled.production.two_bessel(**arguments)
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(actual, reference, rtol=2.0e-11,
                               atol=2.0e-13 * scale)
    arguments["integrand"] = np.asfortranarray(integrand)
    notebook = compiled.notebook.two_bessel(**arguments)
    np.testing.assert_array_equal(actual, notebook)


@pytest.mark.parametrize("orders", [(63, 64), (97, 103)])
@pytest.mark.parametrize("biases", [(1.01, 1.01), (0.81, 1.73)])
def test_high_order_mellin_recurrence(compiled, orders, biases):
    """Check repeated Gamma recurrences against direct complex log-Gamma.

    The C preparation advances each parity from orders zero and one. The
    reference evaluates SciPy loggamma at the requested high order, so it
    does not share that recurrence. Unequal axes also exercise the second
    kernel's independent preparation. The signed input excites nonzero
    Fourier frequencies. Compare its full finite transform using one
    absolute scale, avoiding misleading relative errors at output zeros.
    """
    ell = np.geomspace(start=0.005, stop=500.0, num=64)
    logarithm = np.log(ell)
    first = np.exp(-0.3 * (logarithm - 1.0) ** 2)
    second = np.exp(-0.2 * (logarithm + 0.3) ** 2)
    first *= np.cos(3.0 * logarithm)
    second *= np.sin(2.0 * logarithm)
    arguments = {
        "ell": ell,
        "integrand": np.outer(first, second),
        "order1": orders[0],
        "order2": orders[1],
        "nu1": biases[0],
        "nu2": biases[1],
        "xy": 1.4,
        "dlnt": 0.04,
        "padding": 4,
        "window": True,
    }
    reference = numpy_two_bessel(**arguments)
    actual = compiled.production.two_bessel(**arguments)
    scale = np.max(np.abs(reference))
    assert scale > 0
    assert np.max(np.abs(actual - reference)) / scale < 2.0e-11


def finite_natural_reference(production, ell, phi, indices, multipoles, mmax):
    """Reproduce stored signed-M assembly using independent NumPy FFTs.

    Coupling has its separate defining-integral test above. Here the input
    is deliberately asymmetric between ell1 and ell2, so a wrong transpose
    or component exchange cannot be hidden by equal-leg symmetry.
    """
    lmax = multipoles.shape[0] - 1
    psi = np.arctan2(ell[None, :], ell[:, None])
    coupling = production.legendre_coupling(
        psi=np.ascontiguousarray(psi.ravel()),
        lmax=lmax,
        mmax=mmax,
        npsi=0,
    ).reshape(mmax + 1, lmax + 1, ell.size, ell.size)
    modes = np.zeros((4, indices.size, indices.size, 2 * mmax + 1))
    for mode in range(mmax + 1):
        legendre_sign = (-1.0) ** np.arange(lmax + 1)
        harmonic = np.sum(
            coupling[mode] * multipoles * legendre_sign[:, None, None],
            axis=0,
        )
        integrand = harmonic * ell[:, None] ** 2 * ell[None, :] ** 2
        orders = (
            (mode - 3, -mode - 3),
            (-mode - 1, mode - 1),
            (mode + 1, -mode - 3),
            (mode - 3, -mode + 1),
        )
        for component, (first_order, second_order) in enumerate(orders):
            values = numpy_two_bessel(
                ell=ell,
                integrand=integrand,
                order1=abs(first_order),
                order2=abs(second_order),
            )
            if first_order < 0:
                values *= (-1.0) ** first_order
            if second_order < 0:
                values *= (-1.0) ** second_order
            values = values[np.ix_(indices, indices)] / (2 * np.pi) ** 3
            if component == 0:
                modes[0, :, :, mode + mmax] = values
                modes[0, :, :, -mode + mmax] = values.T
            elif component == 1:
                modes[1, :, :, mode + mmax] = values.T
                modes[1, :, :, -mode + mmax] = values
            elif component == 2:
                modes[2, :, :, mode + mmax] = values
                modes[3, :, :, -mode + mmax] = values.T
            else:
                modes[3, :, :, mode + mmax] = values
                modes[2, :, :, -mode + mmax] = values.T
    phase = np.exp(1j * np.arange(-mmax, mmax + 1)[:, None] * phi)
    return np.einsum("cijm,mf->cijf", modes, phase) / (2 * np.pi)


def small_transform_inputs():
    """Return a finite asymmetric multipole table and small output grid."""
    ell = np.geomspace(start=0.03, stop=30.0, num=24)
    phi = np.array([0.4, 1.1, 2.5])
    indices = np.array([7, 11, 15], dtype=np.int32)
    multipoles = np.zeros((4, ell.size, ell.size))
    for degree in range(4):
        multipoles[degree] = (-0.2) ** degree * np.exp(
            -0.4 * ell[:, None] - 0.2 * ell[None, :]
        )
    return ell, phi, indices, multipoles


def test_all_natural_components_and_reuse(compiled):
    """Test signs, M=0 overwrites, negative M and unequal-side transposes."""
    ell, phi, indices, multipoles = small_transform_inputs()
    reference = finite_natural_reference(
        production=compiled.production,
        ell=ell,
        phi=phi,
        indices=indices,
        multipoles=multipoles,
        mmax=5,
    )
    plan = compiled.production.TransformPlan(
        ell=ell, phi=phi, indices=indices, lmax=3, mmax=5,
        npsi=0, projection="X", ordered_legs=False,
    )
    real, imag = plan.natural_components(b_l=multipoles)
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(real + 1j * imag, reference, rtol=3.0e-11,
                               atol=3.0e-13 * scale)
    np.testing.assert_array_equal(plan.theta, (1 / ell[::-1])[indices])
    original = real.copy()
    plan.natural_components(b_l=2 * multipoles)
    again = plan.natural_components(b_l=multipoles)
    np.testing.assert_array_equal(again[0], original)
    notebook = compiled.notebook.natural_components(
        plan=plan, b_l=np.asfortranarray(multipoles)
    )
    np.testing.assert_array_equal(notebook[0], real)
    np.testing.assert_array_equal(notebook[1], imag)


def test_projection_conversion_and_aperture(compiled):
    """Test both explicit centroid phases and the complete aperture route."""
    ell, phi, indices, multipoles = small_transform_inputs()
    results = {}
    for projection in ("X", "centroid_tatt", "centroid_main"):
        plan = compiled.production.TransformPlan(
            ell=ell, phi=phi, indices=indices, lmax=3, mmax=5,
            npsi=0, projection=projection,
        )
        real, imag = plan.natural_components(b_l=multipoles)
        results[projection] = real + 1j * imag
    theta = 1 / ell[::-1][indices]
    first = theta[:, None, None]
    second = theta[None, :, None] * np.exp(-1j * phi)
    directions = [first + second, -2 * first + second, first - 2 * second]
    phase1, phase2, phase3 = directions
    phase1 = phase1 / phase1.conj()
    phase2 = phase2 / phase2.conj()
    phase3 = phase3 / phase3.conj()
    phases = np.array([
        phase1 * phase2 * phase3 * np.exp(3j * phi),
        phase1.conj() * phase2 * phase3 * np.exp(1j * phi),
        phase1 * phase2.conj() * phase3 * np.exp(3j * phi),
        phase1 * phase2 * phase3.conj() * np.exp(-1j * phi),
    ])
    np.testing.assert_allclose(results["centroid_tatt"],
                               results["X"] * phases, rtol=2.0e-12)
    np.testing.assert_allclose(results["centroid_main"],
                               results["X"] * phases.conj(), rtol=2.0e-12)
    weights = compiled.production.aperture_weights(
        d2=theta, d3=theta, phi=phi, dlnd=0.1, dphi=0.3,
        filters=np.array([[0.6, 0.7, 1.1], [0.8, 0.8, 0.8]]),
    )
    # plan retains the last centroid_main geometry from the loop above.
    actual = plan.aperture_from_multipoles(
        b_l=multipoles, weights_real=weights[0], weights_imag=weights[1]
    )
    converted = -results["centroid_main"][[0, 1, 3, 2]]
    expected = compiled.production.aperture_apply(
        wr=weights[0], wi=weights[1],
        gr=np.ascontiguousarray(converted.real.reshape(1, 4, -1)),
        gi=np.ascontiguousarray(converted.imag.reshape(1, 4, -1)),
    )
    np.testing.assert_array_equal(actual, expected[0])


def test_parallel_plan_matches_serial(compiled):
    """Independent M blocks retain each summation order across workers."""
    ell, phi, indices, multipoles = small_transform_inputs()
    results = []
    for workers in (1, 2):
        compiled.set_omp_threads(nthreads=workers)
        plan = compiled.production.TransformPlan(
            ell=ell, phi=phi, indices=indices, lmax=3, mmax=5,
            npsi=0, projection="X",
        )
        results.append(plan.natural_components(b_l=multipoles))
    compiled.set_omp_threads(nthreads=1)
    for part in range(2):
        np.testing.assert_array_equal(results[0][part], results[1][part])


def test_opening_angle_average_is_sinc(compiled):
    """Average the finite X-mode series by independent angular quadrature."""
    ell, phi, indices, multipoles = small_transform_inputs()
    width = 0.24
    nodes, weights = np.polynomial.legendre.leggauss(deg=16)
    angles = (phi[:, None] + width * nodes[None, :] / 2).ravel()
    point_plan = compiled.production.TransformPlan(
        ell=ell, phi=angles, indices=indices, lmax=3, mmax=5,
        npsi=0, projection="X",
    )
    point_real, point_imag = point_plan.natural_components(b_l=multipoles)
    samples = (point_real + 1j * point_imag).reshape(4, 3, 3, 3, 16)
    expected = np.sum(samples * weights, axis=-1) / 2
    average_plan = compiled.production.TransformPlan(
        ell=ell, phi=phi, indices=indices, lmax=3, mmax=5,
        npsi=0, projection="X", dphi=width,
    )
    real, imag = average_plan.natural_components(b_l=multipoles)
    np.testing.assert_allclose(real + 1j * imag, expected,
                               rtol=3.0e-13, atol=1.0e-15)


def test_geometry_is_owned_and_symmetric_shortcut_agrees(compiled):
    """Copies isolate plans; the transpose identity needs symmetric B.

    The corrected taper removes both Nyquist boundaries. An untapered
    real FFT stores those complex boundary coefficients differently on
    its two axes; its finite operator is not exactly transpose invariant.
    Do not mistake that discretization difference for a physics identity.
    """
    ell, phi, indices, multipoles = small_transform_inputs()
    symmetric = np.ascontiguousarray(
        0.5 * (multipoles + multipoles.transpose(0, 2, 1))
    )
    common = {
        "ell": ell,
        "phi": phi,
        "indices": indices,
        "lmax": 3,
        "mmax": 5,
        "npsi": 0,
        "projection": "centroid",
        "window": True,
    }
    general = compiled.production.TransformPlan(**common)
    historical = compiled.production.TransformPlan(
        **common, ordered_legs=False
    )
    expected = historical.natural_components(b_l=symmetric)
    actual = general.natural_components(b_l=symmetric)
    for part in range(2):
        np.testing.assert_allclose(actual[part], expected[part],
                                   rtol=1.0e-11, atol=1.0e-14)
    # Mutating all caller geometry and returned metadata cannot reach the
    # C-owned tables or the immutable radial-index selection.
    ell *= 1.5
    phi += 0.1
    indices += 1
    returned_theta = general.theta
    returned_theta[:] = -1
    repeated = general.natural_components(b_l=symmetric)
    for part in range(2):
        np.testing.assert_array_equal(actual[part], repeated[part])
    assert np.all(general.theta > 0)


def test_smooth_lengths_and_explicit_coupling_interpolation(compiled):
    """Grid choices are explicit; interpolation uses the requested table."""
    for minimum, expected in ((2, 2), (7, 8), (17, 18), (28, 30),
                              (101, 108), (10201, 10240)):
        assert compiled.production.next_fft_size(minimum=minimum) == expected
    queries = np.array([0.013, 0.62, np.pi / 4, 1.32, 1.55])
    knots = np.linspace(start=0, stop=np.pi / 2, num=200)
    exact = compiled.production.legendre_coupling(
        psi=knots, lmax=5, mmax=7, npsi=0
    )
    actual = compiled.production.legendre_coupling(
        psi=queries, lmax=5, mmax=7, npsi=200
    )
    expected = np.empty_like(actual)
    for mode in range(8):
        for degree in range(6):
            expected[mode, degree] = np.interp(
                x=queries, xp=knots, fp=exact[mode, degree]
            )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-14)


def test_production_rejects_hidden_conversion(compiled):
    """Borrowing inputs must not hide dtype or memory-order copies."""
    ell, phi, indices, multipoles = small_transform_inputs()
    with pytest.raises(TypeError):
        compiled.production.TransformPlan(
            ell=ell.astype(np.float32), phi=phi, indices=indices,
            lmax=3, mmax=5,
        )
    plan = compiled.production.TransformPlan(
        ell=ell, phi=phi, indices=indices, lmax=3, mmax=5,
    )
    with pytest.raises(TypeError):
        plan.natural_components(b_l=np.asfortranarray(multipoles))
