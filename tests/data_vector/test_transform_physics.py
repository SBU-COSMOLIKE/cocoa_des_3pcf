"""Use an analytic scalar bispectrum to check spin phases and normalization.

Take B=(ell1*ell2*ell3)^2 exp[-a(ell1^2+ell2^2+ell3^2)]. Multiplication
by the three spin-2 shear phases cancels its three ell^2 denominators.
The Cartesian three-shear integral is therefore a polynomial Gaussian
integral. Its exact moments below do not use FFTLog, Legendre coupling or
the fastnc projection formula. This distinguishes the historical TATT
centroid phase from the conjugate phase introduced on upstream main.
"""

import numpy as np
from scipy.special import eval_legendre


def gaussian_multipoles(ell, lmax, scale, extra_first_leg_power=False):
    """Project the analytic B onto the documented minus-cosine Legendre basis.

    Args:
        ell = positive Fourier radii, [n].
        lmax = highest Legendre degree.
        scale = a in exp[-a sum(ell_i^2)], positive squared-angle units.
        extra_first_leg_power = multiply B by ell1^2 to break leg symmetry.

    Returns:
        multipoles = real b_L[L,ell1,ell2], C contiguous.

    The small reference uses 96-point Gauss-Legendre integration. It has
    no matter power spectrum, growth factor, source bins or fitted model.
    """
    cosine, weights = np.polynomial.legendre.leggauss(deg=96)
    first_squared = ell[:, None, None] ** 2
    second_squared = ell[None, :, None] ** 2
    third_squared = first_squared + second_squared
    third_squared = third_squared - (
        2 * ell[:, None, None] * ell[None, :, None] * cosine
    )
    bispectrum = first_squared * second_squared * third_squared
    bispectrum *= np.exp(
        -scale * (first_squared + second_squared + third_squared)
    )
    if extra_first_leg_power:
        bispectrum *= first_squared
    multipoles = np.empty((lmax + 1, ell.size, ell.size))
    for degree in range(lmax + 1):
        polynomial = eval_legendre(degree, cosine)
        multipoles[degree] = (2 * degree + 1) / 2 * np.sum(
            bispectrum * weights * polynomial, axis=2
        )
    return multipoles


def gaussian_moment(variables, means, scale):
    """Apply Wick's recursion to a polynomial in complex Gaussian variables.

    Each variable is (vertex, spin_sign), with sign +1 for kx+i ky and
    -1 for kx-i ky. Equal signs have zero covariance. Opposite signs
    have covariance 2/(3a) at the same vertex and -1/(3a) otherwise.
    The recursion chooses either a mean or one partner for its first
    variable, then evaluates the smaller remaining polynomial.
    """
    if len(variables) == 0:
        return np.ones_like(means[0])
    vertex, sign = variables[0]
    first_mean = means[vertex]
    if sign == -1:
        first_mean = -first_mean.conj()
    result = first_mean * gaussian_moment(
        variables=variables[1:], means=means, scale=scale
    )
    for partner in range(1, len(variables)):
        other_vertex, other_sign = variables[partner]
        if sign == other_sign:
            continue
        covariance = -1 / (3 * scale)
        if vertex == other_vertex:
            covariance = 2 / (3 * scale)
        remaining = variables[1:partner] + variables[partner + 1:]
        result += covariance * gaussian_moment(
            variables=remaining, means=means, scale=scale
        )
    return result


def gaussian_centroid_components(
    theta, phi, scale, extra_first_leg_power=False
):
    """Evaluate exact Gaussian moments in the counterclockwise triangle.

    Args:
        theta = two side lengths, [n], in a common angular unit.
        phi = counterclockwise opening angles, [nphi], radians.
        scale = the Gaussian squared-angle scale a.
        extra_first_leg_power = apply ell2^2 at physical vertex r2.

    Returns:
        gamma = complex centroid components [4,n,n,nphi].

    Set vertices r1=0, r2=theta1, r3=theta2 exp(i phi). Integrating over
    ell2 and ell3 leaves a Gaussian with covariance A^-1/2, where
    A=a[[2,1],[1,2]] for each Cartesian direction. The complex means
    below are i A^-1 r/2. Two plus-spin factors have zero covariance;
    a minus-spin and a plus-spin factor at different vertices have
    covariance -1/(3a). Wick's rule gives the six displayed monomials.
    Finally gamma_t=-gamma exp(-2i alpha) rotates each shear toward the
    centroid. Three minus signs account for the overall minus sign.
    """
    vertex2 = theta[:, None, None]
    vertex3 = theta[None, :, None] * np.exp(1j * phi)
    centroid_directions = [
        vertex2 + vertex3,
        -2 * vertex2 + vertex3,
        vertex2 - 2 * vertex3,
    ]
    rotations = []
    for direction in centroid_directions:
        rotations.append(direction / direction.conj())
    means = [
        -1j * (vertex2 + vertex3) / (6 * scale),
        1j * (2 * vertex2 - vertex3) / (6 * scale),
        1j * (-vertex2 + 2 * vertex3) / (6 * scale),
    ]
    quadratic = abs(vertex2) ** 2 + abs(vertex3) ** 2
    quadratic = quadratic - (vertex2 * vertex3.conj()).real
    normalization = np.exp(-quadratic / (6 * scale))
    normalization /= 48 * np.pi ** 2 * scale ** 2
    result = np.empty((4, theta.size, theta.size, phi.size), dtype=complex)
    result[0] = -normalization * means[0] ** 2
    result[0] *= means[1] ** 2 * means[2] ** 2
    result[0] /= rotations[0] * rotations[1] * rotations[2]
    covariance = -1 / (3 * scale)
    for vertex in range(3):
        minus_mean = -means[vertex].conj()
        first_plus = means[(vertex + 1) % 3]
        second_plus = means[(vertex + 2) % 3]
        moment = minus_mean ** 2 * first_plus ** 2 * second_plus ** 2
        moment += 4 * minus_mean * first_plus * second_plus ** 2 * covariance
        moment += 4 * minus_mean * second_plus * first_plus ** 2 * covariance
        moment += 2 * second_plus ** 2 * covariance ** 2
        moment += 8 * first_plus * second_plus * covariance ** 2
        moment += 2 * first_plus ** 2 * covariance ** 2
        result[vertex + 1] = -normalization * moment * rotations[vertex]
        result[vertex + 1] /= (
            rotations[(vertex + 1) % 3] * rotations[(vertex + 2) % 3]
        )
    if extra_first_leg_power:
        # The first radial transform variable is the wave vector attached
        # to r2. Multiplying B by its squared magnitude inserts one plus
        # and one minus complex momentum at vertex index 1. This leaves
        # a valid scalar bispectrum that is not invariant under leg swap.
        for component in range(4):
            variables = []
            phase = np.ones_like(normalization, dtype=complex)
            for vertex in range(3):
                sign = 1
                if component == vertex + 1:
                    sign = -1
                variables.extend([(vertex, sign), (vertex, sign)])
                phase *= rotations[vertex] ** (-sign)
            variables.extend([(1, 1), (1, -1)])
            moment = gaussian_moment(
                variables=variables, means=means, scale=scale
            )
            result[component] = -normalization * moment * phase
    return result


def test_analytic_spin_phases_and_domain_refinement(compiled):
    """All four components approach the independent Gaussian calculation.

    FFTLog is periodic in logarithmic coordinates. Increasing only its
    node count does not remove copies from a too-short Fourier interval.
    This check extends that interval; it measures numerical convergence,
    not execution speed. The allowed 0.2% error belongs only to this toy.
    """
    phi = np.array([0.6, 1.2, 2.2])
    errors = []
    for lower, upper in ((0.005, 15.0), (0.0001, 50.0)):
        ell = np.geomspace(start=lower, stop=upper, num=192)
        full_theta = 1 / ell[::-1]
        indices = np.array([
            np.argmin(abs(full_theta - 1.0)),
            np.argmin(abs(full_theta - 1.6)),
        ], dtype=np.int32)
        multipoles = gaussian_multipoles(ell=ell, lmax=32, scale=0.5)
        plan = compiled.production.TransformPlan(
            ell=ell, phi=phi, indices=indices, lmax=32, mmax=32,
            npsi=0, projection="centroid_tatt",
        )
        real, imag = plan.natural_components(b_l=multipoles)
        actual = real + 1j * imag
        expected = gaussian_centroid_components(
            theta=plan.theta, phi=phi, scale=0.5
        )
        component_errors = []
        for component in range(4):
            error = np.max(abs(actual[component] - expected[component]))
            error /= np.max(abs(expected[component]))
            component_errors.append(error)
        errors.append(np.asarray(component_errors))
    assert np.max(errors[1]) < 0.002
    assert np.all(errors[1] < errors[0] / 5)

    # A conjugate projection does not merely rename components: for this
    # test it introduces a large imaginary part into the real Gamma0.
    main_plan = compiled.production.TransformPlan(
        ell=ell, phi=phi, indices=indices, lmax=32, mmax=32,
        npsi=0, projection="centroid_main",
    )
    main_real, main_imag = main_plan.natural_components(b_l=multipoles)
    main_error = np.max(abs(main_real + 1j * main_imag - expected))
    assert main_error > 0.5 * np.max(abs(expected))


def test_ordered_legs_against_asymmetric_gaussian_moments(compiled):
    """Unequal physical legs require independent positive and negative M.

    Multiplying the scalar toy by one Fourier leg's ell^2 changes the
    physical three-field correlation. Its exact eight-variable Gaussian
    moment supplies a separate check of the general signed-M path. The
    old transpose shortcut must fail here; no source-bin averaging can
    be presumed by the numerical transform.
    """
    ell = np.geomspace(start=0.0001, stop=50.0, num=192)
    full_theta = 1 / ell[::-1]
    indices = np.array([
        np.argmin(abs(full_theta - 1.0)),
        np.argmin(abs(full_theta - 1.6)),
    ], dtype=np.int32)
    phi = np.array([0.6, 1.2, 2.2])
    multipoles = gaussian_multipoles(
        ell=ell, lmax=32, scale=0.5, extra_first_leg_power=True
    )
    plan = compiled.production.TransformPlan(
        ell=ell, phi=phi, indices=indices, lmax=32, mmax=32,
        npsi=0, projection="centroid_tatt", ordered_legs=True,
    )
    real, imag = plan.natural_components(b_l=multipoles)
    actual = real + 1j * imag
    expected = gaussian_centroid_components(
        theta=plan.theta, phi=phi, scale=0.5, extra_first_leg_power=True
    )
    for component in range(4):
        scale = np.max(abs(expected[component]))
        np.testing.assert_allclose(actual[component], expected[component],
                                   rtol=0, atol=0.003 * scale)
    historical_plan = compiled.production.TransformPlan(
        ell=ell, phi=phi, indices=indices, lmax=32, mmax=32,
        npsi=0, projection="centroid_tatt", ordered_legs=False,
    )
    old_real, old_imag = historical_plan.natural_components(b_l=multipoles)
    historical_error = np.max(abs(old_real + 1j * old_imag - expected))
    assert historical_error > 0.05 * np.max(abs(expected))
