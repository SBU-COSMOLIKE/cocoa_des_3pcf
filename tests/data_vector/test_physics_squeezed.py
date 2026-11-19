"""Check squeezed geometry against 80-digit, unrearranged equations.

The reference deliberately uses the ordinary squared-side cosine and
cyclic F2 sums with Decimal arithmetic. It does not reproduce the C
factorizations. Extra precision retains the small differences that double
arithmetic loses for a triangle with one side a billion times shorter.
All inputs to the reference are the exact binary doubles supplied to C.
"""

from des3pcf.variance import reference_physics_model

from decimal import Decimal, localcontext
import json
from pathlib import Path

import numpy as np
import pytest

from test_physics_model import physical_inputs


def decimal_triangle(sides, power):
    """Return direct F2 kernels and their weighted cyclic sum.

    The caller supplies a Decimal context with enough precision. Powers
    come from the model's reader so this comparison isolates the geometry
    and contraction, rather than the separate interpolation calculation.
    """
    lengths = []
    spectra = []
    for value in sides:
        lengths.append(Decimal.from_float(float(value)))
    for value in power:
        spectra.append(Decimal.from_float(float(value)))
    kernels = []
    terms = []
    for cyclic in range(3):
        first = lengths[cyclic]
        second = lengths[(cyclic + 1) % 3]
        closing = lengths[(cyclic + 2) % 3]
        cosine = (closing**2 - first**2 - second**2) / (2 * first * second)
        kernel = Decimal(5) / 7
        kernel += cosine * (first / second + second / first) / 2
        kernel += Decimal(2) * cosine**2 / 7
        kernels.append(kernel)
        terms.append(2 * kernel * spectra[cyclic] * spectra[(cyclic + 1) % 3])
    return lengths, spectra, kernels, terms


def decimal_alignment(sides, power, c1, c2, ct, bta, remove_alignment):
    """Evaluate the original helicity contractions with 80-digit geometry.

    A cyclic ordering puts its closing wavevector on the positive x axis.
    The two incoming unit vectors follow from the cosine law and vector
    closure. These are the unrearranged source equations; using them at
    high precision provides a separate check of the stable C evaluation.
    """
    with localcontext() as context:
        context.prec = 80
        lengths, spectra, kernels, tree = decimal_triangle(sides, power)
        alignment = Decimal.from_float(float(c1))
        quadratic = Decimal.from_float(float(c2))
        velocity = -2 * Decimal.from_float(float(ct)) / 5
        density = Decimal.from_float(float(bta)) * alignment
        density -= 2 * quadratic / 3
        linear = 2 * alignment
        root_two_thirds = (Decimal(2) / 3).sqrt()
        root_three_halves = (Decimal(3) / 2).sqrt()
        vertices = []
        for cyclic in range(3):
            first = lengths[cyclic]
            second = lengths[(cyclic + 1) % 3]
            closing = lengths[(cyclic + 2) % 3]
            cosine = (first**2 + closing**2 - second**2)
            cosine /= 2 * first * closing
            first_x = -cosine
            first_y = -(1 - cosine**2).sqrt()
            second_x = (-closing - first * first_x) / second
            second_y = -first * first_y / second
            dot = first_x * second_x + first_y * second_y
            pair = spectra[cyclic] * spectra[(cyclic + 1) % 3]
            shape0 = linear * kernels[cyclic]
            shape0 += quadratic * (
                (first_x**2 - first_y**2 + second_x**2 - second_y**2) / 4
                + dot * first_x * second_x / 2
            )
            shape0 += density * (
                2 * first_x**2 - first_y**2 + 2 * second_x**2 - second_y**2
            ) / 4
            shape0 += velocity * Decimal(5) * (1 - dot**2) / 7
            shape0 *= 2 * root_two_thirds * pair
            shape1 = -pair * (quadratic + density)
            shape1 *= first_x * first_y + second_x * second_y
            shape2 = quadratic * dot * first_y * second_y / 2
            shape2 += density * (first_y**2 + second_y**2) / 4
            shape2 *= 2 * pair
            vertices.append((shape0, shape1, shape2))

        def tensor002(cyclic, helicity):
            value = vertices[cyclic][helicity]
            if helicity == 0:
                value += root_two_thirds * linear * (
                    tree[(cyclic + 1) % 3] + tree[(cyclic + 2) % 3]
                )
            return value

        def tensor022(cyclic, second_helicity, third_helicity):
            second = Decimal(0)
            third = Decimal(0)
            if second_helicity == 0:
                second = root_two_thirds
            if third_helicity == 0:
                third = root_two_thirds
            value = second * linear * vertices[cyclic][third_helicity]
            value += second * third * linear**2 * tree[(cyclic + 1) % 3]
            value += third * linear * vertices[
                (cyclic + 2) % 3
            ][second_helicity]
            return value

        def tensor222(
            cyclic, first_helicity, second_helicity, third_helicity
        ):
            factors = []
            for helicity in (first_helicity, second_helicity, third_helicity):
                if helicity == 0:
                    factors.append(root_two_thirds)
                else:
                    factors.append(Decimal(0))
            value = factors[0] * factors[1] * vertices[cyclic][third_helicity]
            value += factors[1] * factors[2] * vertices[
                (cyclic + 1) % 3
            ][first_helicity]
            value += factors[2] * factors[0] * vertices[
                (cyclic + 2) % 3
            ][second_helicity]
            return linear**2 * value

        result = [Decimal(0)] * 19
        single_e = (0, 2, 1)
        double_e = (3, 5, 4)
        single_b = (7, 9, 8)
        mixed_eb = (10, 14, 12)
        mixed_be = (11, 15, 13)
        double_e_b = (16, 18, 17)
        for cyclic in range(3):
            result[single_e[cyclic]] = (
                root_three_halves * tensor002(cyclic, 0) - tensor002(cyclic, 2)
            ) / 2
            result[double_e[cyclic]] = root_three_halves * (
                root_three_halves * tensor022(cyclic, 0, 0)
                - tensor022(cyclic, 0, 2)
                - tensor022(cyclic, 2, 0)
            ) / 4
            result[single_b[cyclic]] = -tensor002(cyclic, 1)
            result[mixed_eb[cyclic]] = -(Decimal(3) / 8).sqrt()
            result[mixed_eb[cyclic]] *= tensor022(cyclic, 0, 1)
            result[mixed_be[cyclic]] = -(Decimal(3) / 8).sqrt()
            result[mixed_be[cyclic]] *= tensor022(cyclic, 1, 0)
            result[double_e_b[cyclic]] = -Decimal(3)
            result[double_e_b[cyclic]] *= tensor222(cyclic, 0, 0, 1) / 8
        # EEE contains each placement of helicity two exactly once.
        result[6] = Decimal(3) * (
            root_three_halves * tensor222(0, 0, 0, 0)
            - tensor222(0, 0, 0, 2)
            - tensor222(0, 2, 0, 0)
            - tensor222(0, 0, 2, 0)
        ) / 16
        if remove_alignment:
            total_tree = sum(tree)
            for index in range(3):
                result[index] -= alignment * total_tree
                result[index + 3] -= alignment**2 * total_tree
            result[6] -= alignment**3 * total_tree
        return np.asarray(result, dtype=np.float64)


@pytest.fixture(scope="module")
def squeezed_model(compiled):
    """Use the same smooth physical tables as the source comparison."""
    return reference_physics_model(factory=compiled.PhysicsModel, **physical_inputs())


def squeezed_triangles():
    """Include unequal long sides, cyclic label orders and common scales."""
    triangles = []
    for ratio in (1.0e-5, 1.0e-9, 1.0e-12):
        for opening in (0.0, 0.25, 0.875):
            base = np.asarray([ratio, 1.0, 1.0 + opening * ratio])
            for cyclic in range(3):
                for scale in (1.0 / 32.0, 1.0, 16.0):
                    triangles.append(np.roll(base, shift=cyclic) * scale)
    return np.asarray(triangles, dtype=np.float64)


def test_squeezed_tree_against_decimal(squeezed_model):
    """Cancel cyclic terms in high precision after reading the same powers."""
    sides = squeezed_triangles()
    redshifts = np.zeros(shape=sides.shape[0], dtype=np.float64)
    powers = squeezed_model.power(
        k=sides.reshape(-1).copy(),
        z=np.zeros(shape=sides.size, dtype=np.float64),
        mode=0,
    ).reshape(sides.shape)
    expected = []
    with localcontext() as context:
        context.prec = 80
        for triangle, power in zip(sides, powers):
            _, _, _, terms = decimal_triangle(sides=triangle, power=power)
            expected.append(float(sum(terms)))
    actual = squeezed_model.matter(
        sides=sides, redshifts=redshifts, component=1
    )
    np.testing.assert_allclose(actual, expected, rtol=3.0e-13, atol=0.0)


def test_squeezed_bihalofit_effective_power_sum(squeezed_model):
    """Use source fitted powers with high-precision kernels and cyclic sum.

    The 2e-4 tolerance is the same independently established allowance for
    the source's nonlinear-radius root, used in the ordinary matter test.
    It is not widened for the squeezed geometry. The saved ordinary source
    output also confirms that these cases reach its cancellation error.
    """
    reference_file = Path(__file__).with_name(
        "reference_squeezed_bihalofit.json"
    )
    record = json.loads(reference_file.read_text())
    sides = np.asarray(record["sides"], dtype=np.float64)
    redshift = np.asarray(record["redshifts"], dtype=np.float64)
    expected = np.asarray(record["high_precision_bh3"], dtype=np.float64)
    ordinary = np.asarray(record["ordinary_source_bh3"], dtype=np.float64)
    assert np.max(np.abs(ordinary / expected - 1.0)) > 0.1
    actual = squeezed_model.matter(
        sides=sides, redshifts=redshift, component=3
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-4, atol=0.0)


@pytest.mark.parametrize("remove_alignment", [False, True])
def test_squeezed_all_shape_components(squeezed_model, remove_alignment):
    """Check unit-vector geometry, all E/B outputs and linear subtraction."""
    sides = squeezed_triangles()
    redshifts = np.zeros(shape=sides.shape[0], dtype=np.float64)
    parameters = {
        "a1": 0.8,
        "a2": -1.1,
        "bta": 0.9,
        "ct": 0.25,
        "ct_mode": 1,
        "power_mode": 1,
    }
    powers = squeezed_model.power(
        k=sides.reshape(-1).copy(),
        z=np.zeros(shape=sides.size, dtype=np.float64),
        mode=1,
    ).reshape(sides.shape)
    c1 = -parameters["a1"] * 0.0134 * 0.3
    c2 = parameters["a2"] * 5.0 * 0.0134 * 0.3
    expected = []
    for triangle, power in zip(sides, powers):
        expected.append(decimal_alignment(
            sides=triangle,
            power=power,
            c1=c1,
            c2=c2,
            ct=parameters["ct"],
            bta=parameters["bta"],
            remove_alignment=remove_alignment,
        ))
    expected = np.asarray(expected, dtype=np.float64)
    actual = squeezed_model.ia(
        sides=sides,
        redshifts=redshifts,
        parameters=parameters,
        remove_alignment=remove_alignment,
    )
    # A zero component can have roundoff from subtracting angular terms.
    # Scale that floor to the reference amplitude of its own triangle;
    # every nonzero resolved component also has a relative error bound.
    scales = np.max(np.abs(expected), axis=1)[:, None]
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(
        actual / scales, expected / scales, rtol=2.0e-12, atol=2.0e-15
    )
