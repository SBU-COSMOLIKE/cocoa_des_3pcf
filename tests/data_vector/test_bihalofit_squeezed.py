"""Resolve floating-point conditioning in very squeezed nonzero-gap triangles.

For a short/long ratio u, individual tree terms contain factors of 1/u.
Their sum is finite, but rounding differences in nearly equal long-leg
powers are amplified. A universal relative tolerance near machine epsilon
therefore does not describe this test. Instead, reconstruct F2 independently
with 80 decimal digits from each route's actual power values and propagate
the measured power difference through that same physical equation.
"""

from test_bihalofit_python import retired_matter


from des3pcf.variance import reference_physics_model

from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from des3pcf.bihalofit import BiHalofit
from studies.bihalofit_refinement import pin_inputs
from test_bihalofit_python import comparison_inputs


def squeezed_queries():
    """Include soft, nonlinear and large k with resolved nonzero side gaps."""
    sides = []
    coordinates = []
    for middle in (1.0e-6, 0.1, 1.0e4):
        for ratio in (1.0e-10, 1.0e-11):
            for gap in (0.3, 0.7, 1.0):
                short = middle * ratio
                sides.append([short, middle, middle + short * gap])
                coordinates.append([middle, ratio, gap])
    return np.asarray(sides), np.asarray(coordinates)


def decimal_tree(sides, powers):
    """Return an independent F2 sum and the absolute sizes of its terms.

    Decimal.from_float preserves the actual supplied binary input values.
    High precision here belongs only to the test, not to the production
    numerical method. The ordinary law of cosines is safe at 80 digits for
    these finite triangles, so no production cancellation formula is copied.
    """
    with localcontext() as context:
        context.prec = 80
        lengths = []
        amplitudes = []
        for value in sides:
            lengths.append(Decimal.from_float(float(value)))
        for value in powers:
            amplitudes.append(Decimal.from_float(float(value)))
        total = Decimal(0)
        size = Decimal(0)
        kernels = []
        for first, second, opposite in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            a = lengths[first]
            b = lengths[second]
            c = lengths[opposite]
            cosine = (c*c - a*a - b*b) / (2*a*b)
            kernel = (Decimal(5) / 7 + cosine * (a/b + b/a) / 2
                      + Decimal(2) * cosine*cosine / 7)
            term = 2 * kernel * amplitudes[first] * amplitudes[second]
            total += term
            size += abs(term)
            kernels.append(kernel)
        return float(total), float(size), kernels


def propagated_power_error(kernels, actual, reference):
    """Bound |2 F2 (Pa Pb-Qa Qb)| without subtracting near-equal products."""
    with localcontext() as context:
        context.prec = 80
        first_power = []
        second_power = []
        for value in actual:
            first_power.append(Decimal.from_float(float(value)))
        for value in reference:
            second_power.append(Decimal.from_float(float(value)))
        bound = Decimal(0)
        for kernel, pair in zip(kernels, ((0, 1), (1, 2), (2, 0))):
            first, second = pair
            first_error = abs(first_power[first] - second_power[first])
            second_error = abs(first_power[second] - second_power[second])
            product_error = first_error * abs(second_power[second])
            product_error += second_error * abs(second_power[first])
            product_error += first_error * second_error
            bound += 2 * abs(kernel) * product_error
        return float(bound)


@pytest.mark.parametrize("source_readers", [False, True])
def test_nonzero_gap_squeezed_conditioning(compiled, tmp_path, source_readers):
    """Keep the tight ordinary gate; explain this conditioned stress case."""
    native, supplied = comparison_inputs(source_readers=source_readers)
    sides, coordinates = squeezed_queries()
    redshift = np.full(shape=sides.shape[0], fill_value=0.7)
    dictionaries = {
        "native": native,
        "supplied": supplied,
        "query": {"sides": sides, "z": redshift, "coordinates": coordinates},
    }
    path = tmp_path / "frozen_inputs.npz"
    loaded = pin_inputs(path=path, dictionaries=dictionaries, numpy=np)
    native = loaded["native"]
    supplied = loaded["supplied"]
    sides = loaded["query"]["sides"]
    redshift = loaded["query"]["z"]
    python = BiHalofit(**supplied, source_readers=source_readers)
    reference = reference_physics_model(factory=compiled.PhysicsModel, **native)
    actual_power = np.exp(python._linear.read(
        coordinates=np.log(sides), endpoint_power=True))
    growth = python._growth.read(coordinates=redshift)
    actual_power *= growth[:, None] * growth[:, None]
    native_power = reference.power(k=sides.ravel(),
                                   z=np.repeat(redshift, repeats=3), mode=0)
    native_power = native_power.reshape(sides.shape)
    report = {
        "inputs_npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "native_extension_sha256": hashlib.sha256(
            Path(compiled.__file__).read_bytes()).hexdigest(),
        "survey_data_covariance_mask": "not read by this equation test",
        "python_power": actual_power.tolist(),
        "native_power": native_power.tolist(),
        "relative_errors": {},
        "tree_roundoff_bounds": [],
    }
    for component in range(4):
        actual = python.evaluate(k1=sides[:, 0], k2=sides[:, 1],
                                 k3=sides[:, 2], z=redshift,
                                 component=component)
        expected = retired_matter(case=f"squeezed_{int(source_readers)}",
                                   component=component)
        assert np.all(np.isfinite(actual))
        assert np.all(np.isfinite(expected))
        if component == 0:
            assert np.all(actual > 0.0)
            assert np.all(expected > 0.0)
        if component == 2:
            np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=0.0)
        report["relative_errors"][str(component)] = (
            actual / expected - 1.0).tolist()
        if component != 1:
            continue
        for row in range(sides.shape[0]):
            high_actual, size_actual, kernels = decimal_tree(
                sides=sides[row], powers=actual_power[row])
            high_native, size_native, unused = decimal_tree(
                sides=sides[row], powers=native_power[row])
            # Sixty-four epsilon covers the short floating sequence of
            # products, ratios and additions for three terms. Its scale is
            # the unsubtracted terms, not their much smaller final sum.
            arithmetic = 64.0 * np.finfo(float).eps
            bound_actual = arithmetic * size_actual
            bound_native = arithmetic * size_native
            assert abs(actual[row] - high_actual) <= bound_actual
            assert abs(expected[row] - high_native) <= bound_native
            power_error = propagated_power_error(
                kernels=kernels, actual=actual_power[row],
                reference=native_power[row])
            bound = power_error + bound_actual + bound_native
            assert abs(actual[row] - expected[row]) <= bound
            report["tree_roundoff_bounds"].append(bound)
    (tmp_path / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
