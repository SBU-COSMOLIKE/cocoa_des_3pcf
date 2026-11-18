"""Check the field choices shared by one-halo P*T covariance terms.

P denotes a two-field correlation, including the supplied shape noise.
T denotes the connected four-field contribution of one halo. Each pair
of cubic estimators has nine choices of the two fields assigned to P.
Repeated choices can share their radial contraction, while the remaining
source weights and positional sum must retain their original order.

These tests compare public requests with different lists of observables.
They do not inspect or reproduce the contraction-cache implementation.
The independent labeled-field P*T integral remains in
test_physical_covariance.py. All grids here are small fixed rules used
to test reuse and ownership, not survey accuracy or execution time.
"""

import numpy as np
import pytest

from test_setting_sequences import base_inputs, model_inputs


@pytest.fixture(scope="module")
def model(compiled):
    """Construct the two-source physical model used by the state checks."""
    return compiled.PhysicsModel(**model_inputs(version="a"))


def mixed_inputs(large_area):
    """Return mixed estimators with repeated and differently ordered fields.

    Arguments:
        large_area = 0 retains the finite circular center window; 1 uses
            the declared large-area limit for short-mode terms.

    Returns:
        dict = eight estimator rows, six fields and the small fixed grids
            of the covariance state tests. Aperture radii are in radians.

    Rows 1 and 6 are identical. Row 3 exchanges the first two fields of
    row 1. Rows 5 and 7 introduce other complementary radius pairs and a
    repeated field, so one single-field label cannot identify a P*T term.
    """
    inputs = base_inputs(size="base")
    xi_theta = inputs["theta"][1].copy()
    xi_weights = inputs["theta_weight"][1].copy()
    xi_edges = inputs["theta_edges"][1].copy()
    inputs["large_area"] = large_area
    inputs["field_source"] = np.asarray(
        [
            0,
            1,
            0,
            1,
            0,
            1,
        ],
        dtype=np.int32,
    )
    inputs["field_radius"] = np.asarray(
        [
            0.002,
            0.003,
            0.0,
            0.0,
            0.003,
            0.002,
        ]
    )
    inputs["row_kind"] = np.asarray(
        [
            2,
            1,
            0,
            1,
            3,
            1,
            1,
            1,
        ],
        dtype=np.int32,
    )
    inputs["row_fields"] = np.asarray(
        [
            [
                2,
                3,
                0,
            ],
            [
                0,
                1,
                4,
            ],
            [
                0,
                5,
                0,
            ],
            [
                1,
                0,
                4,
            ],
            [
                3,
                2,
                0,
            ],
            [
                5,
                4,
                1,
            ],
            [
                0,
                1,
                4,
            ],
            [
                0,
                0,
                1,
            ],
        ],
        dtype=np.int32,
    )
    inputs["theta"] = np.zeros(shape=(8, xi_theta.size))
    inputs["theta_weight"] = np.zeros_like(inputs["theta"])
    inputs["theta_edges"] = np.zeros(shape=(8, 2))
    for row in (0, 4):
        inputs["theta"][row] = xi_theta
        inputs["theta_weight"][row] = xi_weights
        inputs["theta_edges"][row] = xi_edges
    return inputs


def select_rows(inputs, selected):
    """Retain the same fields and quadrature while selecting estimator rows.

    The selected indices increase, so every retained covariance cell keeps
    its left/right estimator order. Reversing that order would permit a
    different rounding sequence in finite-window correlations.
    """
    chosen = np.asarray(selected, dtype=np.intp)
    assert np.all(chosen[1:] > chosen[:-1])
    subset = dict(inputs)
    for key in (
        "row_kind",
        "row_fields",
        "theta",
        "theta_weight",
        "theta_edges",
    ):
        subset[key] = np.ascontiguousarray(inputs[key][chosen])
    return subset


def require_bits(actual, expected):
    """Compare float64 arrays including signed zero, after checking finiteness."""
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("adapter_name", ["production", "notebook"])
@pytest.mark.parametrize("large_area", [0, 1])
def test_selected_rows_keep_their_covariance(compiled, model, adapter_name, large_area):
    """Removing other estimators cannot change the retained physical result.

    The cubic-only, quadratic-only and mixed selections require different
    contraction lists. A quadratic-only request requires no P*T entries.
    Duplicate rows are exact copies; exchanged fields describe the same
    continuous observable but may reorder floating-point additions.
    """
    adapter = getattr(compiled, adapter_name)
    inputs = mixed_inputs(large_area=large_area)
    mean, terms = adapter.physical_covariance(model=model, inputs=inputs)
    selections = (
        [
            1,
            5,
            7,
        ],
        [
            0,
            2,
            4,
        ],
        [
            0,
            1,
            4,
            7,
        ],
    )
    assert terms[2, 1, 1] != 0.0
    assert terms[2, 5, 5] != terms[2, 1, 1]
    for selected in selections:
        subset = select_rows(inputs=inputs, selected=selected)
        actual_mean, actual_terms = adapter.physical_covariance(
            model=model, inputs=subset
        )
        expected_terms = terms[:, selected, :][:, :, selected]
        require_bits(actual=actual_mean, expected=mean[selected])
        require_bits(actual=actual_terms, expected=expected_terms)

    # Identical row definitions retain the same summation order, including
    # the two copies' mutual covariance. The exchanged-field comparison
    # instead uses the tolerance of the independent labeled-field integral.
    require_bits(actual=mean[[6]], expected=mean[[1]])
    require_bits(actual=terms[:, 6, 6], expected=terms[:, 1, 1])
    require_bits(actual=terms[:, 1, 6], expected=terms[:, 1, 1])
    np.testing.assert_allclose(mean[3], mean[1], rtol=3.0e-12, atol=0.0)
    np.testing.assert_allclose(
        terms[2, 3, 3], terms[2, 1, 1], rtol=3.0e-12, atol=0.0
    )


@pytest.mark.parametrize("adapter_name", ["production", "notebook"])
@pytest.mark.parametrize("large_area", [0, 1])
def test_contractions_keep_bits_across_worker_counts(
    compiled, model, adapter_name, large_area
):
    """One, two and four workers must preserve every fixed quadrature sum."""
    adapter = getattr(compiled, adapter_name)
    inputs = mixed_inputs(large_area=large_area)
    compiled.set_omp_threads(nthreads=1)
    expected = adapter.physical_covariance(model=model, inputs=inputs)
    try:
        for workers in (
            2,
            4,
            1,
        ):
            compiled.set_omp_threads(nthreads=workers)
            actual = adapter.physical_covariance(model=model, inputs=inputs)
            for value, reference in zip(actual, expected):
                require_bits(actual=value, expected=reference)
    finally:
        compiled.set_omp_threads(nthreads=1)
