"""Small analytic checks of derivatives and joint parameter information.

These tests use polynomial means and two-dimensional Gaussian examples,
so their answers are known without running a cosmology. They test the
statistical operations that later consume the physical C predictions.
No speed measurement or survey-size computation is performed.
"""

import numpy as np
import pytest
from scipy.special import roots_genlaguerre, gamma

from cosmolike_notebook_utils.fisher import (
    ExactMeanCache, differentiate, fisher_matrix, parameter_covariance,
    figure_of_merit,
)
from cosmolike_notebook_utils.covariance.precision import precision_expansion, split_precision_expansion
from des3pcf.survey import shifted_sources
from des3pcf.physical_provider import load_tables, save_tables


def test_exact_keys_and_polynomial_derivatives():
    """Resolve a small step and recover the derivative of a cubic exactly.

    A cache based on an approximate comparison would reuse the first
    prediction at the nearby point and incorrectly return zero slope.
    Returns:
      nothing; assertions compare with analytic polynomial derivatives.
    """
    calls = []

    def mean(parameters):
        """Return a cubic/linear pair and record each physical evaluation.

        Arguments:
          parameters = dictionary containing finite x and y.
        Returns:
          two dimensionless polynomial measurements.
        Side effects:
          Appends one parameter copy to calls for the cache check.
        """
        calls.append(dict(parameters))
        x = parameters["x"]
        y = parameters["y"]
        return np.array([x**3 + 2.0 * y, 3.0 * x - y])

    cached = ExactMeanCache(predict=mean)
    point = {"x": 0.7, "y": -0.2}
    cached(point)
    cached(dict(point))
    assert len(calls) == 1
    cached({"x": 0.7 + 1.0e-9, "y": -0.2})
    assert len(calls) == 2
    names, reference, coarse, fine = differentiate(
        predict=cached, parameters=point, steps={"x": 0.01, "y": 0.02},
    )
    expected = np.array([[3.0 * 0.7**2, 2.0], [3.0, -1.0]])
    assert names == ["x", "y"]
    np.testing.assert_allclose(reference, [0.7**3 - 0.4, 2.3], atol=1.0e-14)
    np.testing.assert_allclose(coarse, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(fine, expected, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("step_multiple", [1.0, 1.4])
def test_refined_stencil_must_remain_distinct(step_multiple):
    """Reject a rounded fine stencil before evaluating any physical mean.

    At x=1, h=one float spacing has a distinct positive h point but its
    positive h/2 point rounds to x. For h=1.4 spacings the positive h/2
    and h points instead round to the same value. Both defeat a formal
    h-versus-h/2 derivative comparison, despite passing an h-only check.
    """
    calls = []

    def mean(parameters):
        """Record whether invalid geometry reached the model callable."""
        calls.append(dict(parameters))
        return np.array([parameters["x"]])

    step = step_multiple * np.spacing(1.0)
    with pytest.raises(ValueError, match="float resolution"):
        differentiate(predict=mean, parameters={"x": 1.0}, steps={"x": step})
    assert calls == []


def test_joint_information_and_marginalized_fom():
    """Retain the covariance cross-block in a known Gaussian problem.

    A model with unit derivatives measures its two parameters directly.
    Its marginalized parameter covariance must equal the data covariance.
    Returns:
      nothing; assertions also compare covariance and precision input paths.
    """
    covariance = np.array([[4.0, 1.0], [1.0, 9.0]])
    jacobian = np.eye(2)
    names = ["w0", "wa"]
    information = fisher_matrix(
        derivatives=jacobian, names=names, covariance=covariance,
    )
    other = fisher_matrix(
        derivatives=jacobian, names=names, precision=np.linalg.inv(covariance),
    )
    np.testing.assert_allclose(information, other, rtol=1.0e-14)
    recovered = parameter_covariance(fisher=information, names=names)
    np.testing.assert_allclose(recovered, covariance, rtol=1.0e-14)
    fom = figure_of_merit(fisher=information, names=names, pair=names)
    np.testing.assert_allclose(fom, 1.0 / np.sqrt(35.0), rtol=1.0e-14)
    singular = np.ones((2, 2))
    with pytest.raises(ValueError, match="cannot resolve"):
        parameter_covariance(fisher=singular, names=names)


def test_wishart_quadratic_expectation_without_random_sampling():
    """Integrate the exact one-dimensional sample-variance distribution.

    For nu=4, nu*s/C follows chi-squared_4. Generalized Laguerre quadrature
    integrates its gamma density and the quadratic PME polynomial exactly.
    This tests the noise-bias correction against a distributional identity,
    rather than comparing it with a second copy of its implementation.
    Returns:
      nothing; the ensemble expectation matches the deterministic expansion.
    """
    nu = 4  # Five independent Gaussian realizations give four degrees of freedom.
    model = np.array([[2.0]])
    truth = 2.4
    nodes, weights = roots_genlaguerre(n=8, alpha=nu / 2.0 - 1.0)
    average = 0.0
    for node, weight in zip(nodes, weights):
        sample = np.array([[truth * 2.0 * node / nu]])
        estimate, diagnostics = precision_expansion(
            analytic=model, sample=sample, independent_count=nu + 1, order=2,
        )
        average += weight * estimate[0, 0] / gamma(nu / 2.0)
        assert diagnostics["independent_count"] == 5
    residual = truth / model[0, 0] - 1.0
    expected = (1.0 - residual + residual**2) / model[0, 0]
    np.testing.assert_allclose(average, expected, rtol=2.0e-14)


def test_independent_split_expectation_with_noncommuting_matrices():
    """Average independent matrix fluctuations with an exact finite ensemble.

    Each group's two equally likely covariances average to the same truth.
    All four independent pairings must recover the deterministic quadratic
    expansion even though individual matrices do not commute.
    Returns:
      nothing; the result checks both whitening and precision restoration.
    """
    model = np.array([[2.0, 0.3], [0.3, 1.4]])
    truth = np.array([[2.2, 0.2], [0.2, 1.3]])
    fluctuation_a = np.array([[0.1, 0.04], [0.04, -0.07]])
    fluctuation_b = np.array([[-0.05, 0.09], [0.09, 0.02]])
    average = np.zeros((2, 2))
    for sign_a in (-1.0, 1.0):
        for sign_b in (-1.0, 1.0):
            estimate = split_precision_expansion(
                analytic=model,
                sample_first=truth + sign_a * fluctuation_a,
                sample_second=truth + sign_b * fluctuation_b,
            )
            average += estimate / 4.0
    inverse = np.linalg.inv(model)
    difference = truth - model
    expected = inverse - inverse @ difference @ inverse
    expected += inverse @ difference @ inverse @ difference @ inverse
    np.testing.assert_allclose(average, expected, rtol=2.0e-14, atol=1.0e-16)


@pytest.mark.parametrize("negative_group", [0, 1])
def test_split_precision_requires_valid_sample_covariances(negative_group):
    """Distinguish a singular covariance from one with a negative variance.

    A finite sample set may leave a direction unresolved. Zero variance
    there is acceptable because the split expansion does not invert either
    sample matrix. A negative direction cannot describe any sample set.
    Check each group independently so neither can conceal the other's error.
    """
    model = np.eye(2)
    first = np.diag([1.0, 0.0])
    second = np.diag([0.0, 1.0])
    allowed = split_precision_expansion(
        analytic=model, sample_first=first, sample_second=second,
    )
    np.testing.assert_array_equal(allowed, 1.5 * np.eye(2))
    samples = [first.copy(), second.copy()]
    samples[negative_group][negative_group, negative_group] = -0.2
    with pytest.raises(ValueError, match="negative mode"):
        split_precision_expansion(
            analytic=model,
            sample_first=samples[0],
            sample_second=samples[1],
        )


def test_photoz_sign_and_normalization():
    """Move a narrow source distribution to larger redshift with positive bias.

    The mean shift is known because the Gaussian tails are negligible at
    both ends of the grid. This catches the common n(z+delta) sign error.
    Returns:
      nothing; input ownership and normalization are also checked.
    """
    z = np.linspace(0.0, 2.0, 401)
    source = np.exp(-0.5 * ((z - 0.7) / 0.08)**2)[None, :]
    before = source.copy()
    shifted = shifted_sources(z=z, sources=source, shifts=np.array([0.1]))
    np.testing.assert_allclose(np.trapz(shifted, x=z, axis=1), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(np.trapz(shifted[0] * z, x=z), 0.8, atol=1.0e-12)
    np.testing.assert_array_equal(source, before)
    # A negative shift can bring probability close to the observer. Its
    # endpoint remains zero and the surviving positive-z density is normalized.
    low_redshift = z**2 * np.exp(-z / 0.08)
    negative = shifted_sources(
        z=z, sources=low_redshift[None, :], shifts=np.array([-0.05]),
    )
    assert negative[0, 0] == 0.0
    np.testing.assert_allclose(np.trapz(negative, x=z, axis=1), 1.0, atol=1.0e-14)
    old_mean = np.trapz(z * low_redshift, x=z) / np.trapz(low_redshift, x=z)
    assert np.trapz(z * negative[0], x=z) < old_mean


def test_physics_table_serialization(tmp_path):
    """Keep shape, units metadata and cosmology values together in a numerical file.

    Arguments:
      tmp_path = pytest-owned temporary directory for this roundtrip check.
    Returns:
      nothing; asserts exact numerical values and rejects an accidental overwrite.
    """
    tables = {
        "k": np.array([0.01, 0.1, 1.0]),
        "p_linear0": np.array([10.0, 2.0, 0.1]),
        "z": np.array([0.0, 0.5, 1.0]),
        "p_nonlinear": np.ones((3, 3)),
        "chi": np.array([0.0, 1300.0, 2200.0]),
        "dz_dchi": np.full(3, 0.0004),
        "growth": np.array([1.0, 0.8, 0.6]),
        "sources": np.array([[0.0, 1.0, 0.0]]),
        "los_z": np.array([0.01, 0.3, 0.6]),
        "omega_m": 0.3,
        "omega_de": 0.7,
        "w0": -1.0,
        "wa": 0.0,
        "ns": 0.96,
        "sigma8": 0.8,
    }
    destination = tmp_path / "tables.npz"
    save_tables(path=destination, tables=tables, metadata={"producer": "test"})
    loaded, metadata = load_tables(path=destination)
    for name, value in tables.items():
        np.testing.assert_array_equal(loaded[name], value)
    assert metadata["producer"] == "test"
    with pytest.raises(FileExistsError):
        save_tables(path=destination, tables=tables, metadata={})
