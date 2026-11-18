"""Test local Fisher preservation and the finite-mock sampling prescription.

The small matrices contain nonzero xi/Map3 cross-correlations. Removing
those correlations or compressing only the standalone xi response gives
detectably different information. No cosmological integration is needed.
"""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import multivariate_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from des3pcf.compression import LinearCompression, fit_compression, load_compression
from des3pcf.likelihood import FixedGaussianLikelihood, FiniteSimulationLikelihood
from des3pcf.survey import SurveyRows


def example():
    """Supply a response hidden in xi-only MOPED but measured jointly."""
    covariance = np.array([[1., 0., 0.], [0., 1., .6], [0., .6, 1.]])
    derivatives = np.array([[1.], [0.], [1.]])
    rows = SurveyRows(
        kind=np.array(["xip", "xim", "map3"]),
        bins=np.array([[0, 0, -1], [0, 0, -1], [0, 0, 0]]),
        angles=np.ones((3, 3)), values=np.zeros(3),
        labels=np.array(["xip:0", "xim:0", "map3:0"]),
    )
    return covariance, derivatives, rows


def linear_mean(parameters):
    """Expose one amplitude shared by xi and Map3."""
    return parameters["amplitude"] * np.array([1., 0., 1.])


def assumptions():
    """Declare independent sample covariance and an independent fixed basis."""
    return {
        "covariance_kind": "sample",
        "independent_mocks": True,
        "independent_data": True,
        "gaussian_mocks": True,
        "compression_independent": True,
    }


@pytest.mark.parametrize("mode", ["joint", "xi_moped", "conditional_xi"])
def test_joint_information_and_retained_map3(mode):
    """Joint and conditional bases close Fisher; xi-only loses the cross signal."""
    covariance, derivatives, rows = example()
    transform = fit_compression(
        covariance=covariance, derivatives=derivatives, labels=rows.labels,
        kind=rows.kind, mode=mode, metadata={"source": "analytic example"},
    )
    reduced = transform.covariance(values=covariance)
    response = transform.apply(values=derivatives)
    fisher = response.T @ np.linalg.solve(reduced, response)
    full = derivatives.T @ np.linalg.solve(covariance, derivatives)
    if mode == "xi_moped":
        np.testing.assert_allclose(actual=fisher, desired=[[2.]], atol=1.e-14)
        assert full[0, 0] - fisher[0, 0] > .5
    else:
        np.testing.assert_allclose(actual=fisher, desired=full, rtol=2.e-14)
    if mode != "joint":
        np.testing.assert_array_equal(x=transform.matrix[-1], y=[0., 0., 1.])
        assert transform.output_labels[-1] == rows.labels[-1]


@pytest.mark.parametrize("mode", ["joint", "conditional_xi"])
def test_units_order_and_dependent_parameter_columns(mode):
    """Unequal units, reordered rows and a redundant parameter preserve the span."""
    covariance, derivatives, rows = example()
    order = np.array([2, 0, 1])
    scale = np.array([1.e-8, 1.e-4, 1.e-4])
    covariance = covariance[np.ix_(order, order)] * scale[:, None] * scale[None, :]
    derivatives = np.column_stack((derivatives[order], 1.e9 * derivatives[order]))
    derivatives *= scale[:, None]
    transform = fit_compression(
        covariance=covariance, derivatives=derivatives, labels=rows.labels[order],
        kind=rows.kind[order], mode=mode, metadata={"source": "unit check"},
    )
    assert transform.metadata["rank"] == 1
    response = transform.apply(values=derivatives)
    reduced = transform.covariance(values=covariance)
    np.testing.assert_allclose(
        actual=response.T @ np.linalg.solve(reduced, response),
        desired=derivatives.T @ np.linalg.solve(covariance, derivatives), rtol=3.e-14,
    )


def test_owned_matrix_serialization_and_row_identity(tmp_path):
    """Saved nested provenance and numerical arrays survive immutable ownership."""
    matrix = np.array([[1., 2., 0.], [0., 0., 1.]])
    metadata = {"training": {"parameters": ["amplitude"], "fiducial": .1}}
    labels = np.array(["a", "b", "c"])
    transform = LinearCompression(matrix=matrix, input_labels=labels,
                                  output_labels=["mode", "c"], metadata=metadata)
    matrix[:] = 0.
    labels[:] = "wrong"
    metadata["training"]["fiducial"] = 2.
    assert transform.metadata["training"]["fiducial"] == .1
    path = tmp_path / "basis.npz"
    transform.save(path=path)
    restored = load_compression(path=path)
    assert restored.metadata_json == transform.metadata_json
    np.testing.assert_array_equal(x=restored.matrix, y=transform.matrix)
    with pytest.raises(FileExistsError):
        transform.save(path=path)
    with pytest.raises(ValueError, match="labels differ"):
        transform.check_labels(labels=["b", "a", "c"])
    with pytest.raises(ValueError):
        transform.matrix.setflags(write=True)


@pytest.mark.parametrize("scale", [1.e-200, 1.e200])
def test_extreme_parameter_units_do_not_erase_response(scale):
    """Normalization must not square raw values before removing their units."""
    covariance, derivatives, rows = example()
    transform = fit_compression(
        covariance=covariance, derivatives=derivatives * scale,
        labels=rows.labels, kind=rows.kind, mode="conditional_xi",
        metadata={"source": "unit test"},
    )
    assert transform.metadata["rank"] == 1
    response = transform.apply(values=derivatives)
    reduced = transform.covariance(values=covariance)
    np.testing.assert_allclose(
        actual=response.T @ np.linalg.solve(reduced, response),
        desired=derivatives.T @ np.linalg.solve(covariance, derivatives), rtol=3.e-14,
    )


def test_compressed_likelihood_transforms_data_model_and_covariance():
    """The likelihood applies one fixed T to every statistical quantity."""
    covariance, derivatives, rows = example()
    selection = np.array([2, 1, 0])
    selected_rows = rows.select(indices=selection)
    transform = fit_compression(
        covariance=covariance[np.ix_(selection, selection)],
        derivatives=derivatives[selection], labels=selected_rows.labels,
        kind=selected_rows.kind, mode="conditional_xi", metadata={"source": "test"},
    )
    likelihood = FixedGaussianLikelihood(
        predict=linear_mean, rows=rows, covariance=covariance,
        parameters={"amplitude": 0.}, data=np.array([.2, -.1, .4]),
        indices=selection, compression=transform,
    )
    expected_mean = transform.apply(values=linear_mean({"amplitude": .3})[selection])
    np.testing.assert_array_equal(x=likelihood.mean(parameters={"amplitude": .3}),
                                  y=expected_mean)
    _, measured = likelihood.score_mean(mean=expected_mean)
    expected_data = transform.apply(values=np.array([.2, -.1, .4])[selection])
    np.testing.assert_array_equal(x=likelihood.data, y=expected_data)
    residual = expected_mean - expected_data
    np.testing.assert_allclose(actual=measured,
                              desired=residual @ np.linalg.solve(likelihood.covariance,
                                                                 residual))
    assert not hasattr(likelihood.rows, "angles")


@pytest.mark.parametrize("compressed", [False, True])
def test_percival_matches_independent_multivariate_t(compressed):
    """The full normalization and tail agree with SciPy's t density."""
    covariance, derivatives, rows = example()
    transform = None
    if compressed:
        transform = fit_compression(
            covariance=covariance, derivatives=derivatives, labels=rows.labels,
            kind=rows.kind, mode="conditional_xi", metadata={"source": "analytic"},
        )
    likelihood = FiniteSimulationLikelihood(
        predict=linear_mean, rows=rows, covariance=covariance,
        parameters={"amplitude": 0.}, data=rows.values, compression=transform,
        mock_count=50, parameter_count=1, assumptions=assumptions(),
    )
    degrees = likelihood.exponent - likelihood.dimension
    if not compressed:
        np.testing.assert_allclose(actual=likelihood.exponent, desired=25804. / 517.,
                                  rtol=1.e-15)
    reference = multivariate_t(
        loc=likelihood.data,
        shape=(likelihood.mock_count - 1.) / degrees * likelihood.covariance,
        df=degrees,
    )
    for value in (0., .3, 8.):
        point = {"amplitude": value}
        expected = reference.logpdf(likelihood.mean(parameters=point))
        np.testing.assert_allclose(actual=likelihood(parameters=point),
                                  desired=expected, atol=2.e-14, rtol=2.e-14)


def test_percival_rejects_incompatible_covariance_and_invalid_normalization():
    """PME covariances and nonintegrable full-vector densities are refused."""
    covariance, _, rows = example()
    for field, value in (("covariance_kind", "pme"), ("independent_mocks", False)):
        declared = assumptions()
        declared[field] = value
        with pytest.raises(ValueError):
            FiniteSimulationLikelihood(
                predict=linear_mean, rows=rows, covariance=covariance,
                parameters={"amplitude": 0.}, mock_count=50, parameter_count=1,
                assumptions=declared, data=rows.values,
            )
    # Ns>Nd+4 alone does not guarantee a normalized data density. With
    # many data rows and few parameters, the matching exponent can be Nd or less.
    large = SurveyRows(
        kind=np.full(30, "map3"), bins=np.zeros((30, 3), dtype=int),
        angles=np.ones((30, 3)), values=np.zeros(30),
        labels=np.array([f"m{index}" for index in range(30)]),
    )
    with pytest.raises(ValueError, match="normalized density"):
        FiniteSimulationLikelihood(
            predict=linear_mean, rows=large, covariance=np.eye(30),
            parameters={"amplitude": 0.}, mock_count=40, parameter_count=1,
            assumptions=assumptions(), data=large.values,
        )


def test_percival_tends_to_gaussian_for_many_independent_mocks():
    """The finite-sample density has the fixed-Gaussian large-Ns limit."""
    covariance, _, rows = example()
    arguments = dict(predict=linear_mean, rows=rows, covariance=covariance,
                     parameters={"amplitude": 0.}, data=rows.values)
    finite = FiniteSimulationLikelihood(**arguments, mock_count=1000000,
                                       parameter_count=1, assumptions=assumptions())
    gaussian = FixedGaussianLikelihood(**arguments)
    np.testing.assert_allclose(actual=finite(parameters={"amplitude": 1.}),
                              desired=gaussian(parameters={"amplitude": 1.}), atol=2.e-5)
