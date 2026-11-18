"""Separate information already in xi from information added by Map3.

The extra three-point information is conditional on the measured two-point
vector. Correlated fluctuations must first be removed from both its mean
derivatives and covariance. Adding independently computed Fisher matrices
would count these fluctuations twice. All operations here are small dense
linear algebra; the physical derivatives are supplied by the C-backed mean.
"""

import numpy as np
from scipy.linalg import solve_triangular

from .forecast import fisher_matrix, parameter_covariance, symmetric_matrix


def compare_information(derivatives, names, kind, covariance=None,
                        precision=None, priors=None):
    """Compare two-point and joint information with a common parameter basis.

    Let D2 and D3 contain derivatives of xi and Map3. With C23 retained,
    the new three-point covariance is S=C33-C32 C22^-1 C23 and its new
    derivatives are E=D3-C32 C22^-1 D2. Then the likelihood information is
    F_joint=F_2+E.T S^-1 E. This equality holds before adding priors.

    Arguments:
      derivatives = finite [nrow,nparameter] mean derivatives.
      names = unique labels in derivative-column order.
      kind = [nrow] labels xip, xim or map3; both probe groups must occur.
      covariance, precision = exactly one positive definite full matrix.
          A supplied precision defines an effective covariance by inversion;
          its upper block is not the marginal two-point precision.
      priors = optional Gaussian standard deviations by parameter name.
    Returns:
      dictionary of likelihood, prior and posterior Fisher matrices, plus
      conditional derivatives/covariance and original row indices.
    Raises:
      ValueError for inconsistent rows, invalid priors or nonpositive matrices.
    """
    jacobian = np.asarray(derivatives, dtype=np.float64)
    kinds = np.asarray(kind)
    if (kinds.ndim != 1 or jacobian.ndim != 2
            or kinds.size != jacobian.shape[0]
            or np.any(~np.isin(kinds, ("xip", "xim", "map3")))):
        raise ValueError("Each derivative row needs its xip, xim or map3 label.")
    second = np.flatnonzero(kinds != "map3")
    third = np.flatnonzero(kinds == "map3")
    if second.size == 0 or third.size == 0:
        raise ValueError("Information comparison requires both xi and Map3 rows.")
    joint = fisher_matrix(
        derivatives=jacobian, names=names, covariance=covariance,
        precision=precision,
    )
    if covariance is not None:
        matrix = symmetric_matrix(values=covariance, name="Covariance", positive=True)
    else:
        # PME supplies a precision, not a corrected covariance estimate.
        # Inverting it here defines only the Gaussian model used for this
        # decomposition; it makes no claim of unbiased covariance estimation.
        inverse = symmetric_matrix(values=precision, name="Precision", positive=True)
        factor = np.linalg.cholesky(inverse)
        inverse_factor = solve_triangular(factor, np.eye(kinds.size), lower=True)
        matrix = inverse_factor.T @ inverse_factor
    block22 = matrix[np.ix_(second, second)]
    block23 = matrix[np.ix_(second, third)]
    block33 = matrix[np.ix_(third, third)]
    factor22 = np.linalg.cholesky(block22)
    whitened23 = solve_triangular(factor22, block23, lower=True)
    whitened_d2 = solve_triangular(factor22, jacobian[second], lower=True)
    conditional_covariance = block33 - whitened23.T @ whitened23
    conditional_derivatives = jacobian[third] - whitened23.T @ whitened_d2
    conditional = fisher_matrix(
        derivatives=conditional_derivatives, names=names,
        covariance=conditional_covariance,
    )
    two_point = whitened_d2.T @ whitened_d2
    # A prior belongs to parameters, not to either data block. Form it once
    # and add that same matrix once to each posterior being compared.
    prior = fisher_matrix(
        derivatives=np.zeros((1, len(names))), names=names,
        covariance=np.ones((1, 1)), priors=priors,
    )
    return {
        "fisher_joint_likelihood": joint,
        "fisher_two_point_likelihood": two_point,
        "fisher_conditional_three_point": conditional,
        "fisher_prior": prior,
        "fisher_joint": joint + prior,
        "fisher_two_point": two_point + prior,
        "conditional_covariance": conditional_covariance,
        "conditional_derivatives": conditional_derivatives,
        "two_point_indices": second,
        "three_point_indices": third,
    }


def marginalized_errors(comparison, names):
    """Return resolvable errors and explicit rank diagnostics for each probe.

    A zero derivative or nearly unconstrained parameter combination cannot
    be repaired by a pseudoinverse. The joint experiment may resolve a
    direction which xi alone cannot; the two statuses are therefore separate.

    Arguments:
      comparison = dictionary returned by compare_information.
      names = the same parameter-column labels.
    Returns:
      arrays dictionary containing only resolved parameter covariances and
      sigmas, and a separate dictionary of human-readable rank statuses.
    """
    arrays = {}
    statuses = {}
    for probe in ("two_point", "joint"):
        try:
            covariance = parameter_covariance(
                fisher=comparison[f"fisher_{probe}"], names=names,
            )
            arrays[f"parameter_covariance_{probe}"] = covariance
            arrays[f"sigma_{probe}"] = np.sqrt(np.diag(covariance))
            statuses[probe] = "resolved"
        except ValueError as error:
            statuses[probe] = str(error)
    if "sigma_two_point" in arrays and "sigma_joint" in arrays:
        arrays["sigma_ratio_two_point_over_joint"] = (
            arrays["sigma_two_point"] / arrays["sigma_joint"]
        )
    return arrays, statuses
