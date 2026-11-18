"""Parameter information from the full selected joint data vector.

A Fisher matrix describes the local curvature of a Gaussian likelihood.
Here the covariance is fixed at the chosen reference cosmology. Each
column of the derivative matrix tells us how every measured quantity
changes when one parameter changes. Contracting those columns with the
precision (inverse covariance) retains the correlations between the
two- and three-point measurements. No compression is applied implicitly.

Finite differences use exact parameter keys: a small cosmological step
must never retrieve the prediction at a nearby, different cosmology.
All linear algebra is small compared with the physical prediction. The
code refuses singular Fisher matrices rather than reporting uncertainties
from an unannounced pseudoinverse.
"""

import numpy as np
from scipy.linalg import solve_triangular
from types import MappingProxyType
from .ownership import immutable_copy


def symmetric_matrix(values, name, positive=False):
    """Check a covariance-like matrix before a statistical contraction.

    Symmetry is tested relative to each matrix's own scale; this matters
    when aperture covariances have very small numerical units. Averaging
    the two triangles removes accepted floating-point asymmetry only.

    Arguments:
      values = finite square numerical array.
      name = quantity named in failure messages.
      positive = require a Cholesky factorization when True.
    Returns:
      owned symmetric float64 matrix, with the input's units.
    Raises:
      ValueError when shape, finiteness, symmetry or positivity fails.
    """
    matrix = np.array(values, dtype=np.float64, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square; received {matrix.shape}.")
    if matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values and at least one row.")
    scale = np.max(np.abs(matrix))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if np.max(np.abs(matrix - matrix.T)) > tolerance:
        raise ValueError(f"{name} is asymmetric; check its row ordering.")
    matrix = 0.5 * (matrix + matrix.T)
    if positive:
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"{name} is not positive definite; inspect its eigenmodes."
            ) from error
    return matrix


class ExactMeanCache:
    """Reuse predictions only for identical parameter dictionaries.

    Every parameter, including nuisance parameters, is part of the key.
    JointPrediction also supplies a cache_token for its mutable numerical
    controls. Changing those controls discards old predictions before a
    parameter lookup. Other callables must remain fixed, or their caller
    must explicitly clear this cache when their hidden inputs change.

    Arguments:
      predict = callable taking one dictionary and returning a 1D vector.
    Side effects:
      Stores owned, read-only predictions until this object is released.
    """

    def __init__(self, predict):
        """Keep the prediction callable and an initially empty table.

        Arguments:
          predict = deterministic callable for one fixed model and row order.
        Returns:
          nothing; initializes this instance without evaluating the model.
        """
        self.predict = predict
        self._values = {}
        self._state = None

    @property
    def values(self):
        """Expose retained entries for inspection without allowing replacement."""
        return MappingProxyType(self._values)

    def clear(self):
        """Forget predictions after changing hidden inputs of a custom callable."""
        self._values.clear()
        self._state = None

    def __call__(self, parameters):
        """Return an owned prediction for precisely these parameter values.

        Hexadecimal float strings preserve every binary floating-point
        bit relevant to a finite-difference step. No numerical tolerance
        participates in deciding whether a stored calculation can be used.

        Arguments:
          parameters = dictionary of finite scalar parameter values.
        Returns:
          read-only 1D float64 prediction with the callable's units.
        Raises:
          ValueError for nonfinite parameters or a malformed prediction.
        """
        state = (id(self.predict),)
        if hasattr(self.predict, "cache_token"):
            state += (self.predict.cache_token(),)
        if state != self._state:
            self.clear()
            self._state = state
        entries = []
        for name in sorted(parameters):
            value = float(parameters[name])
            if not np.isfinite(value):
                raise ValueError(f"Parameter {name}={value} must be finite.")
            entries.append((name, value.hex()))
        key = tuple(entries)
        if key not in self._values:
            result = self.predict(dict(parameters))
            value = np.array(result, dtype=np.float64, copy=True)
            if value.ndim != 1 or not np.all(np.isfinite(value)):
                raise ValueError("Mean prediction must be a finite 1D array.")
            self._values[key] = immutable_copy(value=value)
        return self._values[key]


def differentiate(predict, parameters, steps, bounds=None):
    """Differentiate every selected parameter with a fourth-order stencil.

    Four displaced means give [-f(2h)+8f(h)-8f(-h)+f(-2h)]/(12h).
    Repeating at h/2 exposes step sensitivity without treating agreement
    as proof that the underlying integration has converged. Bounds are
    checked before any model evaluation; no one-sided replacement is made.

    Arguments:
      predict = mean callable, normally ExactMeanCache.
      parameters = all reference parameter values.
      steps = ordered dictionary of positive absolute parameter steps.
      bounds = optional dictionary mapping names to [minimum, maximum].
    Returns:
      names, reference mean, derivatives at h, derivatives at h/2;
      derivative arrays have shape [ndata, nparameter].
    Raises:
      ValueError for invalid steps, bounds, names or changing vector size.
    """
    if not steps:
        raise ValueError("At least one differentiated parameter is required.")
    if bounds is None:
        bounds = {}
    names = list(steps)
    for name in names:
        if name not in parameters:
            raise ValueError(f"Step {name} has no reference parameter value.")
        step = float(steps[name])
        value = float(parameters[name])
        if not np.isfinite(step) or step <= 0.0 or not np.isfinite(value):
            raise ValueError(f"{name} requires a finite value and positive step.")
        # Both stencils use this union of coordinates. Comparing adjacent
        # values detects an h/2 point rounded to the fiducial or to an h
        # point, even when the original +/-h displacement was distinct.
        previous = None
        for offset in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
            coordinate = value + offset * step
            if not np.isfinite(coordinate):
                raise ValueError(f"The stencil for {name} leaves finite values.")
            if previous is not None and coordinate <= previous:
                raise ValueError(
                    f"Step {step} for {name} collapses the h or h/2 "
                    "stencil at float resolution."
                )
            previous = coordinate
        if name in bounds:
            lower, upper = bounds[name]
            if value - 2.0 * step < lower or value + 2.0 * step > upper:
                raise ValueError(f"{name}'s stencil leaves its supplied bounds.")
    reference = np.array(predict(dict(parameters)), dtype=np.float64, copy=True)
    if reference.ndim != 1 or not np.all(np.isfinite(reference)):
        raise ValueError("Reference prediction must be finite and one-dimensional.")
    derivatives = np.empty((reference.size, len(names)), dtype=np.float64)
    refined = np.empty_like(derivatives)
    for column, name in enumerate(names):
        for refinement in range(2):
            step = float(steps[name]) / (2.0 ** refinement)
            displaced = []
            # Order is -2h,-h,+h,+2h, identical for both step choices.
            for offset in (-2.0, -1.0, 1.0, 2.0):
                point = dict(parameters)
                point[name] = float(parameters[name]) + offset * step
                value = np.asarray(predict(point), dtype=np.float64)
                if value.shape != reference.shape or not np.all(np.isfinite(value)):
                    raise ValueError(f"Prediction shape/values changed at {name}.")
                displaced.append(value)
            derivative = (
                displaced[0] - 8.0 * displaced[1]
                + 8.0 * displaced[2] - displaced[3]
            ) / (12.0 * step)
            if refinement == 0:
                derivatives[:, column] = derivative
            else:
                refined[:, column] = derivative
    return names, reference, derivatives, refined


def fisher_matrix(derivatives, names, covariance=None, precision=None, priors=None):
    """Calculate fixed-covariance Fisher information and Gaussian priors.

    If C=L L^T, the whitened derivatives Z=L^-1 J give F=Z^T Z.
    This avoids explicitly inverting a covariance. A supplied precision
    is contracted directly, which also permits precision-matrix expansion.
    Exactly one of covariance and precision is required.

    Arguments:
      derivatives = finite [ndata,nparameter] array in names order.
      names = unique parameter names matching derivative columns.
      covariance = optional positive definite [ndata,ndata] matrix.
      precision = optional positive definite inverse-covariance matrix.
      priors = optional dictionary of Gaussian standard deviations.
    Returns:
      [nparameter,nparameter] Fisher matrix; it may be singular.
    Raises:
      ValueError for inconsistent arrays, priors or matrix choices.
    """
    jacobian = np.asarray(derivatives, dtype=np.float64)
    if (jacobian.ndim != 2 or jacobian.shape[1] != len(names)
            or not np.all(np.isfinite(jacobian)) or len(set(names)) != len(names)):
        raise ValueError("Derivative columns must match unique parameter names.")
    if (covariance is None) == (precision is None):
        raise ValueError("Supply exactly one covariance or precision matrix.")
    if covariance is not None:
        matrix = symmetric_matrix(covariance, name="Covariance", positive=True)
        if matrix.shape[0] != jacobian.shape[0]:
            raise ValueError("Covariance rows do not match derivative rows.")
        factor = np.linalg.cholesky(matrix)
        whitened = solve_triangular(factor, jacobian, lower=True)
        fisher = whitened.T @ whitened
    else:
        matrix = symmetric_matrix(precision, name="Precision", positive=True)
        if matrix.shape[0] != jacobian.shape[0]:
            raise ValueError("Precision rows do not match derivative rows.")
        fisher = jacobian.T @ matrix @ jacobian
    if priors is not None:
        for name, width in priors.items():
            if name not in names or not np.isfinite(width) or width <= 0.0:
                raise ValueError(f"Prior {name} requires a known name and sigma>0.")
            index = names.index(name)
            fisher[index, index] += 1.0 / width**2
    return 0.5 * (fisher + fisher.T)


def parameter_covariance(fisher, names):
    """Invert parameter information only when every direction is constrained.

    Parameters with zero derivatives, such as an IA redshift slope at
    zero IA amplitude, need removal or an explicitly justified prior.
    A pseudoinverse would hide that absence of information.

    Arguments:
      fisher = square information matrix in names order.
      names = parameter labels for a useful failure message.
    Returns:
      marginalized parameter covariance, with squared parameter units.
    Raises:
      ValueError if the matrix has a numerically unresolved eigenmode.
    """
    matrix = symmetric_matrix(fisher, name="Fisher matrix")
    if matrix.shape[0] != len(names):
        raise ValueError("Fisher dimensions do not match parameter names.")
    diagonal = np.diag(matrix)
    if np.any(diagonal <= 0.0):
        raise ValueError(f"Unconstrained Fisher direction among {names}.")
    # Diagonal scaling makes the rank test independent of parameter units.
    scales = np.sqrt(diagonal)
    normalized = matrix / scales[:, None] / scales[None, :]
    eigenvalues = np.linalg.eigvalsh(normalized)
    limit = 128.0 * matrix.shape[0] * np.finfo(np.float64).eps
    if eigenvalues[0] <= limit * eigenvalues[-1]:
        raise ValueError(
            f"Fisher matrix cannot resolve all of {names}; "
            "remove a null parameter or specify a physical prior."
        )
    identity = np.eye(matrix.shape[0])
    inverse_normalized = np.linalg.solve(normalized, identity)
    return inverse_normalized / scales[:, None] / scales[None, :]


def figure_of_merit(fisher, names, pair):
    """Return the inverse area scale for a marginalized parameter pair.

    The convention is FoM=1/sqrt(det(C_pair)); no contour-probability
    multiplier is included. Marginalizing first retains the effect of
    every other fitted parameter and its specified prior.

    Arguments:
      fisher = information matrix in names order.
      names = unique parameter labels.
      pair = two distinct names, for example ['w0','wa'].
    Returns:
      positive scalar in inverse product-parameter units.
    Raises:
      ValueError for an unknown/repeated pair or unresolved Fisher matrix.
    """
    if len(pair) != 2 or pair[0] == pair[1]:
        raise ValueError("The FoM requires two distinct parameter names.")
    indices = []
    for name in pair:
        if name not in names:
            raise ValueError(f"FoM parameter {name} is absent from {names}.")
        indices.append(names.index(name))
    covariance = parameter_covariance(fisher=fisher, names=names)
    selected = covariance[np.ix_(indices, indices)]
    return 1.0 / np.sqrt(np.linalg.det(selected))
