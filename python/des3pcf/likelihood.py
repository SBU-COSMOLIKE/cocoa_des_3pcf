"""A fixed Gaussian likelihood for one labeled joint shear data vector.

The covariance contains the two-point, three-point and cross terms. Selecting
measurements means selecting the same rows of the prediction and both axes
of this matrix before its Cholesky factorization. Selecting entries from an
inverse of the full matrix would instead condition on the discarded data.

The likelihood contains no parameter prior. Nautilus draws from the explicit
prior distributions in the inference YAML. Adding the calibration penalties
here as well would count those priors twice. The Gaussian normalization is
retained so that the sampler's evidence refers to this declared density.
"""

import numpy as np
from scipy.linalg import solve_triangular

from .forecast import symmetric_matrix
from .ownership import immutable_copy, PreparedGeometry


class FixedGaussianLikelihood(PreparedGeometry):
    """Retain an immutable data selection and its fixed covariance.

    Arguments:
      predict = callable from a parameter dictionary to the full mean vector.
      rows = SurveyRows describing that full vector in measurement order.
      covariance = full square covariance in the same row order.
      parameters = baseline parameter dictionary, including fixed parameters.
      data = full observed vector, or None to generate a noiseless fiducial.
      indices = explicit integer selection, or None to retain every row.
    Side effects:
      Synthetic construction evaluates the supplied fiducial once. Physical
      table reuse belongs to predict; this class retains no sample history.
    """

    def __init__(
        self,
        predict,
        rows,
        covariance,
        parameters,
        data=None,
        indices=None,
    ):
        """Validate dimensions and factor the selected covariance once.

        Cholesky writes C=L L^T, with L lower triangular. Solving L y=r
        transforms a residual into coordinates with unit noise covariance;
        chi squared is then the sum of y squared. No matrix inverse is made.
        """
        full = symmetric_matrix(values=covariance, name="Joint covariance")
        size = rows.values.size
        if full.shape != (size, size):
            raise ValueError(
                f"Covariance shape {full.shape} must equal ({size}, {size}); "
                "load the matrix associated with these full row labels."
            )
        selected = indices
        if selected is None:
            selected = np.arange(size, dtype=np.int64)
        selected_rows = rows.select(indices=selected)
        if selected_rows.values.size == 0:
            raise ValueError("The likelihood selection is empty; retain data rows.")
        self.indices = immutable_copy(value=np.asarray(selected, dtype=np.int64))
        self.rows = selected_rows
        self.predict = predict
        self.full_size = size
        self.parameters = dict(parameters)
        for name, value in self.parameters.items():
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"Baseline {name}={value} must be a finite scalar.")

        # np.ix_ forms the Cartesian product of the selected row indices.
        # Both axes therefore retain the exact order used by the data vector.
        matrix = full[np.ix_(self.indices, self.indices)]
        matrix = symmetric_matrix(
            values=matrix, name="Selected covariance", positive=True,
        )
        self.covariance = immutable_copy(value=matrix)
        self.cholesky = immutable_copy(value=np.linalg.cholesky(matrix))
        diagonal = np.diag(self.cholesky)
        log_determinant = 2.0 * np.sum(np.log(diagonal))
        self.log_normalization = -0.5 * (
            selected_rows.values.size * np.log(2.0 * np.pi) + log_determinant
        )

        self.data_kind = "observed"
        if data is None:
            self.data_kind = "synthetic_fiducial"
            data = self.predict(dict(self.parameters))
        values = self._full_vector(values=data, name="Likelihood data")
        self.data = immutable_copy(value=values[self.indices])
        # Synthetic data and the normalization refer to this exact baseline.
        # A notebook must construct another owner to replace that definition.
        self._seal_geometry()

    def _full_vector(self, values, name):
        """Require a finite full vector before selecting its measurement rows.

        Arguments:
          values = numerical prediction or data array.
          name = quantity named in the diagnostic.
        Returns:
          owned one-dimensional array in full measurement order.
        """
        array = np.array(values, dtype=np.float64, copy=True)
        if array.shape != (self.full_size,) or not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} has shape {array.shape}; provide {self.full_size} finite "
                "values in the full row order before selecting measurements."
            )
        return array

    def mean(self, parameters):
        """Merge a sampled point with fixed values and return selected means.

        Arguments:
          parameters = dictionary containing some or all baseline names.
        Returns:
          owned vector in the selected measurement order.
        Raises:
          ValueError for an unknown/nonfinite parameter or malformed prediction.
        """
        point = dict(self.parameters)
        for name, value in parameters.items():
            if name not in point:
                raise ValueError(f"Unknown sampled parameter {name}; check the YAML.")
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"Sampled {name}={value} must be a finite scalar.")
            point[name] = float(value)
        prediction = self.predict(point)
        full = self._full_vector(values=prediction, name="Mean prediction")
        return full[self.indices]

    def score_mean(self, mean):
        """Score a selected mean without repeating its physical calculation.

        Arguments:
          mean = finite vector in the selected row order.
        Returns:
          log likelihood and chi squared, with no parameter prior.
        """
        values = np.asarray(mean, dtype=np.float64)
        if values.shape != self.data.shape or not np.all(np.isfinite(values)):
            raise ValueError("Selected mean must have the finite shape of the data.")
        residual = values - self.data
        whitened = solve_triangular(
            a=self.cholesky, b=residual, lower=True, check_finite=False,
        )
        chi_squared = float(np.dot(whitened, whitened))
        return float(self.log_normalization - 0.5 * chi_squared), chi_squared

    def __call__(self, parameters):
        """Return the normalized log likelihood for one sampled dictionary."""
        prediction = self.mean(parameters=parameters)
        likelihood, _ = self.score_mean(mean=prediction)
        return likelihood
