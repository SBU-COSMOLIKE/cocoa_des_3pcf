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
from scipy.special import gammaln

from .compression import CompressedRows
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
      compression = fixed LinearCompression of the selected rows, or None.
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
        compression=None,
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
        self.compression = compression
        if compression is not None:
            compression.check_labels(labels=selected_rows.labels)
        self.predict = predict
        self.full_size = size
        self.parameters = dict(parameters)
        for name, value in self.parameters.items():
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"Baseline {name}={value} must be a finite scalar.")

        # np.ix_ forms the Cartesian product of the selected row indices.
        # Both axes therefore retain the exact order used by the data vector.
        matrix = full[np.ix_(self.indices, self.indices)]
        if compression is not None:
            matrix = compression.covariance(values=matrix)
        matrix = symmetric_matrix(
            values=matrix, name="Selected covariance", positive=True,
        )
        self.covariance = immutable_copy(value=matrix)
        self.cholesky = immutable_copy(value=np.linalg.cholesky(matrix))
        diagonal = np.diag(self.cholesky)
        log_determinant = 2.0 * np.sum(np.log(diagonal))
        self.log_normalization = -0.5 * (
            matrix.shape[0] * np.log(2.0 * np.pi) + log_determinant
        )

        self.data_kind = "observed"
        if data is None:
            self.data_kind = "synthetic_fiducial"
            data = self.predict(dict(self.parameters))
        values = self._full_vector(values=data, name="Likelihood data")
        selected_data = values[self.indices]
        if compression is not None:
            selected_data = compression.apply(values=selected_data)
            self.rows = CompressedRows(
                labels=compression.output_labels, values=selected_data,
            )
        self.data = immutable_copy(value=selected_data)
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
            # A full point can include fixed integer model selectors as
            # well as continuous parameters. Preserve their supplied types:
            # converting every value to float makes native selector reads
            # fail when a changed amplitude forces a real recomputation.
            if isinstance(value, (int, np.integer)):
                point[name] = value
            else:
                point[name] = float(value)
        prediction = self.predict(point)
        full = self._full_vector(values=prediction, name="Mean prediction")
        selected = full[self.indices]
        if self.compression is not None:
            selected = self.compression.apply(values=selected)
        return selected

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


class FiniteSimulationLikelihood(PreparedGeometry):
    """Use the Percival et al. finite-sample prescription for a fixed sample S.

    Equation 54 of arXiv:2108.10402 defines the exponent m, and equation 52
    gives L proportional to [1+chi_squared/(Ns-1)]^(-m/2). Here Nd counts
    rows AFTER selection and compression, and Np counts varied mean
    parameters, including nuisance parameters. S must be the ordinary
    unbiased sample covariance, without a Hartlap factor. Analytical,
    hybrid and precision-matrix-expansion covariances do not follow its
    sampling law and are refused.

    Gaussian distributed mock vectors and independent realizations give
    the Wishart sampling law for S assumed in the derivation. Independence
    alone does not establish that law. The declarations retain these
    assumptions for later checking against the actual mock ensemble.

    A normalized density exists only for m>Nd. This implementation requires
    that condition so Nautilus cannot report evidence from an unspecified
    proportionality constant. Gaussian compression preserves local Fisher
    information; it does not promise an unchanged finite-sample posterior.
    """

    def __init__(
        self, predict, rows, covariance, parameters, mock_count, parameter_count,
        assumptions, data=None, indices=None, compression=None,
    ):
        """Validate statistical assumptions before constructing a physical datum.

        Arguments:
          predict, rows, covariance, parameters, data, indices, compression =
            the same inputs as FixedGaussianLikelihood.
          mock_count = number Ns of independent covariance realizations.
          parameter_count = number Np of varied continuous mean parameters.
          assumptions = explicit declarations: covariance_kind='sample',
            independent_mocks=True, independent_data=True, gaussian_mocks=True,
            and, when using
            compression, compression_independent=True. These declarations
            record the user's statistical model; they do not verify mocks.
        Raises:
          ValueError for incompatible provenance, Ns<=Nd+4, or m<=Nd.
        """
        if not isinstance(assumptions, dict):
            raise ValueError("Finite-simulation likelihood requires assumptions.")
        if assumptions.get("covariance_kind") != "sample":
            raise ValueError("Finite-simulation likelihood requires raw sample S.")
        for name in ("independent_mocks", "independent_data", "gaussian_mocks"):
            if assumptions.get(name) is not True:
                raise ValueError(f"Finite-simulation assumption {name} must be True.")
        if compression is not None:
            if assumptions.get("compression_independent") is not True:
                raise ValueError("Compression must be independent of the sample S.")
            if compression.metadata.get("covariance_training_independent") is False:
                raise ValueError(
                    "Saved compression records training on sample S; use an "
                    "independent pilot basis or the fixed-Gaussian likelihood."
                )
        selected_rows = rows
        if indices is not None:
            selected_rows = rows.select(indices=indices)
        dimension = selected_rows.values.size
        if compression is not None:
            compression.check_labels(labels=selected_rows.labels)
            dimension = compression.matrix.shape[0]
        for name, value in (("mock_count", mock_count),
                            ("parameter_count", parameter_count)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name}={value} must be an integer.")
        if mock_count <= dimension + 4:
            raise ValueError(f"Ns={mock_count} must exceed Nd+4={dimension + 4}.")
        if not 1 <= parameter_count <= dimension:
            raise ValueError(
                f"Np={parameter_count} must lie between 1 and Nd={dimension}."
            )
        difference = mock_count - dimension
        correction = (difference - 2.0) / ((difference - 1.0) * (difference - 4.0))
        product = correction * (dimension - parameter_count)
        exponent = parameter_count + 2.0 + (mock_count - 1.0 + product) / (1.0 + product)
        if exponent <= dimension:
            raise ValueError(
                f"Percival m={exponent:.8g} must exceed Nd={dimension} for a "
                "normalized density; use an explicitly chosen compression, "
                "more independent mocks, or a different likelihood model."
            )
        self.base = FixedGaussianLikelihood(
            predict=predict, rows=rows, covariance=covariance, parameters=parameters,
            data=data, indices=indices, compression=compression,
        )
        self.mock_count = int(mock_count)
        self.parameter_count = int(parameter_count)
        self.dimension = int(dimension)
        self.exponent = float(exponent)
        self.assumptions = assumptions
        log_determinant = 2.0 * np.sum(np.log(np.diag(self.base.cholesky)))
        self.log_normalization = (
            gammaln(0.5 * exponent) - gammaln(0.5 * (exponent - dimension))
            - 0.5 * dimension * np.log(np.pi * (mock_count - 1.0))
            - 0.5 * log_determinant
        )
        self._seal_geometry()

    @property
    def rows(self):
        """Return the labels of the final measured statistic."""
        return self.base.rows

    @property
    def data(self):
        """Return the observed or synthetic final statistic."""
        return self.base.data

    @property
    def covariance(self):
        """Return the selected and optionally compressed sample covariance."""
        return self.base.covariance

    @property
    def indices(self):
        """Return the full-vector selection applied before optional compression."""
        return self.base.indices

    @property
    def parameters(self):
        """Return immutable baseline and fixed parameter values."""
        return self.base.parameters

    def mean(self, parameters):
        """Evaluate the same selected statistic as the Gaussian likelihood."""
        return self.base.mean(parameters=parameters)

    def score_mean(self, mean):
        """Return normalized log density and chi squared without a prior."""
        _, chi_squared = self.base.score_mean(mean=mean)
        logarithm = np.log1p(chi_squared / (self.mock_count - 1.0))
        score = self.log_normalization - 0.5 * self.exponent * logarithm
        return float(score), chi_squared

    def __call__(self, parameters):
        """Evaluate a physical mean and return its normalized log density."""
        prediction = self.mean(parameters=parameters)
        score, _ = self.score_mean(mean=prediction)
        return score
