"""Fixture for testing native joint statistics with small analytic means."""
import numpy as np
from scipy import stats
DISCRETE_PARAMETERS = {"power_mode", "ct_mode", "suppress_only", "num_massive_neutrinos"}
import cosmolike_des_3pcf_interface as ci

class NativeExample:
    def initialize_joint(
        self, predict, rows, covariance, parameters, data=None, indices=None,
        compression=None, mock_count=0, parameter_count=0, assumptions=None,
    ):
        """Install the joint measurement in the C++ singleton.

        Rows identify the full xi+/xi-/Map3 vector. The native interface
        selects both covariance axes, applies any fixed compression and
        factors the resulting covariance once. Reinitializing replaces the
        active dataset, as in the other Cocoa projects.
        """
        self.predict = predict
        self.parameters = dict(parameters)
        self.full_size = rows.values.size
        for name, value in self.parameters.items():
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"Baseline {name} must be a finite scalar.")
        if indices is None:
            indices = np.arange(self.full_size, dtype=np.int32)
        selected = rows.select(indices=indices)
        self.indices = np.array(indices, dtype=np.int32, copy=True)
        self.rows = selected
        transform = np.empty(shape=(0, 0), dtype=np.float64)
        if compression is not None:
            compression.check_labels(labels=selected.labels)
            transform = np.ascontiguousarray(compression.matrix, dtype=np.float64)
        if mock_count:
            if assumptions is None or assumptions.get("covariance_kind") != "sample":
                raise ValueError("Finite-simulation likelihood requires raw sample S.")
            for name in ("independent_mocks", "independent_data", "gaussian_mocks"):
                if assumptions.get(name) is not True:
                    raise ValueError(f"Finite-simulation assumption {name} must be True.")
            if compression is not None:
                if (assumptions.get("compression_independent") is not True
                        or compression.metadata.get("covariance_training_independent") is False):
                    raise ValueError("Compression must be independent of sample S.")
        self.data_kind = "observed"
        if data is None:
            self.data_kind = "synthetic_fiducial"
            data = predict(dict(parameters))
        values = np.ascontiguousarray(data, dtype=np.float64)
        matrix = np.ascontiguousarray(covariance, dtype=np.float64)
        if values.shape != (self.full_size,) or not np.all(np.isfinite(values)):
            raise ValueError("Likelihood data must have the finite full vector shape.")
        ci.init_joint_data(
            data=values, covariance=matrix, indices=self.indices,
            compression=transform, mock_count=mock_count,
            parameter_count=parameter_count,
        )
        self.mock_count = mock_count
        self.parameter_count = parameter_count
        self.assumptions = assumptions
        self.dimension = self.data.size
        self.exponent = ci.get_joint_exponent()
        if compression is not None:
            from cobaya.likelihoods.des_3pcf.compression import CompressedRows

            self.rows = CompressedRows(labels=compression.output_labels, values=self.data)

    @property
    def data(self):
        """Copy the active selected measurement from C++."""
        return ci.get_joint_data()

    @property
    def covariance(self):
        """Copy the active selected covariance, including cross terms."""
        return ci.get_joint_covariance()

    @property
    def log_normalization(self):
        return ci.get_joint_log_normalization()

    def mean(self, parameters):
        """Evaluate the full prediction, then apply the native selection."""
        point = dict(self.parameters)
        for name, value in parameters.items():
            if name not in point:
                raise ValueError(f"Unknown sampled parameter {name}; check the YAML.")
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"Sampled {name} must be a finite scalar.")
            point[name] = value
        prediction = np.ascontiguousarray(self.predict(point), dtype=np.float64)
        return ci.select_joint_data_vector(prediction=prediction)

    def score_mean(self, mean):
        """Evaluate a selected prediction using the covariance held by C++."""
        return ci.score_joint_mean(mean=np.ascontiguousarray(mean, dtype=np.float64))

    def __call__(self, parameters):
        """Return normalized log likelihood; Cobaya supplies the priors."""
        mean = self.mean(parameters=parameters)
        score, _ = self.score_mean(mean=mean)
        return score

def make_likelihood(**arguments):
    likelihood = NativeExample()
    likelihood.initialize_joint(**arguments)
    return likelihood


def fields(mapping, allowed, required, name):
    """Reject missing or misspelled configuration fields before preparation."""
    if not isinstance(mapping, dict):
        raise ValueError(f"{name} must be a YAML mapping, received {type(mapping)}.")
    unknown = set(mapping) - set(allowed)
    missing = set(required) - set(mapping)
    if unknown or missing:
        raise ValueError(
            f"{name}: unknown fields {sorted(unknown)}, missing {sorted(missing)}; "
            "use the fields in EXAMPLE_NAUTILUS3.yaml."
        )

def prior_distributions(priors):
    """Construct normalized independent densities in YAML parameter order.

    Uniform priors have explicit minimum/maximum. A normal prior has a mean
    and sigma; optional minimum/maximum choose a truncated, renormalized
    Gaussian instead. scipy.stats uses standardized limits for truncnorm,
    so physical limits are first converted to (limit-mean)/sigma.

    Arguments:
      priors = ordered mapping from parameter names to prior specifications.
    Returns:
      dictionary of SciPy distribution objects, exposing pdf, logpdf and ppf.
    Raises:
      ValueError for a malformed prior or a discrete parameter name.
    """
    if not isinstance(priors, dict) or not priors:
        raise ValueError("priors must name at least one sampled parameter.")
    distributions = {}
    for name, specification in priors.items():
        if name in DISCRETE_PARAMETERS:
            raise ValueError(f"{name} is a discrete model choice; keep it fixed.")
        fields(
            mapping=specification,
            allowed={"distribution", "minimum", "maximum", "mean", "sigma", "latex"},
            required={"distribution"}, name=f"prior {name}",
        )
        for key in ("minimum", "maximum", "mean", "sigma"):
            if key in specification:
                value = specification[key]
                if isinstance(value, bool) or not np.isfinite(value):
                    raise ValueError(f"Prior {name}.{key}={value} must be finite.")
        distribution = specification["distribution"]
        lower = specification.get("minimum")
        upper = specification.get("maximum")
        if (lower is None) != (upper is None):
            raise ValueError(f"Prior {name} must declare both limits or neither.")
        if lower is not None and lower >= upper:
            raise ValueError(f"Prior {name} requires minimum < maximum.")
        if distribution == "uniform":
            if lower is None or "mean" in specification or "sigma" in specification:
                raise ValueError(f"Uniform prior {name} requires only two limits.")
            distributions[name] = stats.uniform(loc=lower, scale=upper - lower)
        elif distribution == "normal":
            if "mean" not in specification or specification.get("sigma", 0) <= 0:
                raise ValueError(f"Normal prior {name} requires mean and sigma > 0.")
            mean = specification["mean"]
            sigma = specification["sigma"]
            if lower is None:
                distributions[name] = stats.norm(loc=mean, scale=sigma)
            else:
                distributions[name] = stats.truncnorm(
                    a=(lower - mean) / sigma, b=(upper - mean) / sigma,
                    loc=mean, scale=sigma,
                )
        else:
            raise ValueError(f"Prior {name} must choose uniform or normal.")
    return distributions

def check_likelihood(likelihood, distributions, specification):
    """Evaluate fiducial, displaced and restored points before sampling.

    Returns:
      diagnostics of a changed physical mean and bitwise restored mean.
    Raises:
      ValueError if the displacement changes no retained observable or if
      restored state differs. An unavailable model raises its own diagnostic.
    """
    parameters = dict(likelihood.parameters)
    first = likelihood.mean(parameters=parameters)
    name = specification["check"]["parameter"]
    displaced = dict(parameters)
    displaced[name] += specification["check"]["delta"]
    second = likelihood.mean(parameters=displaced)
    restored = likelihood.mean(parameters=parameters)
    if not np.array_equal(first, restored):
        raise ValueError(
            "Restored fiducial differs bitwise after the check displacement."
        )
    if np.array_equal(first, second):
        raise ValueError(
            "Check displacement changes no selected mean; check model reach."
        )
    first_score, first_chi2 = likelihood.score_mean(mean=first)
    second_score, second_chi2 = likelihood.score_mean(mean=second)
    log_prior = 0.0
    for parameter, distribution in distributions.items():
        log_prior += float(distribution.logpdf(parameters[parameter]))
    return {
        "parameter": name,
        "delta": specification["check"]["delta"],
        "fiducial_log_prior": log_prior,
        "fiducial_log_likelihood": first_score,
        "fiducial_chi_squared": first_chi2,
        "displaced_log_likelihood": second_score,
        "displaced_chi_squared": second_chi2,
        "maximum_absolute_mean_change": float(np.max(np.abs(second - first))),
        "restored_bitwise": True,
    }
