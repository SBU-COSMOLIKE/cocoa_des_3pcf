"""Resolve numerical resolutions from unchanged baseline settings.

Sampling and integration have different jobs. More interpolation points
reduce the error made when reading a prepared function; more quadrature
nodes reduce the error in its integral. Neither change extends the physical
range, changes a survey bin, or adds a missing physical contribution.

An Accuracy object is immutable. Applying it twice to the same baseline
therefore produces the same counts, rather than multiplying a previous
boost again. A notebook can replace this object between evaluations and
the physical provider includes its exact values in its model key.
"""

from dataclasses import asdict, dataclass, fields
from copy import deepcopy

import numpy as np


@dataclass(frozen=True)
class Accuracy:
    """Keep global and individual numerical refinements together.

    Arguments:
      accuracy_boost = multiplier for prepared sampling grids, default 1.
      integration_accuracy_boost = multiplier for fixed integration counts,
          default 1. This is a multiplicative count, not an adaptive tolerance.
      Remaining *_boost values = additional multipliers for just that stage.
          chi_boost refines the supplied LOS intervals; kernel_boost changes
          source-window sampling; fit_z_boost changes matter-fit anchors.
          source_boost refines an explicitly independent source_nz grid,
          leaving the selected reconstruction and CAMB redshifts unchanged.
          power_table_boost changes the C model's dense reader ratio.
          bispectrum_boost refines the four coarse axes of an explicitly
          supplied baseline bispectrum; its dense-reader ratio stays fixed.
          variance_mass/variance_a/variance_radius/variance_inverse boosts
          refine shared smoothing tables from saved unboosted counts.
          triangle/multipole/fft/profile boosts change their named tables.
          angular_boost changes the bispectrum angular integration nodes.
          aperture/xi/covariance boosts change the named fixed quadratures.
    Raises:
      ValueError for a nonfinite multiplier or a multiplier below one.
    """

    # Unit factors preserve the declared baseline exactly. Separate fields
    # permit a one-dimensional convergence test before a global refinement.
    accuracy_boost: float = 1.0
    integration_accuracy_boost: float = 1.0
    chi_boost: float = 1.0
    kernel_boost: float = 1.0
    source_boost: float = 1.0
    power_table_boost: float = 1.0
    bispectrum_boost: float = 1.0
    feedback_table_boost: float = 1.0
    feedback_z_boost: float = 1.0
    feedback_k_boost: float = 1.0
    feedback_u_boost: float = 1.0
    feedback_v_boost: float = 1.0
    fit_z_boost: float = 1.0
    variance_mass_boost: float = 1.0
    variance_a_boost: float = 1.0
    variance_radius_boost: float = 1.0
    variance_inverse_boost: float = 1.0
    triangle_boost: float = 1.0
    multipole_boost: float = 1.0
    angular_boost: float = 1.0
    fft_boost: float = 1.0
    profile_boost: float = 1.0
    profile_upsample_boost: float = 1.0
    aperture_ell_boost: float = 1.0
    aperture_phi_boost: float = 1.0
    xi_ell_boost: float = 1.0
    xi_theta_boost: float = 1.0
    covariance_ell_boost: float = 1.0
    covariance_window_boost: float = 1.0
    covariance_phi_boost: float = 1.0
    covariance_radius_boost: float = 1.0
    covariance_k_boost: float = 1.0
    covariance_mass_boost: float = 1.0

    def __post_init__(self):
        """Validate every multiplier before any model or FFT allocation.

        Returns:
          nothing; dataclass construction stops on the named bad value.
        """
        for field in fields(self):
            value = getattr(self, field.name)
            if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                    or not np.isfinite(value) or value < 1.0):
                raise ValueError(f"{field.name}={value!r} must be finite and >=1.")

    def record(self):
        """Return all resolved multipliers, including defaults, for saved results."""
        return asdict(self)

    def count(self, baseline, stage, integration=False, endpoints=False):
        """Resolve a count without changing its baseline or a previous result.

        Arguments:
          baseline = positive integer count supplied by the calculation.
          stage = name of this object's individual multiplier.
          integration = use the integration global factor instead of sampling.
          endpoints = refine N-1 intervals, keeping the two endpoints.
        Returns:
          integer ceil(base*factor), or ceil((base-1)*factor)+1.
        Raises:
          ValueError for invalid counts or a result outside the C int domain.
        """
        if (isinstance(baseline, (bool, np.bool_)) or not np.isfinite(baseline)
                or int(baseline) != baseline or baseline < 1):
            raise ValueError(f"Baseline count {baseline!r} must be a positive integer.")
        global_factor = self.accuracy_boost
        if integration:
            global_factor = self.integration_accuracy_boost
        factor = global_factor * getattr(self, stage)
        offset = int(endpoints)
        resolved = (baseline - offset) * factor + offset
        if not np.isfinite(resolved) or resolved > np.iinfo(np.int32).max:
            raise ValueError("Resolved count exceeds the C int range; reduce boosts.")
        return int(np.ceil(resolved))

    def los_nodes(self, baseline):
        """Subdivide each LOS redshift interval without moving its old nodes.

        The C model computes chi(z) on the new nodes and trapezoid weights
        in chi. A dyadic subdivision preserves every original redshift,
        including nonuniform inputs. It leaves the integration limits fixed.

        Arguments:
          baseline = increasing positive one-dimensional redshift nodes.
        Returns:
          owned float64 nodes; total count is (N-1)*2^ceil(log2(factor))+1,
          with factor=accuracy_boost*integration_accuracy_boost*chi_boost.
        Raises:
          ValueError for invalid nodes or a requested subdivision above 1024.
        """
        nodes = np.asarray(baseline, dtype=np.float64)
        if (nodes.ndim != 1 or nodes.size < 2 or nodes[0] <= 0.0
                or not np.all(np.isfinite(nodes)) or np.any(np.diff(nodes) <= 0.0)):
            raise ValueError("LOS baseline requires increasing finite positive nodes.")
        factor = self.accuracy_boost * self.integration_accuracy_boost * self.chi_boost
        if not np.isfinite(factor) or factor > 1024:
            raise ValueError("LOS subdivision exceeds 1024; reduce the requested boosts.")
        subdivision = 2**int(np.ceil(np.log2(factor)))
        result = np.empty((nodes.size - 1) * subdivision + 1, dtype=np.float64)
        fractions = np.arange(subdivision, dtype=np.float64) / subdivision
        for index in range(nodes.size - 1):
            start = index * subdivision
            result[start:start + subdivision] = (
                nodes[index] + fractions * (nodes[index + 1] - nodes[index])
            )
        result[-1] = nodes[-1]
        return result

    def model_settings(self, kernel_nz=101, table_boost=8, fit_nz=33):
        """Resolve the C physical model's interpolation counts and reader ratio.

        External k/z power tables remain unchanged: interpolation refinement
        cannot create new Boltzmann information. The dense-reader ratio also
        refines the interpolated matter-fit table, as in the C constructor.
        Returns:
          keyword dictionary accepted by either PhysicsModel binding.
        """
        return {
            "kernel_nz": self.count(kernel_nz, stage="kernel_boost", endpoints=True),
            "table_boost": self.count(table_boost, stage="power_table_boost"),
            "fit_nz": self.count(fit_nz, stage="fit_z_boost", endpoints=True),
        }

    def variance_settings(self, baseline):
        """Refine shared variance tables from saved unboosted node counts.

        The scale-factor multiplier refines both window families. Radius
        and inverse multipliers affect only Gaussian smoothing; mass
        affects the top-hat table. Nested interval doubling follows the
        ordinary core's Gaussian accuracy rule and is capped at sixteen.
        FFTLog integrates its Fourier modes analytically, so the fixed
        quadrature integration_accuracy_boost does not enter these counts.
        More input k/z samples remain a separate producer-grid refinement.
        """
        stages = {
            "mass_nodes": "variance_mass_boost",
            "scale_factor_nodes": "variance_a_boost",
            "gaussian_radius_nodes": "variance_radius_boost",
            "gaussian_scale_factor_nodes": "variance_a_boost",
            "gaussian_inverse_nodes": "variance_inverse_boost",
        }
        if set(baseline) != set(stages):
            raise ValueError("Variance baseline requires all five core sampling counts.")
        result = {}
        for name, stage in stages.items():
            count = baseline[name]
            if (isinstance(count, (bool, np.bool_)) or not np.isfinite(count)
                    or int(count) != count or count < 2):
                raise ValueError(f"Variance baseline {name}={count!r} must be an integer >=2.")
            factor = self.accuracy_boost * getattr(self, stage)
            if not np.isfinite(factor) or factor > 16.0:
                raise ValueError(f"Combined variance boost for {name} must not exceed 16.")
            multiplier = 2**int(np.ceil(np.log2(factor)))
            resolved = (int(count) - 1) * multiplier + 1
            if resolved > np.iinfo(np.int32).max:
                raise ValueError(f"Resolved variance count {name} exceeds the C int range.")
            result[name] = resolved
        return result

    def projection_settings(self, baseline):
        """Resolve triangle, multipole and angular nodes from the C defaults.

        The bounds, angular truncation lmax and interpolation density ratios
        keep their explicit values. Both production and upstream-reference
        coordinate choices receive their corresponding refined node counts.
        Arguments:
          baseline = user projection settings; omitted counts use C defaults.
        Returns:
          independent settings dictionary for model.projection_grid(s).
        """
        result = deepcopy(baseline)
        triangle = {"nr": 35, "nu": 35, "nv": 25, "np": 35,
                    "npu_log": 8, "npu_linear": 22, "nshape": 80}
        multipole = {"nell": 100, "npsi_log": 50, "npsi_linear": 80, "npsi": 80}
        for name, default in triangle.items():
            result[name] = self.count(result.get(name, default),
                                      stage="triangle_boost", endpoints=True)
        for name, default in multipole.items():
            # A zero-length logarithmic psi segment is a supported reference
            # choice; multiplying its count must not invent a new segment.
            if result.get(name, default) == 0 and name == "npsi_log":
                continue
            result[name] = self.count(result.get(name, default),
                                      stage="multipole_boost", endpoints=True)
        for name, default in (("nmu_log", 30), ("nmu_linear", 50)):
            result[name] = self.count(result.get(name, default),
                                      stage="angular_boost", integration=True)
        return result

    def quadrature_settings(self, baseline, stage):
        """Copy a fixed quadrature and refine only its count, preserving bounds."""
        result = deepcopy(baseline)
        result["count"] = self.count(result["count"], stage=stage, integration=True)
        return result


def accuracy_settings(value=None):
    """Accept an Accuracy object or a dictionary, rejecting unknown control names.

    Arguments:
      value = None for unit factors, an Accuracy object, or its keyword dictionary.
    Returns:
      immutable Accuracy object, safe to include in an exact model-cache key.
    Raises:
      ValueError for an unsupported object or an unknown dictionary field.
    """
    if value is None:
        return Accuracy()
    if isinstance(value, Accuracy):
        return value
    if not isinstance(value, dict):
        raise ValueError("Accuracy requires an Accuracy object or a control dictionary.")
    allowed = set()
    for field in fields(Accuracy):
        allowed.add(field.name)
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown accuracy controls: {sorted(unknown)}.")
    return Accuracy(**value)
