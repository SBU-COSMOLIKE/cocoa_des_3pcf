"""Describe the physical covariance calculation in survey units.

The C covariance model adds Gaussian, bispectrum-squared, power times
one-halo trispectrum, power times bispectrum, connected one-halo, and
selected halo-abundance response terms. Each term retains its own output
matrix. Their sum is a chosen analytic model whose integration ranges
and halo prescription must be tested, not an exact catalogue covariance.

The current physical estimator consists of full aperture filters and
xi± angular-bin averages over a common survey-center window. A finite
triangle selection in the mean must therefore be compared with that
full-filter definition before this covariance is used for a forecast.
The model explicitly excludes nonlinear-IA connected covariance and
arbitrary catalogue masks. Simulation corrections can address measured
residuals only after these differences have been quantified.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from .accuracy import accuracy_settings


def quadrature(lower, upper, count, logarithmic=False):
    """Construct Gauss-Legendre nodes with weights in the original variable.

    For a logarithmic rule x=exp(t), dx=x dt. Including x in the returned
    weight means every C integral receives d(ell), dk, dM or dr exactly
    once; a caller must not multiply by another log-Jacobian.

    Arguments:
      lower, upper = finite ordered integration boundaries.
      count = integer number of nodes, at least 2.
      logarithmic = integrate in log(x), requiring lower>0.
    Returns:
      increasing float64 nodes[count] and positive dx weights[count].
    Raises:
      ValueError for nonfinite/reversed bounds, bad count or nonpositive log range.
    """
    if (not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper
            or int(count) != count or count < 2):
        raise ValueError("Quadrature requires finite lower<upper and integer count>=2.")
    if logarithmic and lower <= 0.0:
        raise ValueError(
            "Logarithmic quadrature requires a strictly positive lower bound."
        )
    left = float(lower)
    right = float(upper)
    if logarithmic:
        left = np.log(left)
        right = np.log(right)
    nodes, weights = leggauss(deg=int(count))
    nodes = left + 0.5 * (nodes + 1.0) * (right - left)
    weights *= 0.5 * (right - left)
    if logarithmic:
        nodes = np.exp(nodes)
        weights *= nodes
    return np.ascontiguousarray(nodes), np.ascontiguousarray(weights)


def shape_noise(number_density, sigma_component):
    """Convert independent source-bin shape noise to angular power.

    The supplied dispersion is for one ellipticity component, so the
    result is N_ii=sigma_component_i^2/n_i with no extra factor of two.
    The same white noise enters E and B auto power. A catalogue overlap
    can create off-diagonal noise; supply a measured full noise matrix
    directly to covariance_inputs for that case.

    Arguments:
      number_density = positive [nbin] effective galaxies per square arcminute.
      sigma_component = positive [nbin] per-component ellipticity dispersion.
    Returns:
      diagonal [nbin,nbin] white angular power in steradians.
    Raises:
      ValueError for nonpositive, nonfinite or mismatched source properties.
    """
    density = np.asarray(number_density, dtype=np.float64)
    dispersion = np.asarray(sigma_component, dtype=np.float64)
    if (density.ndim != 1 or density.size == 0 or dispersion.shape != density.shape
            or np.any(density <= 0.0) or np.any(dispersion <= 0.0)
            or not np.all(np.isfinite(density)) or not np.all(np.isfinite(dispersion))):
        raise ValueError(
            "Noise requires positive finite density and component sigma per bin."
        )
    arcminute = np.pi / (180.0 * 60.0)
    density_per_steradian = density / arcminute**2
    return np.diag(dispersion**2 / density_per_steradian)


def covariance_inputs(rows, noise, grids, physical, theta_count, pair_weight,
                      accuracy=None):
    """Build the C covariance arrays from explicit survey and numerical choices.

    Each distinct (source,radius) defines one filtered field. Aperture rows
    contain three such fields; xi rows contain two unfiltered shear fields.
    This reuses common halo profiles and power contractions across rows.

    Arguments:
      rows = SurveyRows in the required output order.
      noise = [nbin,nbin] angular E/B shape-noise power, inserted once by C.
      grids = dictionaries for ell, window_ell, phi, radius, k and mass,
          each with lower, upper, count and optional logarithmic;
          profile_radius is an explicit increasing array beginning at zero.
      physical = explicit scalar C settings, including area in steradians,
          window, estimator/IA choices and ST/Duffy halo parameters;
          noise_mode=1 supplies exact uniform-area xi bin edges for analytic NN.
      theta_count = number of separation quadrature nodes per xi bin.
      pair_weight = 'uniform_area' for density proportional to theta, or
          explicit [nrow,theta_count] nonnegative weights on returned theta.
      accuracy = global and individual fixed-grid controls; baseline arrays
          remain unchanged. Explicit measured theta weights cannot be refined.
    Returns:
      dictionary of owned arrays/scalars accepted by physical_covariance.
    Raises:
      ValueError for a missing grid or unsupported pair-weight convention.
    """
    result = dict(physical)
    controls = accuracy_settings(value=accuracy)
    for name, weight_name in (
        ("ell", "dell"), ("phi", "dphi"), ("radius", "dr"),
        ("k", "dk"), ("mass", "dmass"), ("window_ell", "window_dell"),
    ):
        if name not in grids:
            raise ValueError(f"Covariance settings require the {name} quadrature.")
        stage = "covariance_" + name + "_boost"
        if name == "window_ell":
            stage = "covariance_window_boost"
        settings = controls.quadrature_settings(baseline=grids[name], stage=stage)
        nodes, weights = quadrature(**settings)
        result[name] = nodes
        result[weight_name] = weights
    profile = np.array(grids["profile_radius"], dtype=np.float64, copy=True)
    if (profile.ndim != 1 or profile.size < 2 or profile[0] != 0.0
            or not np.all(np.isfinite(profile)) or np.any(np.diff(profile) <= 0.0)):
        raise ValueError("Profile radius requires increasing finite nodes from zero.")
    # Insert equal subintervals between the supplied radii. Keeping every
    # original radius distinguishes refinement from a change of range.
    factor = controls.accuracy_boost * controls.profile_boost
    if not np.isfinite(factor) or factor > 1024:
        raise ValueError("Profile subdivision exceeds 1024; reduce its boosts.")
    subdivision = 2**int(np.ceil(np.log2(factor)))
    fractions = np.arange(subdivision, dtype=np.float64) / subdivision
    refined = profile[:-1, None] + np.diff(profile)[:, None] * fractions[None, :]
    result["profile_radius"] = np.append(refined.ravel(), profile[-1])
    result["profile_upsample"] = controls.count(
        result.get("profile_upsample", 8), stage="profile_upsample_boost",
    )
    result["noise"] = np.array(noise, dtype=np.float64, order="C", copy=True)
    fields = {}
    sources = []
    radii = []
    row_fields = np.zeros((rows.values.size, 3), dtype=np.int32)
    row_kind = np.empty(rows.values.size, dtype=np.int32)
    kinds = {"map3": 1, "xip": 2, "xim": 3}
    for row in range(rows.values.size):
        kind = str(rows.kind[row])
        if kind not in kinds:
            raise ValueError(f"Covariance row kind {kind!r} is unsupported.")
        row_kind[row] = kinds[kind]
        legs = 3 if kind == "map3" else 2
        for leg in range(legs):
            radius = 0.0
            if kind == "map3":
                radius = float(rows.angles[row, leg])
            source = int(rows.bins[row, leg])
            field = (source, radius)
            if field not in fields:
                fields[field] = len(sources)
                sources.append(source)
                radii.append(radius)
            row_fields[row, leg] = fields[field]
    result["field_source"] = np.ascontiguousarray(sources, dtype=np.int32)
    result["field_radius"] = np.ascontiguousarray(radii, dtype=np.float64)
    result["row_fields"] = row_fields
    result["row_kind"] = row_kind
    resolved_theta = controls.count(
        theta_count, stage="xi_theta_boost", integration=True,
    )
    if resolved_theta != theta_count and not isinstance(pair_weight, str):
        raise ValueError(
            "Refining theta nodes requires new measured pair weights; "
            "supply them at the desired baseline count or use uniform_area."
        )
    theta, weights = xi_bin_weights(
        rows=rows, count=resolved_theta, pair_weight=pair_weight,
    )
    result["theta"] = theta
    result["theta_weight"] = weights
    if result.get("noise_mode", 0) == 1:
        if not isinstance(pair_weight, str) or pair_weight != "uniform_area":
            raise ValueError(
                "Analytic xi NN requires declared uniform_area pair weights."
            )
        if result.get("large_area", 0) != 1:
            raise ValueError(
                "Analytic xi NN requires the large-area estimator approximation."
            )
        # Continuous annulus overlap fixes the white-noise delta function.
        # Point quadrature samples alone do not determine that overlap.
        edges = np.zeros((rows.values.size, 2), dtype=np.float64)
        selected = rows.kind != "map3"
        edges[selected] = rows.angles[selected, :2]
        result["theta_edges"] = edges
    return result


def xi_bin_weights(rows, count, pair_weight):
    """Define the actual normalized separation average for every xi row.

    A uniform area pair distribution has weight theta dtheta. Measured
    pair weights can differ because of a mask or source weights. An
    explicit array supplies the complete quadrature weights at the same
    nodes; integrated NPAIRS per bin alone cannot reconstruct them.

    Arguments:
      rows = SurveyRows, including any map3 rows whose theta entries are unused.
      count = integer Gauss-Legendre nodes per angular bin.
      pair_weight = 'uniform_area' or complete [nrow,count] discrete weights.
    Returns:
      theta[nrow,count] in radians and normalized dimensionless weights.
    Raises:
      ValueError for unsupported weight strings or invalid explicit arrays.
    """
    if int(count) != count or count < 2:
        raise ValueError("Xi angular-bin integration requires integer count>=2.")
    theta = np.ones((rows.values.size, int(count)), dtype=np.float64)
    weights = np.full_like(theta, fill_value=1.0 / count)
    uniform = isinstance(pair_weight, str)
    if uniform and pair_weight != "uniform_area":
        raise ValueError(
            "Pair-weight name must be 'uniform_area'; otherwise supply an array."
        )
    if not uniform:
        supplied = np.asarray(pair_weight, dtype=np.float64)
        if (supplied.shape != theta.shape or np.any(supplied < 0.0)
                or not np.all(np.isfinite(supplied))):
            raise ValueError(
                "Explicit pair weights must be finite nonnegative [nrow,count]."
            )
    for row in range(rows.values.size):
        if rows.kind[row] == "map3":
            continue
        nodes, measure = quadrature(
            lower=rows.angles[row, 0], upper=rows.angles[row, 1], count=count,
        )
        theta[row] = nodes
        if uniform:
            weights[row] = measure * nodes
        else:
            weights[row] = supplied[row]
        normalization = np.sum(weights[row])
        if normalization <= 0.0:
            raise ValueError(f"Xi row {row} has zero total pair weight.")
        weights[row] /= normalization
    return np.ascontiguousarray(theta), np.ascontiguousarray(weights)
