"""Read saved joint derivatives as parameter responses and fitted residuals.

These notebook operations use arrays already checked by FisherWorkflow.
They never request a physical model, change a prior or regenerate a mean.
"""

import importlib

import numpy as np


def derivative_comparison(result, target, mimic, target_scale, mimic_scale):
    """Select named h/2 columns and compare them with the full saved covariance.

    Arguments:
      result = FisherWorkflow.ensure_result arrays in their recorded row order.
      target, mimic = distinct saved parameter names.
      target_scale, mimic_scale = positive parameter intervals for display;
          they do not impose priors or select finite-difference steps.
    Returns:
      Shared template comparison, with parameter names and scales recorded.
      Other cosmological and nuisance parameters are fixed in this pair fit.
    Raises:
      ValueError for missing columns, invalid scales or inconsistent row shapes.
    """
    names = result["parameter_names"].tolist()
    if target == mimic or target not in names or mimic not in names:
        raise ValueError("Select two distinct names in the saved parameter list.")
    scales = np.asarray([target_scale, mimic_scale], dtype=float)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("Display intervals must be finite positive numbers.")
    derivatives = np.asarray(result["derivatives_h2"], dtype=float)
    kinds = np.asarray(result["row_kind"])
    if (derivatives.shape != (kinds.size, len(names))
            or kinds.ndim != 1
            or not np.all(np.isin(kinds, ("xip", "xim", "map3")))):
        raise ValueError("Derivative columns and xi/Map3 rows must match saved labels.")
    target_response = target_scale * derivatives[:, names.index(target)]
    mimic_response = mimic_scale * derivatives[:, names.index(mimic)]
    shared = importlib.import_module(
        name="cosmolike_notebook_utils.3pcf.derivative_templates",
    )
    comparison = shared.compare_templates(
        target=target_response, mimic=mimic_response,
        covariance=result["fiducial_covariance"], two_point=kinds != "map3",
    )
    comparison["parameters"] = [target, mimic]
    comparison["scales"] = scales.copy()
    comparison["row_labels"] = result["row_labels"].copy()
    return comparison


def plot_derivatives(result, comparison, parameter_labels, source_bin=3):
    """Draw xi+, xi- and equal-radius Map3 for one source auto-bin.

    Arguments:
      result = the same saved Fisher arrays supplied to derivative_comparison.
      comparison = its result; coefficients were fitted to all saved rows.
      parameter_labels = two mathematical display labels naming their scales.
      source_bin = zero-based source index; the displayed title counts from one.
    Returns:
      Shared figure/axes. Only presentation selects one bin. Fits retain all
      auto- and cross-bin rows, and the covariance cross block.
    Raises:
      ValueError for an absent bin, unequal aperture radii or malformed rows.
    """
    if (isinstance(source_bin, bool) or not isinstance(source_bin, (int, np.integer))
            or source_bin < 0):
        raise ValueError("Select a nonnegative integer source-bin index.")
    nrow = result["row_kind"].size
    bins = np.asarray(result["row_bins"])
    angles = np.asarray(result["row_angles_radians"])
    if bins.shape != (nrow, 3) or angles.shape != (nrow, 3):
        raise ValueError("Saved source labels and angles must have three columns.")
    if comparison["templates"].shape != (2, nrow):
        raise ValueError("The comparison and saved result must have the same rows.")
    if not np.array_equal(comparison["row_labels"], result["row_labels"]):
        raise ValueError("The comparison and saved result have different row labels.")
    panels = []
    for kind, quantity in (
        ("xip", r"$\xi_+$"),
        ("xim", r"$\xi_-$"),
        ("map3", r"$\langle M_{\rm ap}^3\rangle$"),
    ):
        selected = result["row_kind"] == kind
        selected &= bins[:, 0] == source_bin
        selected &= bins[:, 1] == source_bin
        coordinate_column = 2
        coordinate_label = r"$\theta$ [arcmin]"
        if kind == "map3":
            selected &= bins[:, 2] == source_bin
            coordinate_column = 0
            coordinate_label = r"$R$ [arcmin]"
            if not np.all(angles[selected] == angles[selected, :1]):
                raise ValueError("This figure requires equal aperture radii.")
        indices = np.flatnonzero(selected)
        if indices.size < 2:
            raise ValueError(f"Source bin {source_bin + 1} lacks {kind} samples.")
        coordinate = angles[indices, coordinate_column] * (180.0 * 60.0 / np.pi)
        order = np.argsort(coordinate)
        indices = indices[order]
        panels.append({
            "coordinate": coordinate[order],
            "templates": comparison["templates"][:, indices],
            "residuals": np.array([
                comparison["two_point"]["residual"][indices],
                comparison["joint"]["residual"][indices],
            ]),
            "title": f"{quantity}, source bin {source_bin + 1}",
            "coordinate_label": coordinate_label,
        })
    shared = importlib.import_module(
        name="cosmolike_notebook_utils.3pcf.derivative_templates",
    )
    return shared.plot_derivative_templates(
        panels=panels, parameter_labels=parameter_labels, show=None,
    )
