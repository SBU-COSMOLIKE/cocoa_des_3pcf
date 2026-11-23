"""Prepare the physical inputs for the operator-decomposition notebook.

The operator decomposition splits one aperture-statistic prediction into
the separate physical operators that add up to it (the matter term and
each intrinsic-alignment term), so the notebook can plot each
contribution alone. The configuration names a source model, cosmology,
triangle estimator and numerical grids; selecting a few source triplets
changes only which outputs are requested. The complete operator
calculation uses the same C routines as the ordinary aperture
prediction, with explicit diagnostic selectors.
"""

from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT.parents[1] / "external_modules/code/cosmolike_core"))

import run_3pcf as driver


def aperture_components(configuration, triplets, velocity_shear=0.0):
    """Compute selected Y3 component curves with recorded physical assumptions.

    Arguments:
      configuration = project YAML/JSON path with prepared physical tables.
      triplets = requested zero-based source triples, shape [ntriplet,3].
      velocity_shear = the velocity-shear amplitude Ct in units of the
        linear tidal-alignment coefficient C1 (the convention of fastnc,
        the external three-point natural-components code this project
        compares against); zero gives TATT without the additional
        velocity-shear operator.
    Returns:
      dictionary containing components, the selected SurveyRows (the
      labeled row table: one label, row kind, bin indices and angle per
      data-vector entry), physical-table metadata, IA parameters and the
      fully resolved run configuration.
    Raises:
      ValueError for missing triples, a nonfinite velocity_shear, or a
      configuration whose aperture estimator is not finite_sas (the
      estimator built from finite shear aperture statistics; the
      pipeline's other choice is full_filter).
    Side effects:
      Loads tables, prepares C/FFT objects and calculates fresh components.
      No saved prediction is silently reused or replaced.
    """
    # ----------------------------------------------------------------------
    # Configuration and validation. load_configuration returns a
    # three-value tuple, in this order: the validated configuration
    # dictionary, the resolved path (rebinding the name path) and the
    # SHA-256 digest of the file, which this diagnostic does not record.
    # The triple array must be integer [ntriplet, 3].
    # ----------------------------------------------------------------------
    path = Path(configuration).resolve()
    config, path, digest = driver.load_configuration(path=path, command="evaluate")
    if config["mean"]["aperture_estimator"] != "finite_sas":
        raise ValueError("This notebook wrapper requires a finite_sas configuration.")
    requested = np.asarray(triplets)
    if (requested.ndim != 2 or requested.shape[1] != 3
            or requested.dtype.kind not in "iu" or not np.isfinite(velocity_shear)):
        raise ValueError("Supply integer source triples and a finite velocity_shear.")

    # ----------------------------------------------------------------------
    # Row selection: keep exactly the aperture (map3) rows whose bin
    # triple matches a requested one; a triple with no matching row is an
    # error, not an empty result.
    # ----------------------------------------------------------------------
    rows = driver.rows_for_run(config=config, path=path)
    selected = np.zeros(shape=rows.values.size, dtype=bool)
    for triplet in requested:
        match = (rows.kind == "map3") & np.all(rows.bins == triplet, axis=1)
        if not np.any(match):
            raise ValueError(f"No aperture rows have the requested triple {triplet}.")
        selected |= match
    rows = rows.select(indices=np.flatnonzero(selected))

    # ----------------------------------------------------------------------
    # Physical preparation: the compiled extension, the provider with its
    # physical tables, and the prediction object owning the aperture
    # engine for the selected rows.
    # ----------------------------------------------------------------------
    compiled = driver.compiled_extension(config=config)
    provider, metadata = driver.provider_for_run(
        compiled=compiled, config=config, path=path,
    )
    prediction = driver.prediction_for_run(
        compiled=compiled, config=config, path=path, rows=rows, provider=provider,
    )
    model, tables, resolved_metadata = provider.get(parameters=config["parameters"])

    # ----------------------------------------------------------------------
    # Components. dict(config["ia"], ct=..., ct_mode=0) copies the
    # configured IA dictionary and overrides two keys in the copy: ct, the
    # requested velocity-shear amplitude, and ct_mode = 0, which selects
    # the physical convention Ct = C1 * ct (mode 1 would read ct as Ct
    # directly). aperture_components then returns one curve per operator.
    # ----------------------------------------------------------------------
    ia = dict(config["ia"], ct=float(velocity_shear), ct_mode=0)
    components = prediction.map_engine.aperture_components(
        model=model,
        ia=ia,
    )
    return {
        "components": components,
        "rows": rows,
        "physical_metadata": resolved_metadata,
        "ia": ia,
        "configuration": config,
    }
