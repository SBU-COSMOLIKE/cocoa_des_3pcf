"""Prepare the physical inputs for the operator-decomposition notebook.

The configuration names a source model, cosmology, triangle estimator and
numerical grids. Selecting a few source triplets changes only which outputs
are requested. The complete operator calculation uses the same C routines
as the ordinary aperture prediction, with explicit diagnostic selectors.
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
      velocity_shear = Ct/C1 in the implemented fastnc convention; zero
        gives TATT without the additional velocity-shear operator.
    Returns:
      dictionary containing components, selected SurveyRows, physical-table
      metadata, IA parameters and fully resolved run configuration.
    Raises:
      ValueError for missing triples, nonfinite Ct/C1 or a nonfinite-SAS model.
    Side effects:
      Loads tables, prepares C/FFT objects and calculates fresh components.
      No saved prediction is silently reused or replaced.
    """
    path = Path(configuration).resolve()
    config, path, digest = driver.load_configuration(path=path, command="evaluate")
    if config["mean"]["aperture_estimator"] != "finite_sas":
        raise ValueError("This notebook wrapper requires a finite_sas configuration.")
    requested = np.asarray(triplets)
    if (requested.ndim != 2 or requested.shape[1] != 3
            or requested.dtype.kind not in "iu" or not np.isfinite(velocity_shear)):
        raise ValueError("Supply integer source triples and a finite velocity_shear.")
    rows = driver.rows_for_run(config=config, path=path)
    selected = np.zeros(shape=rows.values.size, dtype=bool)
    for triplet in requested:
        match = (rows.kind == "map3") & np.all(rows.bins == triplet, axis=1)
        if not np.any(match):
            raise ValueError(f"No aperture rows have the requested triple {triplet}.")
        selected |= match
    rows = rows.select(indices=np.flatnonzero(selected))
    compiled = driver.compiled_extension(config=config)
    provider, metadata = driver.provider_for_run(
        compiled=compiled, config=config, path=path,
    )
    prediction = driver.prediction_for_run(
        compiled=compiled, config=config, path=path, rows=rows, provider=provider,
    )
    model, tables, resolved_metadata = provider.get(parameters=config["parameters"])
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
