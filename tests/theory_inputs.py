"""Explicit small theory tables for projection integration tests.

Counts here exercise ownership and projection, not DES sampling accuracy.
The equation tests separately compare BiHalofit with frozen native and
upstream results. This helper never replaces a compiled function.
"""


from theory.bihalofit.model import bispectrum_axes, prepare_reference_bispectrum
from des3pcf.physical_provider import validate_tables
from theory.bihalofit.variance import reference_matter_fraction


import cosmolike_des_3pcf_interface as ci


def prepared_physics_model(compiled, tables, **settings):
    """Prepare theory inputs explicitly before invoking the native factory."""
    controls = ci.init_3pcf_accuracy(value=settings.get("accuracy"))
    resolved = ci.accuracy_model_settings(accuracy=controls,
        kernel_nz=settings.get("kernel_nz", 101),
        table_boost=settings.get("table_boost", 8),
        fit_nz=settings.get("fit_nz", 33),
    )
    tables = validate_tables(tables=tables)
    if "omega_m_z" not in tables:
        tables["omega_m_z"] = reference_matter_fraction(
            tables=tables, redshift=tables["z"],
        )
    axes = bispectrum_axes(zmax=tables["z"][-1], counts=(4, 6, 5, 4))
    prepared, baseline = prepare_reference_bispectrum(
        tables=tables, axes=axes, fit_nz=resolved["fit_nz"],
        table_boost=resolved["table_boost"],
        source_readers=settings.get("source_readers", False),
    )
    return physics_model(compiled=compiled, tables=prepared,
                         baseline=baseline["native"], **settings)


def physics_model(
    compiled, tables, kernel_nz=101, source_readers=False, table_boost=8, fit_nz=33,
    accuracy=None, adapter="production", feedback=None, pixel_window=None,
    baseline=None, variance=None,
):
    """Construct the owned C model after validating its physical input arrays.

    Arguments:
      compiled = this project's imported extension module.
      tables = power, background and source arrays in h-based units.
      kernel_nz = source-window interpolation node count used by fastnc's baseline.
      source_readers = explicit upstream spline readers for reference comparisons.
      table_boost = dense uniform-table refinement for production linear readers.
      fit_nz = legacy native keyword; fit sampling belongs to the theory producer.
      accuracy = dictionary of global and stage-specific multipliers.
      adapter = production or notebook; both own the resolved input copies.
      feedback = optional immutable FeedbackPair with paired total-matter
          responses; the IA dictionary selects its common amplitude.
      pixel_window = optional W(ell) at consecutive integers starting at zero.
          Three-point angular observables use one W per leg. Raw projected
          bispectra and ordinary xi retain their declared unsmoothed convention.
      baseline = required dictionary of bispectrum arrays and interpolation settings.
          It also needs tables['omega_m_z'] from the same background provider.
          This factory does not regenerate B when C sampling changes; use the
          physical provider to refine theory and projection together.
      variance = retained compatibility keyword; smoothing inputs belong to
          the theory producer and are not passed into the projection model.
    Returns:
      production.PhysicsModel, owning copies of its prepared physical state.
    Raises:
      ValueError for invalid physical table shapes or domains.
    """
    checked = dict(tables)
    if checked["k"].size < 4 or checked["z"].size < 4:
        raise ValueError(
            "Physical-model spline preparation requires at least four k/z nodes."
        )
    if checked.get("source_z", checked["z"]).size < 4:
        raise ValueError(
            "Physical-model source preparation requires at least four nodes."
        )
    if int(kernel_nz) != kernel_nz or kernel_nz < 4:
        raise ValueError("kernel_nz must be an integer of at least four.")
    if abs(checked["omega_m"] + checked["omega_de"] - 1.0) > 1.0e-8:
        raise ValueError(
            "The projected physical model requires a spatially flat background."
        )
    controls = compiled.init_3pcf_accuracy(value=accuracy)
    settings = compiled.accuracy_model_settings(accuracy=controls,
        kernel_nz=kernel_nz, table_boost=table_boost, fit_nz=fit_nz,
    )
    if adapter not in ("production", "notebook"):
        raise ValueError("Physical adapter must be production or notebook.")
    if baseline is None:
        raise ValueError(
            "A theory-supplied baseline bispectrum is required. Prepare it with "
            "a matter-bispectrum theory producer before constructing the C model."
        )
    response = {}
    if feedback is not None:
        response = dict(feedback.native)
        response["boost"] = compiled.accuracy_count(
            accuracy=controls,
            baseline=feedback.native["boost"],
            stage="feedback_table_boost",
        )
    if pixel_window is not None:
        from des3pcf.pixel_window import checked_window

        pixel_window = checked_window(values=pixel_window)
    binding = getattr(compiled, adapter)
    return binding.PhysicsModelWithBispectrum(
        k=checked["k"],
        p_linear0=checked["p_linear0"],
        z=checked["z"],
        p_nonlinear=checked["p_nonlinear"],
        chi=checked["chi"],
        dz_dchi=checked["dz_dchi"],
        growth=checked["growth"],
        sources=checked["sources"],
        los_z=compiled.accuracy_los_nodes(
            accuracy=controls,
            baseline=checked["los_z"],
        ),
        omega_m=checked["omega_m"],
        sigma8=checked["sigma8"],
        source_z=checked.get("source_z"),
        omega_m_z=checked["omega_m_z"],
        baseline=baseline,
        kernel_nz=settings["kernel_nz"],
        table_boost=settings["table_boost"],
        fit_nz=settings["fit_nz"],
        source_readers=source_readers,
        feedback=response,
        pixel_window=pixel_window,
    )
