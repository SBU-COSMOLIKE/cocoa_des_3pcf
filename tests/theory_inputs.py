"""Explicit small theory tables for projection integration tests.

Counts here exercise ownership and projection, not DES sampling accuracy.
The equation tests separately compare BiHalofit with frozen native and
upstream results. This helper never replaces a compiled function.
"""

from des3pcf.accuracy import accuracy_settings
from des3pcf.bihalofit import bispectrum_axes, prepare_reference_bispectrum
from des3pcf.mean import physics_model
from des3pcf.variance import reference_matter_fraction


def prepared_physics_model(compiled, tables, **settings):
    """Prepare theory inputs explicitly before invoking the native factory."""
    controls = accuracy_settings(value=settings.get("accuracy"))
    resolved = controls.model_settings(
        kernel_nz=settings.get("kernel_nz", 101),
        table_boost=settings.get("table_boost", 8),
        fit_nz=settings.get("fit_nz", 33),
    )
    tables = dict(tables)
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
                         baseline=baseline, **settings)
