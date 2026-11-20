"""Check the assembled mean pipeline with physical C calculations.

The arrays describe two smooth source samples and a controlled power
spectrum. The small quadratures are adequate for checking how physical
inputs and source labels pass through the pipeline; they do not establish
DES integration accuracy. Multiplicative shear changes have an exact
polynomial answer independent of quadrature resolution.
"""

import numpy as np
import pytest

from des3pcf.physical_provider import PhysicalProvider
from des3pcf.pipeline import JointPrediction
from des3pcf.survey import forecast_rows


def toy_tables():
    """Return two source curves and positive power in the documented h units.

    Returns:
      tables and their matching cosmological parameter dictionary.
    """
    k = np.geomspace(1.0e-5, 1.0e3, 96)
    z = np.linspace(0.0, 2.0, 41)
    power = 2.0e4 * k / (1.0 + (k / 0.15)**2)**1.8
    sources = np.empty((2, z.size), dtype=np.float64)
    sources[0] = z**2 * np.exp(-(z / 0.6)**1.5)
    sources[1] = z**2 * np.exp(-(z / 0.95)**1.5)
    tables = {
        "k": k,
        "p_linear0": power,
        "z": z,
        "p_nonlinear": power[None, :] / (1.0 + z[:, None])**2,
        "chi": 3000.0 * z / (1.0 + 0.35 * z),
        "dz_dchi": (1.0 + 0.35 * z)**2 / 3000.0,
        "growth": 1.0 / (1.0 + z),
        "sources": sources,
        "los_z": np.array([0.1, 0.4, 0.8, 1.5]),
        "omega_m": 0.3,
        "omega_m_z": 0.3 * (1.0 + z)**3 / (0.3 * (1.0 + z)**3 + 0.7),
        "omega_de": 0.7,
        "w0": -1.0,
        "wa": 0.0,
        "ns": 0.965,
        "sigma8": 0.8,
    }
    parameters = {
        "omega_m": 0.3,
        "omega_b": 0.05,
        "h": 0.7,
        "w0": -1.0,
        "wa": 0.0,
        "ns": 0.965,
        "sigma8": 0.8,
    }
    return tables, parameters


def compact_bispectrum_grid(zmax=2.0):
    """Use explicit small theory axes for ownership tests, not DES accuracy."""
    from theory.bihalofit.model import bispectrum_axes

    return bispectrum_axes(zmax=zmax, counts=(4, 6, 5, 4))


@pytest.mark.parametrize("projection_mode", ("direct", "prepared"))
def test_joint_physical_mean_preserves_rows_and_calibration(compiled, projection_mode):
    """Keep all xi rows and apply one calibration factor per shear leg.

    Arguments:
      compiled = the project extension fixture, with one worker.
      projection_mode = direct reference or prepared uniform interpolation.
    Returns:
      nothing; assertions use the exact field-rescaling law.
    """
    tables, parameters = toy_tables()
    provider = PhysicalProvider(
        bispectrum_table=compact_bispectrum_grid(zmax=tables["z"][-1]),
        compiled=compiled, reference_tables=tables, reference_metadata={},
        reference_parameters=parameters, cosmology_mode="fixed_tables",
    )
    rows = forecast_rows(
        nbin=2, aperture_radii=[0.003], xi_edges=[0.001, 0.002, 0.004],
    )
    assert rows.values.size == 16  # 12 xi rows and all four source triplets.
    prediction = JointPrediction(
        compiled=compiled, rows=rows, provider=provider,
        ia_defaults={"a1": 0.2, "eta1": 0.3}, aperture_estimator="full_filter",
        aperture_settings={
            "projection_mode": projection_mode,
            "projection_batch_size": 3,
            "projection_settings": {
                "nr": 10, "nu": 12, "nv": 10, "np": 10, "nshape": 12,
            },
            "ell": {"lower": 20.0, "upper": 3000.0, "count": 6, "logarithmic": True},
            "phi": {"lower": 0.0, "upper": 2.0 * np.pi, "count": 6},
        },
        xi_settings={
            "ell": {"lower": 20.0, "upper": 3000.0, "count": 24, "logarithmic": True},
            "theta_count": 4,
            "pair_weight": "uniform_area",
        },
    )
    baseline = prediction(parameters)
    original_model = provider.model
    displaced = dict(parameters)
    displaced["m1"] = 0.04
    displaced["m2"] = -0.02
    changed = prediction(displaced)
    assert provider.model is original_model
    factors = np.ones(rows.values.size)
    for row in range(rows.values.size):
        for source in rows.bins[row]:
            if source == 0:
                factors[row] *= 1.04
            elif source == 1:
                factors[row] *= 0.98
    np.testing.assert_allclose(changed, baseline * factors, rtol=5.0e-15, atol=0.0)
    assert np.all(np.isfinite(changed))
    assert np.any(baseline != 0.0)
    # The same immutable physical model can serve a different IA point,
    # but its retained projected mean must then be recalculated. Returning
    # to the old amplitude recovers the original physical result exactly.
    displaced_ia = dict(parameters)
    displaced_ia["a1"] = 0.5
    aligned = prediction(displaced_ia)
    assert provider.model is original_model
    assert not np.array_equal(aligned, baseline)
    np.testing.assert_array_equal(prediction(parameters), baseline)
    displaced["sigma8"] = 0.81
    with pytest.raises(ValueError, match="parameter changed"):
        prediction(displaced)


@pytest.mark.parametrize("name", ("h", "omega_b", "mnu"))
def test_provider_rejects_mismatched_physical_provenance(compiled, name):
    """Refuse a baseline whose transfer physics differs from its configuration.

    Arguments:
      compiled = project extension; no C model is constructed in this check.
      name = physical choice absent from the minimal table's scalar schema.
    Returns:
      nothing; a mismatch must fail before a derivative can mix models.
    """
    tables, parameters = toy_tables()
    recorded = dict(parameters)
    recorded["mnu"] = 0.0
    recorded[name] += 0.01
    with pytest.raises(ValueError, match="table provenance"):
        PhysicalProvider(
        bispectrum_table=compact_bispectrum_grid(zmax=tables["z"][-1]),
            compiled=compiled, reference_tables=tables,
            reference_metadata={"parameters": recorded},
            reference_parameters=parameters,
        )
