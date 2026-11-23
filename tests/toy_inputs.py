"""Controlled toy inputs shared by the component tests.

toy_tables returns two smooth source curves, a positive power spectrum
and a flat matter plus Lambda background in h-based units;
compact_bispectrum_grid returns small bispectrum axes. They test state
ownership and input handling of the compiled model, not DES accuracy.
"""

import numpy as np


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
