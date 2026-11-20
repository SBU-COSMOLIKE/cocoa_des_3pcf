"""Read a declared scalar HEALPix window without installing healpy.

A pixel represents an average over a finite angular area. The archived
theory multiplies a shear bispectrum by one scalar window per Fourier leg.
This convention uses the TEMPERATURE column of the HEALPix table, not its
polarization column. It is an explicit reproduction choice, not a claim
that scalar smoothing exactly describes every shear-map estimator.

healpy.pixwin(nside) defaults to ell=0..3*nside-1. The input FITS extends
farther; using all its rows would define a different calculation. Linear
interpolation and zero above the last retained ell follow the archived
theory. The cutoff is discontinuous and needs angular-convergence checks.
"""

import hashlib
from pathlib import Path

import numpy as np
from astropy.io import fits


def checked_window(values):
    """Copy finite nonnegative samples at integer ell, including ell=0.

    Arguments:
      values = one-dimensional angular window with at least two entries.
    Returns:
      contiguous float64 vector. It is copied again by the owning C model.
    """
    window = np.array(values, dtype=np.float64, order="C", copy=True)
    if (window.ndim != 1 or window.size < 2 or window.size > np.iinfo(np.int32).max
            or not np.all(np.isfinite(window)) or np.any(window < 0.0)):
        raise ValueError("Pixel window needs >=2 finite nonnegative integer-ell samples.")
    return window


def load_healpix_window(path, nside, sha256):
    """Verify a local scalar FITS table and retain the archived default ell range.

    Arguments:
      path = already downloaded HEALPix pixel-window FITS.
      nside = declared positive integer HEALPix resolution.
      sha256 = required digest of those exact FITS bytes.
    Returns:
      owned samples and a JSON-compatible provenance dictionary.
    Side effects:
      Reads a local file only. This reader never downloads an input.
    """
    if isinstance(nside, bool) or not isinstance(nside, int) or nside < 1:
        raise ValueError("HEALPix nside must be a positive integer.")
    filename = Path(path)
    digest = hashlib.sha256(filename.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError("Pixel-window SHA256 differs; restore the declared input file.")
    with fits.open(filename, memmap=False) as hdus:
        table = hdus[1]
        if (table.header.get("NSIDE") != nside
                or table.header.get("EXTNAME") != "PIXEL WINDOW"
                or "TEMPERATURE" not in table.columns.names):
            raise ValueError(
                "Pixel-window FITS must match NSIDE and scalar table schema."
            )
        if table.header.get("MAX-LPOL", -1) < 3 * nside - 1:
            raise ValueError("Pixel-window FITS does not cover ell=0..3*nside-1.")
        samples = table.data["TEMPERATURE"]
        if samples.ndim != 1 or samples.size < 3 * nside:
            raise ValueError(
                "Pixel-window scalar column has insufficient integer samples."
            )
        window = checked_window(values=samples[:3 * nside])
    # Official W0 can exceed one by several ulps. Preserve those samples;
    # do not silently renormalize the historical theory convention.
    metadata = {
        "path": str(filename.resolve()),
        "sha256": digest,
        "nside": nside,
        "column": "TEMPERATURE",
        "ell_min": 0,
        "ell_max": 3 * nside - 1,
        "interpolation": "linear_integer_ell",
        "upper_extension": "zero",
        "application": "three_point_angular_observables_only",
    }
    return window, metadata
