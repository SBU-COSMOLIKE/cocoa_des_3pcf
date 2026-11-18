"""Supply a paired matter-power and matter-bispectrum feedback response.

Gas physics changes P and B differently. A simulation or emulator must
supply both ratios with one stated matter definition. The numerical family
here is R_X(A)=exp(A*log(R_X)), so A=0 is dark-matter-only and A=1 is the
supplied pair. It is an interpolation model, not a calibrated mapping from
A to AGN temperature or another physical feedback parameter.

Each pair is an immutable snapshot. A loaded file holds its ratios fixed
when cosmology changes; BaccoFeedback instead supplies a new pair at each
physical point. The declared IA approximation
changes the nonlinear-alignment baseline and holds quadratic IA spectra at
their dark-matter-only values. Matter-only response tables cannot determine
the feedback response of intrinsic galaxy shapes. The covariance remains
the separately chosen fixed fiducial covariance.
"""

import json
from pathlib import Path

import numpy as np

from .ownership import PreparedGeometry


class FeedbackPair(PreparedGeometry):
    """Own uniform ratio tables and their scientific interpretation.

    Arguments:
      z, logk, logu, v = uniform increasing axes, each with >=4 nodes.
        logk=ln(k/[h/Mpc]). For sorted B sides a<=b<=c, k=b,
        u=a/b and v=(c-b)/a. z>=0, logu<=0, and 0<=v<=1.
      log_power = ln(P_hydro/P_DMO), [nz,nk].
      log_bispectrum = ln(B_hydro/B_DMO), [nz,nk,nu,nv].
      metadata = source/name and the explicit conventions checked below.
        Only total-matter, fixed-ratio, NLA-baseline feedback is supported.
        domain='clamp' explicitly holds each boundary value outside the
        measured range; domain='error' refuses those queries.
      boost = positive integer dense-grid interval multiplier. The C model
        uses natural cubic preparation and uniform linear hot-path reads.
    Returns:
      immutable owned arrays, metadata and native constructor dictionary.
    Raises:
      ValueError for incomplete provenance, unsupported physics or bad arrays.
    """

    def __init__(self, z, logk, logu, v, log_power, log_bispectrum, metadata, boost=2):
        """Validate before native construction or any physical evaluation."""
        required = {
            "name",
            "source_url",
            "matter_field",
            "cosmology_dependence",
            "ia_response",
            "amplitude_model",
            "domain",
        }
        if not isinstance(metadata, dict) or not required.issubset(metadata):
            raise ValueError(f"Feedback metadata requires {sorted(required)}.")
        choices = {
            "matter_field": "total",
            "cosmology_dependence": "fixed_ratio",
            "ia_response": "nla_only",
            "amplitude_model": "log_response",
        }
        for name, expected in choices.items():
            if metadata[name] != expected:
                raise ValueError(f"Feedback {name} must be {expected} for this model.")
        if (not isinstance(metadata["name"], str)
                or not metadata["name"].strip()
                or not isinstance(metadata["source_url"], str)
                or not metadata["source_url"].strip()
                or metadata["domain"] not in ("clamp", "error")):
            raise ValueError("Feedback needs a name, source URL and clamp/error domain.")
        if (isinstance(boost, bool) or not np.isscalar(boost)
                or not np.isfinite(boost) or int(boost) != boost or boost < 1):
            raise ValueError("Feedback boost must be a positive integer.")
        axes = {}
        for name, supplied in (("z", z), ("logk", logk), ("logu", logu), ("v", v)):
            values = np.array(supplied, dtype=np.float64, order="C", copy=True)
            if (values.ndim != 1 or values.size < 4 or not np.all(np.isfinite(values))
                    or np.any(np.diff(values) <= 0.0)):
                raise ValueError(f"Feedback {name} needs four increasing finite nodes.")
            uniform = np.linspace(values[0], values[-1], values.size)
            tolerance = 64 * np.finfo(float).eps * max(1., np.max(np.abs(values)))
            if not np.allclose(values, uniform, rtol=0., atol=tolerance):
                raise ValueError(f"Feedback {name} must be uniform; prepare it first.")
            if int(boost) * (values.size - 1) + 1 > np.iinfo(np.int32).max:
                raise ValueError("Dense feedback axis exceeds the native integer range.")
            width = float(values[-1]) - float(values[0])
            dense_intervals = int(boost) * (values.size - 1)
            if (not np.isfinite(width) or width <= 0.0
                    or width < dense_intervals / np.finfo(float).max):
                raise ValueError("Feedback axis has unrepresentable dense spacing.")
            axes[name] = values
        if axes["z"][0] < 0. or axes["logu"][-1] > 0.:
            raise ValueError("Feedback needs z>=0 and logu<=0.")
        if axes["v"][0] < 0. or axes["v"][-1] > 1.:
            raise ValueError("Feedback v must stay within [0,1].")
        power = np.asarray(log_power, dtype=np.float64)
        bispectrum = np.asarray(log_bispectrum, dtype=np.float64)
        shape = tuple(axes[name].size for name in ("z", "logk", "logu", "v"))
        if (power.shape != shape[:2] or bispectrum.shape != shape
                or not np.all(np.isfinite(power))
                or not np.all(np.isfinite(bispectrum))):
            raise ValueError("Feedback log ratios must be finite on their declared axes.")
        self.metadata_json = json.dumps(metadata, allow_nan=False, sort_keys=True)
        self.metadata = json.loads(self.metadata_json)
        self.native = dict(axes)
        self.native["log_power"] = np.ascontiguousarray(power).ravel()
        self.native["log_bispectrum"] = np.ascontiguousarray(bispectrum).ravel()
        self.native["boost"] = int(boost)
        self.native["clamp"] = int(metadata["domain"] == "clamp")
        self._seal_geometry()

    def record(self):
        """Return an independent JSON-compatible copy of this fixed pair's provenance."""
        return json.loads(self.metadata_json)


def load_feedback(path, boost=2):
    """Read a paired NPZ without pickle or inferred physical conventions.

    Arguments:
      path = NPZ containing z/logk/logu/v/log_power/log_bispectrum and
        a scalar JSON metadata string; ratios have the shapes above.
      boost = explicit preparation refinement, recorded by FeedbackPair.
    Returns:
      an immutable FeedbackPair; no C model is constructed.
    Raises:
      ValueError for missing fields, malformed JSON or invalid responses.
    """
    with np.load(Path(path), allow_pickle=False) as saved:
        required = ("z", "logk", "logu", "v", "log_power", "log_bispectrum")
        for name in required + ("metadata",):
            if name not in saved:
                raise ValueError(f"Paired feedback file lacks {name}.")
        arrays = {}
        for name in required:
            arrays[name] = saved[name].copy()
        metadata = json.loads(str(saved["metadata"].item()))
    return FeedbackPair(**arrays, metadata=metadata, boost=boost)


def feedback_power_tables(model, tables, amplitude):
    """Modify only nonlinear P for the existing two-point CFASTPT calculation.

    Arguments:
      model = C model owning the paired response table.
      tables = DMO physical-table dictionary with [nz,nk] nonlinear power.
      amplitude = finite common log-response amplitude used for Map3 too.
    Returns:
      a shallow dictionary copy with a new nonlinear-P array. Linear P,
      growth and all source tables are retained unchanged. The input is
      never modified, including after an A/B/A notebook sequence.
    """
    if not np.isfinite(amplitude):
        raise ValueError("Feedback amplitude must be finite.")
    result = dict(tables)
    if amplitude == 0.0:
        return result
    wave, redshift = np.meshgrid(tables["k"], tables["z"])
    ratio = model.power_response(
        k=wave.ravel(), z=redshift.ravel(), amplitude=float(amplitude),
    ).reshape(tables["p_nonlinear"].shape)
    result["p_nonlinear"] = tables["p_nonlinear"] * ratio
    return result


def demonstration_feedback_file(path):
    """Write a deterministic teaching pair with no simulation-calibration claim.

    Arguments:
      path = NPZ output destination, normally runs/paired_feedback_demo.npz.
    Returns:
      resolved path. An existing identical file is reused; different arrays
      or metadata are refused so a user-supplied response is never replaced.
    Side effects:
      Creates the parent directory and file if absent. The two suppression
      shapes and their amplitudes are explicit artificial teaching choices.
    """
    z = np.linspace(0., 3.1, 16)
    logk = np.linspace(-15., 14., 32)
    logu = np.linspace(-24., 0., 12)
    v = np.linspace(0., 1., 12)
    wave = np.exp(logk)
    # A bounded turnover makes both ratios approach unity at large scales.
    # Their different amplitudes and B shape dependence are supplied
    # independently; no identity R_B=R_P^(3/2) is assumed.
    turnover = (wave / (1. + wave))**2
    power = 1. - .20 * turnover[None, :] / (1. + z[:, None])
    shape = 1. + .15 * (1. - np.exp(logu[:, None])) * v[None, :]
    bispectrum = 1. - (.32 * turnover[None, :, None, None]
                       * shape[None, None, :, :] / (1. + z[:, None, None, None]))
    metadata = {
        "name": "artificial paired response for teaching and numerical tests",
        "source_url": "FEEDBACK.md#demonstration; analytic functions, no hydro data",
        "matter_field": "total",
        "cosmology_dependence": "fixed_ratio",
        "ia_response": "nla_only",
        "amplitude_model": "log_response",
        "domain": "clamp",
    }
    arrays = {
        "z": z,
        "logk": logk,
        "logu": logu,
        "v": v,
        "log_power": np.log(power),
        "log_bispectrum": np.log(bispectrum),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    # Exercise the same validation used for externally prepared responses.
    FeedbackPair(z=z, logk=logk, logu=logu, v=v, log_power=np.log(power),
                 log_bispectrum=np.log(bispectrum), metadata=metadata)
    destination = Path(path).resolve()
    if destination.exists():
        with np.load(destination, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays):
                raise ValueError("Existing feedback file has different fields.")
            for name, value in arrays.items():
                if not np.array_equal(saved[name], value):
                    raise ValueError("Existing feedback file differs; choose a new path.")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **arrays)
    return destination
