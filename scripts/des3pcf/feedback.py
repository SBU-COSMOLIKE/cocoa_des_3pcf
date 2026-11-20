"""Read paired survey-example responses and prepare artificial test inputs."""
import json
from pathlib import Path
import numpy as np
from theory.bacco_bispectrum.response import FeedbackPair

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

def demonstration_feedback_file(path):
    """Write a deterministic teaching pair with no simulation-calibration claim.

    Arguments:
      path = NPZ output destination, normally chains/paired_feedback_demo.npz.
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
