"""Connect retained DES stencils to the shared Fisher-noise diagnostic.

The estimator belongs to cosmolike_notebook_utils.fisher. This module only
adapts saved vectors and dictionary-based physical parameters to its API.
One baseline Fisher matrix, including its priors, supplies the marginalized
parameter covariance for every candidate accuracy setting. Scoring saved
stencils requires no new physical prediction. A live candidate instead
needs four displaced vectors per selected column and one shared center.

For five points at offsets [-2,-1,0,1,2] times an absolute step eps, the
shared estimator uses fourth-difference weights [1,-4,6,-4,1]. Their
squared sum is 70. The fourth-order derivative uses [1,-8,0,8,-1]/(12 eps),
whose squared coefficient sum is 130/(144 eps**2). The resulting average
diagonal Fisher addition is a=(130/144)*(sigma/eps)**2, where sigma is the
fourth difference's covariance-weighted norm divided by sqrt(70).

Equation eq:fomall in the Fisher machinery note then weights these additions
by each parameter's marginalized variance and correlation with the selected
pair. The shared predict_fom_noise returns both the first-order change and
the exact FoM ratio for F+diag(a). That exact ratio still assumes the stated
diagonal-noise model; it is not the exact error of a physical forecast.

The fourth difference cancels every cubic or lower smooth response. A smooth
fourth derivative can contaminate the estimate, while an odd coherent error
can bias a derivative and remain invisible. Correlated errors across stencil
points or parameter columns invalidate the independent-jitter interpretation.
The realized terms involving the true derivative and its error are also
absent from the diagonal model. Small scores therefore do not establish full
numerical convergence. Rank candidate columns by their measured baseline
contributions, report the omitted fraction, and retain direct grid/step
checks. A small correlation weight alone gives no bound on unknown noise.
"""

import json
from pathlib import Path

import numpy as np

from cosmolike_notebook_utils.fisher import (
    get_fom_noise_weights,
    get_stencil_noise,
    predict_fom_noise,
)


def retained_stencil_noise(stencil_values, steps, invcov):
    """Score saved five-point columns without evaluating a physical model.

    Pass the h archive with h, or the h/2 archive with h/2. The two archives
    share several physical points, so their inferred noise is correlated;
    they are separate diagnostics, not independent observations to average.

    Arguments:
      stencil_values = finite [nparameter,5,ndata] vectors in the order
                       [-2,-1,0,1,2]; every center must be identical.
      steps = positive absolute parameter steps, shape [nparameter].
      invcov = precision matrix for these same data rows, shape [ndata,ndata].
    Returns:
      dictionary with shared sigma, eps and a arrays [nparameter], and
      shared ddv columns [ndata,nparameter]. These derivatives use the
      shared helper's accumulation order, which can differ in roundoff
      from forecast.differentiate. Keep the saved baseline Fisher matrix.
    Raises:
      ValueError for malformed arrays, inconsistent centers or collapsed
      offset coordinates. The shared helper validates the precision input.
    """
    values = np.asarray(stencil_values, dtype=np.float64)
    if (values.ndim != 3 or values.shape[0] == 0 or values.shape[1] != 5
            or values.shape[2] == 0 or not np.all(np.isfinite(values))):
        raise ValueError("Saved stencils must be finite [nparameter,5,ndata] arrays.")
    nparameter = values.shape[0]
    ndata = values.shape[2]
    absolute_steps = np.asarray(steps, dtype=np.float64)
    if (absolute_steps.shape != (nparameter,)
            or not np.all(np.isfinite(absolute_steps))
            or np.any(absolute_steps <= 0.0)):
        raise ValueError("Saved stencils require one positive absolute step per column.")
    precision = np.asarray(invcov, dtype=np.float64)
    if precision.shape != (ndata, ndata):
        raise ValueError("The precision matrix must describe the saved stencil rows.")
    for column in range(nparameter):
        if not np.array_equal(
            a1=values[column, 2].view(dtype=np.uint64),
            a2=values[0, 2].view(dtype=np.uint64),
        ):
            raise ValueError("Saved stencil centers must have identical float64 values.")
    result = {
        "sigma": np.empty(shape=nparameter, dtype=np.float64),
        "eps": np.empty(shape=nparameter, dtype=np.float64),
        "a": np.empty(shape=nparameter, dtype=np.float64),
        "ddv": np.empty(shape=(ndata, nparameter), dtype=np.float64),
    }
    center = np.zeros(shape=nparameter, dtype=np.float64)
    multiples = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    for column in range(nparameter):
        coordinates = multiples * absolute_steps[column]
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("The absolute step produces nonfinite stencil offsets.")
        if np.any(np.diff(coordinates) <= 0.0):
            raise ValueError("The absolute step collapses stencil offsets.")
        rows = {}
        for row in range(5):
            rows[float(coordinates[row]).hex()] = row

        def lookup(param, AccuracyBoost=1.0):
            """Read the current saved column at an exact requested offset.

            The shared call completes synchronously before the outer loop
            advances to another column. There is no retained callable or
            physical-model fallback for an unrecognized point.
            """
            if AccuracyBoost != 1.0:
                raise ValueError("A retained stencil cannot change accuracy.")
            offsets = np.asarray(param, dtype=np.float64)
            if offsets.shape != (nparameter,) or not np.all(np.isfinite(offsets)):
                raise ValueError("Retained lookup requires finite parameter offsets.")
            for index in range(nparameter):
                if index != column and offsets[index] != 0.0:
                    raise ValueError("Retained lookup received a different column.")
            key = float(offsets[column]).hex()
            if key not in rows:
                raise ValueError("Retained lookup received an unsaved stencil offset.")
            return values[column, rows[key]].copy()

        measured = get_stencil_noise(
            dv=lookup,
            index=column,
            h=float(absolute_steps[column]),
            CV=center,
            AccuracyBoost=1.0,
            invcov=precision,
        )
        for name in ("sigma", "eps", "a"):
            result[name][column] = measured[name]
        result["ddv"][:, column] = measured["ddv"]
    return result


def make_offset_predictor(predict, reference, names):
    """Adapt absolute offsets to a configured dictionary-based prediction.

    Call shared get_stencil_noise with CV=zeros(len(names)) and h equal to
    the absolute YAML step. Then a negative or zero physical fiducial uses
    exactly reference[name]+offset, as forecast.differentiate does. Other
    parameters retain their reference values, including a signed-zero center.

    Arguments:
      predict = callable taking one physical parameter dictionary. Supply
                ExactMeanCache when centers and repeated points should reuse
                their vectors. Accuracy settings belong to this callable.
      reference = finite scalar physical parameter dictionary, copied here.
      names = distinct differentiated parameter names in Fisher-column order.
    Returns:
      dv(param, AccuracyBoost=1.0), compatible with the shared estimator.
      Construction evaluates no mean. Each call invokes predict once.
    Raises:
      ValueError for malformed names/offsets or AccuracyBoost other than one.
      Construct another configured predictor to examine another accuracy.
    """
    if not callable(predict):
        raise ValueError("The physical prediction must be callable.")
    if isinstance(names, str):
        raise ValueError("Parameter names must be a sequence, not one string.")
    ordered_names = tuple(names)
    if not ordered_names or len(set(ordered_names)) != len(ordered_names):
        raise ValueError("Differentiated parameter names must be nonempty and distinct.")
    physical_reference = dict(reference)
    for name, value in physical_reference.items():
        if not np.isfinite(float(value)):
            raise ValueError(f"Reference parameter {name} must be finite.")
    for name in ordered_names:
        if not isinstance(name, str) or name not in physical_reference:
            raise ValueError(f"Missing differentiated reference parameter: {name}")

    def evaluate(param, AccuracyBoost=1.0):
        """Apply physical offsets without multiplying by the fiducial value."""
        if AccuracyBoost != 1.0:
            raise ValueError(
                "Configure candidate accuracy in predict, not AccuracyBoost."
            )
        offsets = np.asarray(param, dtype=np.float64)
        if (offsets.shape != (len(ordered_names),)
                or not np.all(np.isfinite(offsets))):
            raise ValueError("Provide one finite absolute offset per parameter name.")
        point = dict(physical_reference)
        for index, name in enumerate(ordered_names):
            if offsets[index] == 0.0:
                continue
            reference_value = float(physical_reference[name])
            coordinate = reference_value + float(offsets[index])
            if not np.isfinite(coordinate):
                raise ValueError(f"The offset for {name} leaves finite values.")
            if coordinate == reference_value:
                raise ValueError(f"The offset for {name} is below float resolution.")
            point[name] = coordinate
        return predict(point)

    return evaluate


def accuracy_cases_for_request(candidates, only, output):
    """Visit requested cases plus earlier archives in the current plan.

    Existence selects a case for inspection, not scientific reuse. Every
    retained file must subsequently pass load_retained_noise_case with its
    complete expected baseline, configuration, and runtime identity.
    """
    output = Path(output)
    result = []
    for candidate in candidates:
        requested = only is None or candidate["name"] in only
        saved = output / f"{candidate['name']}_noise.npz"
        if requested or saved.is_file():
            result.append(candidate)
    return result


def load_retained_noise_case(path, identity):
    """Read a completed case only after exact dependency-identity validation.

    No value is recovered from summary.json: that file can omit an earlier
    case after a partial invocation. The archive is the authoritative saved
    stencil, and a stale archive raises instead of silently joining a scan.
    """
    with np.load(file=path, allow_pickle=False) as saved:
        recorded = json.loads(str(saved["identity_json"].item()))
        if recorded != identity:
            raise ValueError(
                f"{path} has different inputs; choose a new study directory."
            )
        return {
            "a": saved["a"].copy(),
            "sigma": saved["sigma"].copy(),
            "stencil_values": saved["stencil_values"].copy(),
            "evaluation_count": int(saved["evaluation_count"].item()),
        }
