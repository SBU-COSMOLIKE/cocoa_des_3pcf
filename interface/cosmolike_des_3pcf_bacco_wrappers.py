"""Teach the joint BACCO forecast with explicit model and sampling comparisons.

The executable YAML supplies all 480 Y3 rows, their cross-covariance and the
archived triangle geometry. This smaller study varies sigma8, two physical
feedback coordinates and two IA amplitudes, holding the other cosmological
and nuisance parameters fixed. It measures how the response assumptions affect
that conditional forecast; it does not reproduce a published posterior.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_triangular


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
sys.path.insert(0, str(PROJECT / "scripts"))

import run_3pcf as driver
from des3pcf.forecast import ExactMeanCache, differentiate, figure_of_merit
from des3pcf.fisher_comparison import compare_information, marginalized_errors
from des3pcf.provenance import file_digest, prediction_identity
from des3pcf.provenance import verify_prediction_identity


# Sigma8 exercises the full CAMB-to-emulator dependency. The remaining
# columns ask whether joint shear separates gas feedback from galaxy shapes.
PARAMETERS = (
    "sigma8",
    "M_c_baccoemu",
    "eta_baccoemu",
    "a1",
    "a2",
)
FEEDBACK_PAIR = ("M_c_baccoemu", "eta_baccoemu")
CASES = (
    ("baseline", {}, {}, "five_point"),
    ("cold_on_total", {"neutrino_model": "cold_on_total"}, {}, "central"),
    ("side_clamp", {"high_k_mode": "side_clamp"}, {}, "central"),
    ("high_k_5", {"high_k_anchor": 5.}, {}, "central"),
    ("low_k_004", {"low_k_anchor": .04}, {}, "mean"),
    ("high_z_fade", {"high_z_mode": "fade_to_unity"}, {}, "mean"),
    ("feedback_z", {}, {"feedback_z_boost": 2.}, "mean"),
    ("feedback_k", {}, {"feedback_k_boost": 2.}, "central"),
    ("feedback_u", {}, {"feedback_u_boost": 2.}, "mean"),
    ("feedback_v", {}, {"feedback_v_boost": 2.}, "mean"),
    ("feedback_dense", {}, {"feedback_table_boost": 1.5}, "mean"),
)


def _identity(config, path, compiled, prediction):
    """Bind an archive to its actual inputs, libraries and this study driver."""
    identity = prediction_identity(
        config=config, configuration_path=path, project=PROJECT,
        three_point_path=compiled.__file__,
        two_point_path=prediction.two_point_core.core.__file__,
    )
    identity["study_driver_sha256"] = file_digest(path=__file__)
    identity["resolved_configuration"] = config
    identity["sampled_parameters"] = list(PARAMETERS)
    return identity


def run_case(
    configuration, name, settings, accuracy, stencil, output, threads=4,
    compute_if_missing=False,
):
    """Compute one declared case or verify an exactly matching saved archive.

    Arguments:
      configuration = full BACCO Fisher YAML; other parameters remain fixed.
      name, settings, accuracy = case label, response-policy replacements and
        individual resolution boosts. Each case owns a new provider.
      stencil = five_point (h and h/2), central (+/-h), or mean only.
      output = directory for one independently resumable NPZ per case.
      threads = one to four OpenMP workers; BLAS and emulator workers are one.
      compute_if_missing = explicitly permit a physical run when no saved
        archive exists. False makes this a checked read with no calculation.
    Returns:
      numerical arrays plus a JSON-compatible record. An existing archive is
      reused only after exact dependency/configuration equality is verified.
    Side effects:
      Runs real CAMB/native/emulator calculations as needed and saves results.
      No input files or native libraries are changed.
    """
    destination = Path(output) / f"{name}.npz"
    if not destination.exists() and not compute_if_missing:
        raise FileNotFoundError(
            f"No saved BACCO case at {destination}; explicitly set "
            "compute_if_missing=True to calculate it."
        )
    config, path, _ = driver.load_configuration(path=configuration, command="fisher")
    config = deepcopy(config)
    if threads not in (1, 2, 3, 4):
        raise ValueError("Use one to four OpenMP workers.")
    if stencil not in ("five_point", "central", "mean"):
        raise ValueError("Unknown BACCO study derivative stencil.")
    config["threads"] = threads
    config["feedback"]["settings"].update(settings)
    config["accuracy"].update(accuracy)
    compiled = driver.compiled_extension(config=config)
    rows = driver.rows_for_run(config=config, path=path)
    provider, _ = driver.provider_for_run(compiled=compiled, config=config, path=path)
    prediction = driver.prediction_for_run(compiled=compiled, config=config,
                                            path=path, rows=rows, provider=provider)
    identity = _identity(config=config, path=path, compiled=compiled,
                         prediction=prediction)
    identity["stencil"] = stencil
    if destination.exists():
        with np.load(destination, allow_pickle=False) as archive:
            record = json.loads(str(archive["metadata_json"].item()))
            verify_prediction_identity(recorded=record["identity"], current=identity,
                                       description=name)
            arrays = {}
            for key in archive.files:
                if key != "metadata_json":
                    arrays[key] = archive[key].copy()
        return arrays, record
    covariance = driver.read_matrix(settings=config["covariance"], path=path,
                                    rows=rows, positive=True)
    reference = dict(config["ia"], **config["parameters"])
    steps = {parameter: config["fisher"]["steps"][parameter] for parameter in PARAMETERS}
    cached = ExactMeanCache(predict=prediction)
    print(f"{name}: fiducial prediction", flush=True)
    mean = cached(parameters=reference)
    response_record = provider._active_feedback.record()
    arrays = {
        "mean": mean,
        "covariance": covariance,
        "kind": rows.kind,
        "bins": rows.bins,
        "angles": rows.angles,
        "row_labels": rows.labels,
        "names": np.asarray(PARAMETERS),
    }
    fixed_parameters = {}
    for key, value in reference.items():
        if key not in PARAMETERS:
            fixed_parameters[key] = value
    record = {
        "identity": identity,
        "name": name,
        "stencil": stencil,
        "feedback_reference": response_record,
        "priors": {},
        "fixed_parameters": fixed_parameters,
    }
    if stencil != "mean":
        derivative = np.empty((mean.size, len(PARAMETERS)))
        refined = np.empty_like(derivative)
        central = np.empty_like(derivative)
        for column, parameter in enumerate(PARAMETERS):
            print(f"{name}: derivative of {parameter}", flush=True)
            step = steps[parameter]
            if stencil == "five_point":
                _, _, d_h, d_h2 = differentiate(
                    predict=cached, parameters=reference, steps={parameter: step},
                    bounds=config["fisher"].get("bounds"),
                )
                derivative[:, column], refined[:, column] = d_h[:, 0], d_h2[:, 0]
            low_point = dict(reference)
            high_point = dict(reference)
            low_point[parameter] -= step
            high_point[parameter] += step
            low = cached(parameters=low_point)
            high = cached(parameters=high_point)
            central[:, column] = (high-low) / (2.*step)
        arrays["derivatives_central"] = central
        comparisons = {"central": central}
        if stencil == "five_point":
            arrays["derivatives_h"], arrays["derivatives_h2"] = derivative, refined
            comparisons.update(h=derivative, h2=refined)
        record["forecasts"] = {}
        for label, jacobian in comparisons.items():
            comparison = compare_information(derivatives=jacobian, names=PARAMETERS,
                                             kind=rows.kind, covariance=covariance)
            errors, statuses = marginalized_errors(comparison=comparison,
                                                   names=PARAMETERS)
            for key, value in dict(comparison, **errors).items():
                arrays[f"{label}_{key}"] = value
            merits = {}
            for probe in ("two_point", "joint"):
                try:
                    merits[probe] = figure_of_merit(
                        fisher=comparison[f"fisher_{probe}"], names=PARAMETERS,
                        pair=FEEDBACK_PAIR,
                    )
                except ValueError as error:
                    merits[probe] = str(error)
            record["forecasts"][label] = {"rank_status": statuses,
                                            "feedback_fom": merits}
        # Re-entering the center after BCM/IA changes also exercises the
        # provider's restoration; do not use ExactMeanCache for this check.
        np.testing.assert_array_equal(prediction(parameters=reference), mean)
        record["direct_restoration_exact"] = True
    final_identity = _identity(config=config, path=path, compiled=compiled,
                                prediction=prediction)
    final_identity["stencil"] = stencil
    verify_prediction_identity(recorded=identity, current=final_identity,
                                description=f"{name} during evaluation")
    record["prediction_count"] = len(cached.values)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        np.savez_compressed(stream, **arrays,
                            metadata_json=np.asarray(json.dumps(record, sort_keys=True)))
    print(f"{name}: saved {destination.name}", flush=True)
    return arrays, record


def mean_distance(reference, changed):
    """Measure a model change in the full survey covariance, with cross terms.

    Delta chi squared is the expected separation of two fixed mean vectors.
    It is not a fitted parameter bias or an interpolation error percentage.
    """
    factor = np.linalg.cholesky(reference["covariance"])
    difference = solve_triangular(factor, changed["mean"]-reference["mean"], lower=True)
    return float(difference @ difference)
