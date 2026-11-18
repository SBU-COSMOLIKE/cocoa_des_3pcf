"""Run the physical feedback examples through Cocoa's existing theory block.

The notebook compares the change in all 400 Y3 two-point rows at the same
cosmology. It changes only nonlinear matter power, as the LSST Y1 feedback
example does. The CFASTPT quadratic IA loops retain their linear DMO input.
No three-point response is inferred from these power-only scenarios.
BACCO's separate paired diagnostic uses its calibrated cold-matter models.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]  # Owns notebook/configs/runs.
sys.path.insert(0, str(PROJECT / "python"))
sys.path.insert(0, str(PROJECT / "scripts"))

import run_3pcf as driver
from des3pcf.baryon_scenarios import BaccoColdResponses, bfmt_power_response


def _two_point_mean(prediction, tables, metadata, parameters, ia):
    """Install the supplied power arrays and compute all selected xi rows.

    Arguments:
      prediction = joint calculation restricted to two-point rows.
      tables, metadata = physical arrays and their preparation record.
      parameters, ia = the shared cosmology/nuisance and alignment choices.
    Returns:
      xi+[theta,bin-pair] and xi- values in the supplied row order.
      The adapter reinstalls core state on every call, including restoration.
    """
    even, odd = prediction.two_point_core.spectra(
        parameters=parameters, tables=tables, metadata=metadata, ia=ia,
        ell=prediction.xi_engine.ell, bins=prediction.xi_engine.rows.bins[:, :2],
    )
    calibration = np.asarray([
        parameters.get(f"m{index + 1}", 0.0)
        for index in range(prediction.provider.nbin)
    ])
    return prediction.xi_engine.evaluate_spectra(
        spectra_e=even, spectra_b=odd, calibration=calibration,
    )


def run_scenarios(configuration, catalog_path):
    """Compute every declared bfmt example and its two-point response.

    Arguments:
      configuration = existing joint-demo YAML; its paired-feedback entry
        is removed for this power-only comparison, and amplitude is zero.
      catalog_path = YAML listing bfmt selectors, feedback points and papers.
    Returns:
      dictionary with rows, DMO/feedback means, R_P[z,k], grids and records.
      A second FLAMINGO prediction holds R_P constant above 10 h/Mpc to
      illustrate the consequence of extending the tabulated response.
    Raises:
      ValueError for unsupported catalog format or changed DMO restoration;
      physical model errors are propagated, never replaced by dummy curves.
    Side effects:
      Runs the existing bfmt block once per scenario and the ordinary core
      serially. No input files change and no sampler or Fisher run starts.
    """
    path = Path(configuration).resolve()
    catalog_path = Path(catalog_path).resolve()
    catalog = yaml.safe_load(catalog_path.read_text())
    if catalog["format"] != "des3pcf-bfmt-scenarios-v1":
        raise ValueError("Unknown bfmt scenario catalog format.")
    config, path, digest = driver.load_configuration(path=path, command="evaluate")
    config.pop("feedback", None)
    config["ia"]["feedback_amplitude"] = 0.0
    config["threads"] = 1  # Notebook comparisons are light serial checks.
    compiled = driver.compiled_extension(config=config)
    all_rows = driver.rows_for_run(config=config, path=path)
    rows = all_rows.select(indices=np.flatnonzero(all_rows.kind != "map3"))
    provider, metadata = driver.provider_for_run(
        compiled=compiled, config=config, path=path,
    )
    prediction = driver.prediction_for_run(
        compiled=compiled, config=config, path=path, rows=rows, provider=provider,
    )
    model, tables, metadata = provider.get(parameters=config["parameters"])
    reference = _two_point_mean(
        prediction=prediction, tables=tables, metadata=metadata,
        parameters=config["parameters"], ia=config["ia"],
    )
    means = []
    responses = []
    records = []
    names = []
    tail_test = None
    failure = None
    try:
        for scenario in catalog["scenarios"]:
            options = dict(catalog["theory_options"], **scenario["options"])
            response, record = bfmt_power_response(
                cocoa_root=PROJECT.parents[1], parameters=config["parameters"],
                metadata=metadata, options=options, point=scenario["parameters"],
                z=tables["z"], k=tables["k"],
            )
            changed = dict(tables)
            changed["p_nonlinear"] = tables["p_nonlinear"] * response
            mean = _two_point_mean(
                prediction=prediction, tables=changed, metadata=metadata,
                parameters=config["parameters"], ia=config["ia"],
            )
            names.append(scenario["name"])
            means.append(mean)
            responses.append(response)
            records.append(dict(record, name=scenario["name"], paper=scenario["paper"]))
            # This diagnostic changes only the continuation above the stated
            # pivot. It is an alternative modeling assumption, not a repair or
            # a proof of convergence. All original bfmt curves remain available.
            if scenario["name"] == "FLAMINGO":
                held = response.copy()
                for index in range(held.shape[0]):
                    value = np.interp(np.log(10.0), np.log(tables["k"]),
                                      np.log(response[index]))
                    held[index, tables["k"] > 10.0] = np.exp(value)
                changed["p_nonlinear"] = tables["p_nonlinear"] * held
                tail_test = _two_point_mean(
                    prediction=prediction, tables=changed, metadata=metadata,
                    parameters=config["parameters"], ia=config["ia"],
                )
            print(scenario["name"], "completed; all two-point rows:", mean.size)
    except Exception as error:
        failure = error
        raise
    finally:
        # A failed later scenario must not leave the shared ordinary core
        # with an earlier scenario's modified power in a live notebook.
        try:
            restored = _two_point_mean(
                prediction=prediction, tables=tables, metadata=metadata,
                parameters=config["parameters"], ia=config["ia"],
            )
            if not np.array_equal(restored, reference):
                raise ValueError("The core did not restore the exact DMO prediction.")
        except Exception as restore_error:
            if failure is None:
                raise
            failure.add_note(f"DMO restoration also failed: {restore_error}")
    input_path = driver.resolve_path(path, config["physics"]["tables"])
    return {
        "rows": rows, "reference": reference, "means": np.asarray(means),
        "responses": np.asarray(responses), "k": tables["k"], "z": tables["z"],
        "names": names, "records": records, "tail_test": tail_test,
        "record": {
            "configuration_sha256": digest,
            "effective_overrides": {
                "feedback": "removed for power-only comparison",
                "ia.feedback_amplitude": 0.0, "threads": 1,
            },
            "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
            "physical_tables_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "threads": 1, "restoration_exact": True,
            "scope": "power-only scenario; no inferred bispectrum or feedback constraint",
            "flamingo_tail_test": "hold R_P constant above 10 h/Mpc",
        },
    }


def bacco_diagnostic(records):
    """Evaluate the installed paired models on two supported triangle families.

    Arguments:
      records = bfmt records from run_scenarios, including BACCO coordinates.
    Returns:
      k[80], power[2,80], bispectrum[2,2,80], and JSON-compatible provenance.
      Axis meanings are redshift [0,1], triangle [equilateral,scalene], k.
      Scalene sides are [0.6k,k,1.2k]. All are inside the weights' side box.
    Side effects:
      Loads two already-installed small emulators. No physical projection.
    """
    record = next(item for item in records if item["name"] == "BACCO")
    emulator = BaccoColdResponses(coordinates=record["bacco_coordinates"])
    # The side range 0.03--6 stays away from calibration boundaries for
    # both shapes. Avoid squeezed or degenerate triangles in this example.
    wave = np.geomspace(0.05, 5.0, 80)
    power = []
    bispectrum = []
    for redshift in (0.0, 1.0):
        shapes = []
        for ratios in ((1.0, 1.0, 1.0), (0.6, 1.0, 1.2)):
            sides = wave[:, None] * np.asarray(ratios)[None, :]
            response_p, response_b = emulator.evaluate(z=redshift, k=wave, sides=sides)
            shapes.append(response_b)
        power.append(response_p)
        bispectrum.append(shapes)
    return {
        "k": wave, "power": np.asarray(power), "bispectrum": np.asarray(bispectrum),
        "record": {"coordinates": record["bacco_coordinates"],
                   "sigma8_source": record["bacco_sigma8_source"],
                   "weights": emulator.weights, "matter_field": "CDM+baryons",
                   "redshifts": [0.0, 1.0],
                   "shapes": [[1.0, 1.0, 1.0], [0.6, 1.0, 1.2]]},
    }


def save_scenarios(result, paired, destination):
    """Save the notebook's plotted arrays and input evidence without pickle.

    Arguments:
      result, paired = outputs of run_scenarios and bacco_diagnostic.
      destination = notebook-owned NPZ under runs/; reruns replace this output.
    Returns:
      resolved output path. Source inputs and emulator weights are untouched.
    """
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = dict(result["record"], scenarios=result["records"], paired=paired["record"])
    arrays = dict(
        k=result["k"], z=result["z"], responses=result["responses"],
        reference=result["reference"], means=result["means"],
        labels=result["rows"].labels,
        paired_k=paired["k"], paired_power=paired["power"],
        paired_bispectrum=paired["bispectrum"],
        metadata=np.asarray(json.dumps(record, sort_keys=True)),
    )
    if result["tail_test"] is not None:
        arrays["tail_test"] = result["tail_test"]
    np.savez_compressed(destination, **arrays)
    return destination
