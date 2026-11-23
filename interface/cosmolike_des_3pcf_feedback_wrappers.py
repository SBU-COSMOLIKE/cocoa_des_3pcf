"""Run the physical feedback examples through Cocoa's existing theory block.

The notebook compares the change in all 400 Y3 two-point rows at the same
cosmology when baryonic feedback modifies the nonlinear matter power.

Terms used in this file:
  bfmt = Cocoa's baryonic-feedback Cobaya theory block (the repository
      baryon_suppression); given a feedback model and its parameters it
      returns R_P = P_feedback/P_DMO, the ratio of the matter power
      spectrum with baryonic feedback to the gravity-only one.
  DMO = dark-matter-only: the power spectrum without baryonic feedback,
      the reference every response ratio divides by.
  R_P = that power-spectrum response ratio, tabulated over [z, k].
  FLAMINGO = a hydrodynamical simulation suite (one of the bfmt feedback
      models); BACCO = an emulator calibrated on simulations, whose paired
      diagnostic below evaluates its power and bispectrum responses
      together, from its calibrated cold-matter models.

Each scenario changes only the nonlinear matter power, as the LSST Y1
feedback example does. The CFASTPT quadratic intrinsic-alignment loops
retain their linear DMO input, and no three-point response is inferred
from these power-only scenarios.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]  # Owns notebook/configs/runs.
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT.parents[1] / "external_modules/code/cosmolike_core"))

import run_3pcf as driver
from theory.bacco_bispectrum.emulator import BaccoColdResponses
from des3pcf.baryon_scenarios import bfmt_power_response


def _two_point_mean(prediction, tables, metadata, parameters, ia):
    """Install the supplied power arrays and compute all selected xi rows.

    The ordinary two-point core keeps its state in C global variables, so
    this function reinstalls the full cosmology, samples and nuisance
    parameters from the supplied tables on every call; calling it again
    with the original tables restores the original state exactly.

    Arguments:
      prediction = joint calculation restricted to two-point rows.
      tables, metadata = physical arrays and their preparation record.
      parameters, ia = the shared cosmology/nuisance and alignment choices.
    Returns:
      xi+[theta,bin-pair] and xi- values in the supplied row order.
    Raises:
      ValueError when massive-neutrino inputs are incomplete or the
      source redshift grid is not uniform.
    Side effects:
      Overwrites the compiled two-point core's cosmology, source/lens
      samples and nuisance parameters with the supplied values.
    """
    # ----------------------------------------------------------------------
    # Validation. checked is the same object as tables, renamed to mark
    # the point past which its entries have been checked. A run with
    # massive neutrinos needs both the recorded omega_nu_h2 and the linear
    # cold (baryon + CDM) power P_cb.
    # ----------------------------------------------------------------------
    core = prediction.compiled
    checked = tables
    nbin = prediction.provider.nbin
    threads = prediction.two_point_core["threads"]
    h = float(parameters["h"])
    omega_b = float(parameters["omega_b"])
    omega_nu = float(metadata.get("omega_nu_h2", 0.0))
    if parameters.get("mnu", 0.0) > 0.0 and "omega_nu_h2" not in metadata:
        raise ValueError("Massive-neutrino tables require recorded omega_nu_h2.")
    if omega_nu > 0.0 and "p_linear_cb" not in checked:
        raise ValueError(
            "Massive-neutrino core preparation requires linear P_cb[z,k]."
        )

    # ----------------------------------------------------------------------
    # Intrinsic-alignment slot arrays, in the ordinary core convention and
    # without any amplitude or pivot conversion: under the power-law
    # redshift evolution the core reads slot 0 as the amplitude and slot 1
    # as the redshift exponent (first = tidal alignment, second = tidal
    # torquing, density = the density-weighting amplitude alone).
    # ----------------------------------------------------------------------
    first = np.zeros(nbin, dtype=np.float64)
    second = np.zeros(nbin, dtype=np.float64)
    density = np.zeros(nbin, dtype=np.float64)
    first[:2] = [ia.get("a1", 0.0), ia.get("eta1", 0.0)]
    second[:2] = [ia.get("a2", 0.0), ia.get("eta2", 0.0)]
    density[0] = ia.get("bta", 0.0)

    # ----------------------------------------------------------------------
    # Source grid check. The ordinary midpoint reader reconstructs uniform
    # coordinates from the endpoint spacing, so an irregular independent
    # axis would be silently moved; refuse it instead.
    # ----------------------------------------------------------------------
    source_z = checked.get("source_z", checked["z"])
    uniform = np.linspace(start=source_z[0], stop=source_z[-1],
                          num=source_z.size)
    tolerance = 32.0 * np.finfo(float).eps * source_z[-1]
    if not np.allclose(source_z, uniform, rtol=0.0, atol=tolerance):
        raise ValueError("The two-point bridge requires a uniform source_z grid.")

    # ----------------------------------------------------------------------
    # Samples and cosmology into the core. The sample table is one column
    # of redshifts followed by one n(z) column per bin; the one-bin lens
    # sample only fills the core's mandatory lens slot. The old core
    # stores P as [k,z], whereas the new tables store [z,k]; Fortran
    # flattening of the latter gives exactly the former order.
    # ----------------------------------------------------------------------
    source_table = np.column_stack((source_z, checked["sources"].T))
    lens_table = source_table[:, :2].copy()
    core.set_source_sample(nofz=source_table)
    core.set_lens_sample(nofz=lens_table)
    core.init_ntomo_powerspectra()
    cold = []
    if "p_linear_cb" in checked:
        cold = np.log(checked["p_linear_cb"]).flatten(order="F")
    core.set_cosmology(
        omegam=checked["omega_m"],
        H0=100.0 * h,
        log10k_2D=np.log10(checked["k"]),
        z_2D=checked["z"],
        lnP_linear=np.log(checked["p_linear"]).flatten(order="F"),
        lnP_nonlinear=np.log(checked["p_nonlinear"]).flatten(order="F"),
        G=checked["growth"] * (1.0 + checked["z"]),
        z_1D=checked["z"],
        chi=checked["chi"],
        omegab=omega_b,
        z_G=checked["z"],
        omegan2=omega_nu,
        lnP_linear_cb=cold,
    )

    # ----------------------------------------------------------------------
    # Nuisance parameters: zero photo-z shifts and native calibrations
    # (the calibration is applied in Python below), then the alignment
    # amplitudes.
    # ----------------------------------------------------------------------
    core.set_nuisance_shear_photoz(bias=np.zeros(nbin))
    core.set_nuisance_shear_calib(M=np.zeros(nbin))
    core.set_nuisance_ia(
        A1=first,
        A2=second,
        B_TA=density,
    )

    # ----------------------------------------------------------------------
    # Projection: native E- and B-mode spectra per bin pair, then the
    # fixed quadrature turns them into the binned xi rows. The list
    # comprehension collects the per-bin calibration m1..m<nbin> from the
    # parameters (0 where a name is absent) into one array.
    # ----------------------------------------------------------------------
    core.set_omp_threads(n=threads)
    even, odd = core.shear_power_rows(
        ell=prediction.xi_ell,
        bins=np.ascontiguousarray(prediction.xi_rows.bins[:, :2], dtype=np.int32),
    )
    calibration = np.asarray([
        parameters.get(f"m{index + 1}", 0.0)
        for index in range(nbin)
    ])
    return core.production.xi_from_power(
        ell=prediction.xi_ell,
        dell=prediction.xi_dell,
        theta=prediction.xi_theta,
        theta_weight=prediction.xi_weight,
        sign=prediction.xi_sign,
        spectra_e=even,
        spectra_b=odd,
        bins=np.ascontiguousarray(prediction.xi_rows.bins[:, :2], dtype=np.int32),
        calibration=calibration,
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
    # ----------------------------------------------------------------------
    # Configuration. The catalog format tag guards against reading an
    # unrelated YAML; the joint-demo configuration is reduced to a
    # power-only comparison (no paired feedback, zero feedback amplitude).
    # ----------------------------------------------------------------------
    path = Path(configuration).resolve()
    catalog_path = Path(catalog_path).resolve()
    catalog = yaml.safe_load(catalog_path.read_text())
    if catalog["format"] != "des3pcf-bfmt-scenarios-v1":
        raise ValueError("Unknown bfmt scenario catalog format.")
    # load_configuration returns a three-value tuple, in this order: the
    # validated configuration dictionary, the resolved path and the
    # SHA-256 digest of the file (recorded in the result below).
    config, path, digest = driver.load_configuration(path=path, command="evaluate")
    config.pop("feedback", None)
    config["ia"]["feedback_amplitude"] = 0.0
    config["threads"] = 1  # Notebook comparisons are light serial checks.

    # ----------------------------------------------------------------------
    # The shared calculation: compiled core, the two-point rows (Map3 rows
    # are dropped), the physical provider and the DMO reference mean.
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # Scenario loop: for each catalog entry, ask bfmt for R_P on the
    # shared [z, k] grids, multiply the nonlinear power by it and
    # recompute the xi rows. The try/finally pair guarantees the shared
    # core is restored afterwards, whatever happens inside the loop:
    # `failure = error; raise` keeps a reference to the original exception
    # while re-raising it, so the finally block can attach restoration
    # problems to it instead of hiding it.
    # ----------------------------------------------------------------------
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

            # The FLAMINGO tail test changes only the continuation above
            # the stated pivot: hold R_P at its 10 h/Mpc value for every
            # larger k. It is an alternative modeling assumption, not a
            # repair or a proof of convergence; all original bfmt curves
            # remain available.
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
        # with an earlier scenario's modified power in a live notebook, so
        # the DMO tables are reinstalled and the result compared with the
        # reference. When the loop itself already raised, a restoration
        # problem is attached to that exception with add_note (which
        # appends one extra text line to an exception's report; it exists
        # from Python 3.11 on) instead of replacing it; a raise inside a
        # finally block would otherwise swallow the original error.
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

    # ----------------------------------------------------------------------
    # Result: the curves plus a record naming every override and input
    # digest, so the saved npz is traceable to its exact inputs.
    # ----------------------------------------------------------------------
    input_path = driver.resolve_path(path, config["physics"]["tables"])
    return {
        "rows": rows,
        "reference": reference,
        "means": np.asarray(means),
        "responses": np.asarray(responses),
        "k": tables["k"],
        "z": tables["z"],
        "names": names,
        "records": records,
        "tail_test": tail_test,
        "record": {
            "configuration_sha256": digest,
            "effective_overrides": {
                "feedback": "removed for power-only comparison",
                "ia.feedback_amplitude": 0.0,
                "threads": 1,
            },
            "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
            "physical_tables_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "threads": 1,
            "restoration_exact": True,
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
    Raises:
      StopIteration when the records contain no BACCO entry.
    Side effects:
      Loads two already-installed small emulators. No physical projection.
    """
    # next(...) takes the first record whose name is BACCO from the
    # generator expression; with no such record the generator is exhausted
    # and next raises StopIteration.
    record = next(item for item in records if item["name"] == "BACCO")
    emulator = BaccoColdResponses(coordinates=record["bacco_coordinates"])

    # The side range 0.03--6 stays away from calibration boundaries for
    # both shapes; the 80 sampled sides run over 0.05--5 h/Mpc. Squeezed
    # and degenerate triangles are avoided in this example.
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
        "k": wave,
        "power": np.asarray(power),
        "bispectrum": np.asarray(bispectrum),
        "record": {
            "coordinates": record["bacco_coordinates"],
            "sigma8_source": record["bacco_sigma8_source"],
            "weights": emulator.weights,
            "matter_field": "CDM+baryons",
            "redshifts": [0.0, 1.0],
            "shapes": [[1.0, 1.0, 1.0], [0.6, 1.0, 1.2]],
        },
    }


def save_scenarios(result, paired, destination):
    """Save the notebook's plotted arrays and input evidence without pickle.

    Arguments:
      result, paired = outputs of run_scenarios and bacco_diagnostic.
      destination = notebook-owned NPZ under chains/; reruns replace this output.
    Returns:
      resolved output path. Source inputs and emulator weights are untouched.
    Side effects:
      Creates the destination directory when needed and writes the npz
      file (an existing file at the destination is replaced).
    """
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    # One JSON metadata string carries every record, so the npz holds only
    # plain arrays and loads with allow_pickle=False.
    record = dict(result["record"], scenarios=result["records"], paired=paired["record"])
    arrays = dict(
        k=result["k"],
        z=result["z"],
        responses=result["responses"],
        reference=result["reference"],
        means=result["means"],
        labels=result["rows"].labels,
        paired_k=paired["k"],
        paired_power=paired["power"],
        paired_bispectrum=paired["bispectrum"],
        metadata=np.asarray(json.dumps(record, sort_keys=True)),
    )
    if result["tail_test"] is not None:
        arrays["tail_test"] = result["tail_test"]
    np.savez_compressed(destination, **arrays)
    return destination
