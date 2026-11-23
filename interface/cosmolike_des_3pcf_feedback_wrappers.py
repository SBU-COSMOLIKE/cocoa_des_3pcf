"""Run the bfmt feedback scenarios of EXAMPLE_FEEDBACK_SCENARIOS through Cobaya.

The notebook compares the change in all 400 DES Y3 two-point rows at one
cosmology when baryonic feedback modifies the nonlinear matter power. Every
prediction comes from the project's Cobaya likelihood des_3pcf.joint_shear,
built with cobaya.model.get_model from a numbered EXAMPLE yaml.

Terms used in this file:
  bfmt = Cocoa's baryonic-feedback Cobaya theory block (the repository
      baryon_suppression); given a feedback model and its parameters it
      returns R_P = P_feedback/P_DMO, the ratio of the matter power
      spectrum with baryonic feedback to the gravity-only one.
  DMO = dark-matter-only: the power spectrum without baryonic feedback,
      the reference every response ratio divides by.
  R_P = that power-spectrum response ratio, tabulated over [z, k].
  external_baryon_suppression = the likelihood yaml key that requests R_P
      from bfmt and multiplies the nonlinear power of the xi rows by it.
      A power-only model supplies no bispectrum response, so the
      likelihood then refuses every Map3 row; the scenarios therefore use
      data/des_y3_xi1.dataset, whose mask cuts all Map3 rows.
  scenario catalog = configs/feedback_scenarios.yaml: one entry per bfmt
      model with its selector options, fixed feedback parameters and paper.
  FLAMINGO = a hydrodynamical simulation suite (one of the bfmt feedback
      models); BACCO = an emulator calibrated on simulations, whose paired
      diagnostic below evaluates its power and bispectrum responses
      together, from its calibrated cold-matter models.

Each scenario changes only the nonlinear matter power, as the LSST Y1
feedback example does. Intrinsic-alignment loops keep their linear DMO
input, and no three-point response is inferred from these power-only
scenarios.
"""

import copy
import hashlib
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]  # projects/des_3pcf
COCOA = PROJECT.parents[1]

# Cobaya's name of the joint xi/Map3 likelihood in the example yamls, and
# the dataset whose mask keeps the 400 xi rows and cuts all 80 Map3 rows.
LIKELIHOOD = "des_3pcf.joint_shear"
XI_DATASET = "des_y3_xi1.dataset"

# The catalog format tag guards against reading an unrelated YAML.
CATALOG_FORMAT = "des3pcf-bfmt-scenarios-v1"

# The FLAMINGO tail test holds R_P constant above this wavenumber [h/Mpc].
TAIL_PIVOT = 10.0


class _TheoryMessages(logging.Handler):
    """Collect bfmt warnings so a failed emulator cannot pass as no feedback.

    Some bfmt error paths log a message and return R_P = 1 instead of
    raising an exception. run_scenarios attaches this handler to the bfmt
    logger for each evaluation and refuses the scenario when it collected
    any warning or error. The declared high-redshift continuation of bfmt
    (above_zmax) produces no warning and is accepted.
    """

    def __init__(self):
        """Start with an empty message list; the process logging is unchanged."""
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        """Keep the fully formatted message produced by the theory block."""
        self.messages.append(record.getMessage())


def _require_installed_theory_data(model_number):
    """Refuse missing emulator weights before a bfmt dependency can download.

    Several emulator packages fetch absent weight files from the internet
    when they are first constructed. A notebook must not do that silently,
    so the files those constructors read are checked first.

    Arguments:
      model_number = bfmt selector baryon_model, 1 through 5.
    Returns:
      nothing when every needed file is present.
    Raises:
      FileNotFoundError naming the first missing file; Cocoa's installation
      keys (external_modules/code/baryon_suppression/README.md) supply it.
    """
    from theory.bacco_bispectrum.emulator import installed_bacco_weights

    paths = []
    if model_number in (2, 5):
        spec = importlib.util.find_spec(name="BCemu")
        if spec is None:
            raise FileNotFoundError("Install BCemu with Cocoa's installation keys.")
        base = Path(spec.origin).parent / "input_data"
        if model_number == 2:
            for redshift in ("0", "0p5", "1", "1p5", "2"):
                paths.append(base / f"kpls_emulator_z{redshift}_nComp3.pkl")
        else:
            paths.append(base / "BCemu2025_meta.json")
    if model_number == 4:
        installed_bacco_weights()
        spec = importlib.util.find_spec(name="baccoemu")
        base = Path(spec.origin).parent
        # bfmt also loads the ordinary BACCO power constructor, whose
        # directories trigger downloads when absent.
        for name in ("sigma8_emu_2.0.2", "cold_matter_linear_emu_1.0.2",
                     "total_matter_linear_emu_1.0.2", "no_wiggles_emu_1.0.2",
                     "nonlinear_emu_1.0.3"):
            paths.append(base / name)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Cocoa setup must supply {path} before evaluation.")


def _held_tail_theory():
    """Return a bfmt subclass that holds R_P constant above TAIL_PIVOT h/Mpc.

    The FLAMINGO tail test replaces the model's continuation on the
    smallest scales by its value at the pivot. Inside Cobaya the response
    reaches the likelihood only through the bfmt product, so the test uses
    this subclass in place of bfmt: it runs the ordinary bfmt calculation,
    then, at every redshift, sets R_P at every requested wavenumber above
    the pivot to its value there. bfmt requests are in 1/Mpc, so the pivot
    is TAIL_PIVOT * h in those units. This is an alternative modeling
    assumption for one diagnostic, not a repair of the model.

    Returns:
      the class, for the "external" entry of a Cobaya theory block.
    """
    from cobaya.theories.bfmt.bfmt import bfmt

    class held_tail_bfmt(bfmt):
        """bfmt with R_P held at its TAIL_PIVOT h/Mpc value on smaller scales."""

        def calculate(self, state, want_derived=True, **params_values_dict):
            """Run bfmt, then hold each R_P row above the pivot wavenumber.

            Returns:
              False when bfmt rejected the point (Cobaya then assigns zero
              probability), otherwise None.
            Side effects:
              replaces state["baryon_suppression"] rows in place.
            """
            accepted = super().calculate(state, want_derived, **params_values_dict)
            if accepted is False:
                return False
            h = self.provider.get_param("H0") / 100.0
            pivot = TAIL_PIVOT * h
            log_k = np.log(self.requested_k)
            above = self.requested_k > pivot
            for redshift, row in state["baryon_suppression"].items():
                value = np.exp(np.interp(np.log(pivot), log_k, np.log(row)))
                held = row.copy()
                held[above] = value
                state["baryon_suppression"][redshift] = held
            return None

    return held_tail_bfmt


def _scenario_information(information, scenario, theory_options, tail=False):
    """Return the Cobaya input of one bfmt scenario on the xi-only dataset.

    Arguments:
      information = the example yaml as loaded by yaml_load_file, without
        its sampler and output blocks; it is copied, never modified.
      scenario = one catalog entry (name, options, parameters, paper), or
        None for the gravity-only (DMO) reference.
      theory_options = the catalog's shared bfmt options (nz, nk, above_zmax).
      tail = True replaces bfmt by the held-tail subclass (FLAMINGO test).
    Returns:
      a new Cobaya input dictionary.
    Raises:
      ValueError when a feedback parameter already appears in the example.
    """
    info = copy.deepcopy(information)
    likelihood = info["likelihood"][LIKELIHOOD]
    likelihood["data_file"] = XI_DATASET
    if scenario is None:
        return info

    # The fixed feedback parameters enter as constant Cobaya parameters;
    # one already present would be silently overwritten, so refuse it.
    for name in scenario["parameters"]:
        if name in info["params"]:
            raise ValueError(
                f"The feedback parameter {name} of scenario {scenario['name']} "
                "is already a parameter of the example yaml."
            )
    likelihood["external_baryon_suppression"] = True
    options = dict(theory_options)
    options.update(scenario["options"])
    if tail:
        options["external"] = _held_tail_theory()
    info["theory"]["bfmt"] = options
    info["params"].update(scenario["parameters"])
    return info


def _evaluate(info, point):
    """Evaluate one Cobaya model and return its xi rows and response grid.

    bfmt's BACCO branch reads the primordial amplitude A_s from the
    provider without declaring it as a requirement. The example yaml
    samples sigma8, so A_s exists only as a CAMB derived parameter; the
    explicit add_requirements call makes CAMB provide it to every model.

    Arguments:
      info = Cobaya input from _scenario_information.
      point = the example's evaluate point (sampler.evaluate.override).
    Returns:
      dictionary with xi (the xi rows), response (R_P[z,k] in h/Mpc, or None
      for the DMO model), h, As, the likelihood (for its grids) and, for
      bfmt BACCO, the cold-matter coordinates of the paired diagnostic.
    Raises:
      RuntimeError when the posterior is not finite or bfmt logged any
      warning or error (including its fallback to R_P = 1).
    Side effects:
      Runs CAMB, BiHalofit, bfmt and the native xi projection once; the
      native cosmolike state is overwritten, as at every likelihood call.
    """
    from cobaya.model import get_model

    # ----------------------------------------------------------------------
    # Model and message capture. bfmt's logger is switched to WARNING only
    # for this evaluation; the finally block restores it even on errors.
    # ----------------------------------------------------------------------
    model = get_model(info_or_yaml_or_file=info)
    model.add_requirements(requirements={"As": None})
    messages = _TheoryMessages()
    logger = None
    previous_level = None
    if "bfmt" in model.theory:
        logger = model.theory["bfmt"].log
        previous_level = logger.level
        logger.setLevel(logging.WARNING)
        logger.addHandler(messages)

    # ----------------------------------------------------------------------
    # One evaluation. BCEmu's import changes NumPy's global error policy;
    # np.errstate restores the caller's policy when the block ends.
    # ----------------------------------------------------------------------
    try:
        with np.errstate():
            posterior = model.logposterior(params_values=point)
    finally:
        if logger is not None:
            logger.removeHandler(messages)
            logger.setLevel(previous_level)
    if messages.messages:
        raise RuntimeError("bfmt reported: " + "; ".join(messages.messages))
    if not np.isfinite(posterior.logpost):
        raise RuntimeError(
            f"The Cobaya posterior is {posterior.logpost}; bfmt or CAMB "
            "rejected the declared feedback/cosmology point."
        )

    # ----------------------------------------------------------------------
    # Results: the xi rows of the scored vector and, with feedback on, the
    # response the likelihood applied (baryon_suppression_table reads R_P
    # at h * k for the h/Mpc table nodes).
    # ----------------------------------------------------------------------
    likelihood = model.likelihood[LIKELIHOOD]
    datavector = np.array(likelihood.last_datavector, copy=True)
    result = {
        "xi": datavector[likelihood.xi_indices],
        "response": None,
        "h": float(likelihood.h),
        "As": float(model.provider.get_param("As")),
        "likelihood": likelihood,
        "model": model,
    }
    if likelihood.external_baryon_suppression:
        result["response"] = likelihood.baryon_suppression_table()
    if "bfmt" in model.theory and model.theory["bfmt"].baryon_model == 4:
        result["bacco_coordinates"] = _bacco_coordinates(model=model, point=point)
    return result


def _bacco_coordinates(model, point):
    """Return the cold-matter BACCO coordinates of the bfmt evaluation.

    The paired diagnostic must use the same coordinates as the bfmt power
    response: cold-matter density omega_cold = Omega_m - Omega_nu (bfmt's
    own massive-neutrino convention, Omega_nu h^2 = m_nu/94.13 eV), the
    baryon density, and sigma8 of the cold field from the BACCO amplitude
    emulator that bfmt carries (sigma8 of total matter is not sigma8_cold).

    Arguments:
      model = evaluated Cobaya model whose bfmt block runs BACCO.
      point = the evaluated parameter point (the five BACCO coordinates
        come from the model's constant parameters).
    Returns:
      JSON-compatible dictionary of the eight BaccoColdResponses coordinates.
    """
    provider = model.provider
    h = provider.get_param("H0") / 100.0
    neutrino_mass = provider.get_param("mnu")
    omega_cold = provider.get_param("omegam") - neutrino_mass / 94.13 / h**2
    omega_baryon = provider.get_param("omegab")
    sigma = model.theory["bfmt"].baccoemulator.get_sigma8(
        cold=True,
        omega_cold=omega_cold,
        omega_baryon=omega_baryon,
        A_s=provider.get_param("As"),
        hubble=h,
        ns=provider.get_param("ns"),
        neutrino_mass=neutrino_mass,
        w0=provider.get_param("w"),
        wa=provider.get_param("wa"),
    )
    coordinates = {
        "omega_cold": float(omega_cold),
        "omega_baryon": float(omega_baryon),
        "sigma8_cold": float(np.asarray(sigma).item()),
    }
    constants = model.parameterization.constant_params()
    for name in ("M_c", "eta", "beta", "M1_z0_cen", "theta_inn"):
        coordinates[name] = float(constants[name + "_baccoemu"])
    return coordinates


def run_scenarios(information, point, catalog_path):
    """Compute every catalog bfmt scenario and its two-point response.

    For the gravity-only reference and then for each catalog entry, a
    Cobaya model is built from the example yaml on the xi-only dataset;
    the entries add the bfmt block and switch external_baryon_suppression
    on. The Map3 rows are refused by the likelihood (they are NaN in its
    data vector) and are not returned. Afterwards the reference model is
    evaluated again, with Cobaya's cache off, and its xi rows must equal
    the first ones bit for bit: no feedback state may survive in the
    native cosmolike globals that every model shares.

    Arguments:
      information = the example yaml (for example EXAMPLE_EVALUATE1.yaml)
        loaded with cobaya.yaml.yaml_load_file, sampler and output removed.
      point = its evaluate point (sampler.evaluate.override).
      catalog_path = YAML listing bfmt selectors, feedback points and papers.
    Returns:
      dictionary with rows (the 400 xi rows), reference (DMO xi), means
      [scenario, row], responses R_P[scenario, z, k], k [h/Mpc], z, names,
      records (one per scenario), tail_test (FLAMINGO xi with R_P held
      above TAIL_PIVOT h/Mpc, or None) and record (run provenance).
    Raises:
      ValueError for an unknown catalog format, a changed DMO restoration
      or a feedback parameter already in the example; RuntimeError when
      bfmt rejects a point or reports a warning. Physical model errors are
      propagated, never replaced by unity responses.
    Side effects:
      Builds 2 + len(scenarios) Cobaya models (plus one for the tail
      test), each running CAMB, BiHalofit and the xi projection once. No
      input file changes and no sampler or Fisher run starts.
    """
    # ----------------------------------------------------------------------
    # Catalog. Every bfmt model's weight files are checked before the
    # first model is built.
    # ----------------------------------------------------------------------
    catalog_path = Path(catalog_path).resolve()
    catalog = yaml.safe_load(catalog_path.read_text())
    if catalog.get("format") != CATALOG_FORMAT:
        raise ValueError(
            f"{catalog_path} has format {catalog.get('format')!r}; "
            f"run_scenarios reads {CATALOG_FORMAT!r} catalogs."
        )
    for scenario in catalog["scenarios"]:
        _require_installed_theory_data(model_number=scenario["options"]["baryon_model"])

    # ----------------------------------------------------------------------
    # Gravity-only reference on the xi-only dataset (feedback key off).
    # ----------------------------------------------------------------------
    reference_info = _scenario_information(
        information=information, scenario=None,
        theory_options=catalog["theory_options"],
    )
    reference_run = _evaluate(info=reference_info, point=point)
    reference_likelihood = reference_run["likelihood"]
    rows = reference_likelihood.xi_rows
    reference = reference_run["xi"]

    # ----------------------------------------------------------------------
    # Scenario loop: one model per catalog entry. FLAMINGO also runs the
    # tail test, in which only the continuation above the pivot changes.
    # ----------------------------------------------------------------------
    source = COCOA / "external_modules/code/baryon_suppression/bfmt.py"
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    names = []
    means = []
    responses = []
    records = []
    tail_test = None
    for scenario in catalog["scenarios"]:
        info = _scenario_information(
            information=information, scenario=scenario,
            theory_options=catalog["theory_options"],
        )
        run = _evaluate(info=info, point=point)
        names.append(scenario["name"])
        means.append(run["xi"])
        responses.append(run["response"])
        record = {
            "name": scenario["name"],
            "paper": scenario["paper"],
            "producer": "Cocoa bfmt baryon_suppression through des_3pcf.joint_shear",
            "supplied_spectra": ["P"],
            "source_sha256": source_sha256,
            "options": dict(catalog["theory_options"], **scenario["options"]),
            "feedback_parameters": dict(scenario["parameters"]),
            "h": run["h"],
            "As": run["As"],
            "bfmt_request_k_units": "1/Mpc, the table numbers 10^log10k",
            "likelihood_k_units": "h/Mpc; R_P read at h*k",
        }
        if "bacco_coordinates" in run:
            record["bacco_coordinates"] = run["bacco_coordinates"]
            record["bacco_sigma8_source"] = "bfmt Burger2025 sigma8_cold emulator"
            record["matter_field"] = "cold; applying to total P is an approximation"
        records.append(record)
        run["model"].close()

        if scenario["name"] == "FLAMINGO":
            tail_info = _scenario_information(
                information=information, scenario=scenario,
                theory_options=catalog["theory_options"], tail=True,
            )
            tail_run = _evaluate(info=tail_info, point=point)
            tail_test = tail_run["xi"]
            tail_run["model"].close()
        print(scenario["name"], "completed; all two-point rows:", run["xi"].size)

    # ----------------------------------------------------------------------
    # DMO restoration: the reference model again, without Cobaya's cache,
    # after every scenario model has changed the shared native state.
    # ----------------------------------------------------------------------
    reference_run["model"].logposterior(params_values=point, cached=False)
    restored = reference_likelihood.last_datavector[reference_likelihood.xi_indices]
    if not np.array_equal(restored, reference):
        raise ValueError(
            "The gravity-only xi rows changed after the scenarios; maximum "
            f"relative change {np.max(np.abs(restored / reference - 1.0))}."
        )
    reference_run["model"].close()

    # ----------------------------------------------------------------------
    # Result: the curves plus a record naming the inputs, so the saved npz
    # is traceable to its exact example yaml, catalog and dataset.
    # ----------------------------------------------------------------------
    return {
        "rows": rows,
        "reference": reference,
        "means": np.asarray(means),
        "responses": np.asarray(responses),
        "k": 10.0**reference_likelihood.log10k_interp_2D,
        "z": reference_likelihood.z_interp_2D.copy(),
        "names": names,
        "records": records,
        "tail_test": tail_test,
        "record": {
            "likelihood": LIKELIHOOD,
            "dataset": XI_DATASET,
            "dataset_sha256": hashlib.sha256(
                (PROJECT / "data" / XI_DATASET).read_bytes()).hexdigest(),
            "point": dict(point),
            "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
            "restoration_exact": True,
            "scope": "power-only scenario; no inferred bispectrum or feedback constraint",
            "flamingo_tail_test": f"hold R_P constant above {TAIL_PIVOT} h/Mpc",
        },
    }


def bacco_diagnostic(records):
    """Evaluate the installed paired models on two supported triangle families.

    Arguments:
      records = records from run_scenarios, including BACCO coordinates.
    Returns:
      k[80], power[2,80], bispectrum[2,2,80], and JSON-compatible provenance.
      Axis meanings are redshift [0,1], triangle [equilateral,scalene], k.
      Scalene sides are [0.6k,k,1.2k]. All are inside the weights' side box.
    Raises:
      StopIteration when the records contain no BACCO entry.
    Side effects:
      Loads two already-installed small emulators. No physical projection.
    """
    from theory.bacco_bispectrum.emulator import BaccoColdResponses

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
