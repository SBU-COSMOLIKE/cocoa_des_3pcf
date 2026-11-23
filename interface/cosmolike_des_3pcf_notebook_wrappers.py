"""Notebook wrappers for the des_3pcf CosmoLike interface.

Every EXAMPLE notebook of this project imports its helpers from this one
module, as the lsst_y1 notebooks import cosmolike_lsst_y1_notebook_wrappers:

    import cosmolike_des_3pcf_interface as ci
    import cosmolike_des_3pcf_notebook_wrappers as nw

Importing the module prepares no cosmology and starts no calculation. The
sections below serve one notebook each:

  1. IA operator decomposition (EXAMPLE_IA_COMPONENTS): splits the Map3
     prediction of an evaluated Cobaya likelihood into its gravity-only
     and intrinsic-alignment operators.
  2. bfmt feedback scenarios (EXAMPLE_FEEDBACK_SCENARIOS): evaluates the
     two-point rows of the Cobaya likelihood under each feedback model of
     configs/feedback_scenarios.yaml.
  3. Covariance teaching example (EXAMPLE_COVARIANCE_PME and
     EXAMPLE_COVARIANCE_MAPS): precision-matrix corrections on controlled
     Gaussian vectors, and the reader of external simulation vectors.

Each section opens with its own description and the terms it uses.
"""

import copy
import hashlib
import importlib
import importlib.util
import json
import logging
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_triangular
import yaml


# PROJECT = projects/des_3pcf (owns interface/, data/ and configs/);
# COCOA = the Cocoa root; CORE = the cosmolike core checkout, whose Python
# supplies cosmolike_notebook_utils.
PROJECT = Path(__file__).resolve().parents[1]
COCOA = PROJECT.parents[1]
CORE = COCOA / "external_modules/code/cosmolike_core"
sys.path.insert(0, str(CORE))

from cosmolike_notebook_utils.covariance.precision import component_precision_expansion
from cosmolike_notebook_utils.covariance.precision import (
    precision_expansion,
    split_precision_expansion,
)
from cosmolike_notebook_utils.covariance.precision import whiten_covariance

# ci = the compiled cosmolike library of this project; the notebook puts
# interface/ on sys.path before importing this file, and the likelihood
# imports the same module, so both share one extension and its state.
import cosmolike_des_3pcf_interface as ci


# Cobaya's name of the joint xi/Map3 likelihood in the example yamls.
LIKELIHOOD = "des_3pcf.joint_shear"


# ==========================================================================
# 1. IA operator decomposition (EXAMPLE_IA_COMPONENTS)
# ==========================================================================
#
# Split the Map3 prediction of an evaluated Cobaya likelihood into IA operators.
#
# The operator decomposition splits one aperture-statistic prediction
# (Map3, the third-order aperture-mass moment) into the separate physical
# operators that add up to it: the gravity-only term and each
# intrinsic-alignment (IA) term, further split by which of the three source
# legs carry an intrinsic shape. EXAMPLE_IA_COMPONENTS.ipynb plots each
# contribution alone.
#
# Every physical input comes from the Cobaya likelihood des_3pcf.joint_shear
# after one evaluation: the Map3 physics model that set_source_related built
# from the theory blocks (CAMB distances, growth and power, the BiHalofit
# matter bispectrum, the shifted source n(z)), the IA record of the same
# point, the shear calibrations and the likelihood's own aperture settings.
# This file adds no physics; it selects rows and calls the compiled
# decomposition.
# ==========================================================================


def aperture_components(cobaya_model, triplets, velocity_shear=0.0):
    """Decompose the Map3 rows of selected source triples at the evaluated point.

    The likelihood's aperture engine (likelihood.map_engine) is a transform
    plan for all of its Map3 rows, and the compiled plan offers no way to
    evaluate a subset of the rows it was built for. The decomposition
    repeats the full aperture integral once per operator and placement
    (40 times, one more for the total), so on the full plan it would
    project every source triple of the survey 41 times. This function
    therefore builds a second plan with the same constructor and the same
    settings as the likelihood's initialize, restricted to the requested
    rows. A plan projects each source triple on its own, so each selected
    row gets the value it has in the full plan; the returned "prediction"
    (the likelihood's own Map3 rows) lets the caller check that.

    The IA dictionary is the likelihood's own record of this point
    (amplitudes a1, eta1, a2, eta2, bta, normalization c1rho and pivot
    zpivot, power_mode, fb and feedback_amplitude), with two keys
    replaced in a copy: ct, the requested velocity-shear amplitude, and
    ct_mode = 0, under which the physical amplitude is Ct = C1 * ct, with
    C1 the linear tidal-alignment coefficient (the convention of fastnc,
    the external three-point natural-components code this project
    compares against). The likelihood itself always runs with ct = 0.
    The shear calibrations (1 + m_i) of the evaluated point multiply every
    component, three factors per row, as they do in the likelihood.

    Arguments:
      cobaya_model = Cobaya model (from cobaya.model.get_model) of a
        numbered example yaml that contains des_3pcf.joint_shear, already
        evaluated once (for example with logposterior at the yaml's
        sampler.evaluate.override point); a TATT example (EXAMPLE_EVALUATE2)
        gives every operator a nonzero amplitude.
      triplets = requested zero-based source triples, integer array of
        shape [ntriplet, 3]; [0, 2, 3] is the triple labelled 134.
      velocity_shear = the velocity-shear amplitude Ct in units of C1
        (ct_mode = 0 above); zero gives TATT without the additional
        velocity-shear operator.

    Returns:
      dictionary with
        components = the compiled result: "operators" (5 names),
          "placements" (8 names, GGG ... III), "values" [5, 8, nrow],
          "total" [nrow] (the complete calculation, evaluated separately)
          and "closure" [nrow] (sum of values minus total);
        rows = the selected SurveyRows (the labeled row table: one label,
          row kind, bin indices and angle per data-vector entry), in the
          likelihood's row order;
        indices = positions of those rows in likelihood.last_datavector;
        prediction = likelihood.last_datavector at those positions, the
          Map3 rows the likelihood scored (computed with ct = 0);
        ia = the IA dictionary passed to the decomposition.

    Raises:
      ValueError when the model has no des_3pcf.joint_shear likelihood,
      the likelihood has not been evaluated, the triples are not an
      integer [ntriplet, 3] array, velocity_shear is not finite, or a
      requested triple has no Map3 row.

    Side effects:
      Builds one transform plan and runs 41 aperture projections with
      the likelihood's OpenMP thread count. The likelihood's own state
      (its model, IA record, plan and last_datavector) is read, never
      changed, and no file is written.
    """
    # ----------------------------------------------------------------------
    # Validation, before any expensive setup. cobaya_model.likelihood is
    # Cobaya's {name: likelihood object} table. set_source_related stores
    # model, ia and calibration, and logp stores last_datavector, in the
    # same evaluation, so their presence means one complete evaluation.
    # ----------------------------------------------------------------------
    if LIKELIHOOD not in cobaya_model.likelihood:
        raise ValueError(
            f"The Cobaya model has no {LIKELIHOOD} likelihood; build it "
            "from a numbered EXAMPLE_EVALUATE yaml of des_3pcf.")
    likelihood = cobaya_model.likelihood[LIKELIHOOD]
    for name in ("model", "ia", "calibration", "last_datavector"):
        if not hasattr(likelihood, name):
            raise ValueError(
                f"The likelihood has no {name} yet; evaluate the model once "
                "(cobaya_model.logposterior(point)) before decomposing it.")
    requested = np.asarray(triplets)
    if (requested.ndim != 2 or requested.shape[1] != 3
            or requested.dtype.kind not in "iu"):
        raise ValueError(
            f"triplets has shape {requested.shape} and dtype {requested.dtype}; "
            "supply an integer array of shape [ntriplet, 3].")
    if not np.isfinite(velocity_shear):
        raise ValueError(
            f"velocity_shear = {velocity_shear}; supply a finite number.")

    # ----------------------------------------------------------------------
    # Row selection: keep exactly the Map3 rows whose bin triple matches a
    # requested one; a triple with no matching row is an error, not an
    # empty result. selected is a True/False flag per data-vector row;
    # flatnonzero turns it into the row positions, in the likelihood order.
    # ----------------------------------------------------------------------
    rows = likelihood.rows
    selected = np.zeros(shape=rows.values.size, dtype=bool)
    for triplet in requested:
        match = (rows.kind == "map3") & np.all(rows.bins == triplet, axis=1)
        if not np.any(match):
            raise ValueError(
                f"No Map3 row has the source triple {triplet}; the bins run "
                f"from 0 to {likelihood.source_ntomo - 1}.")
        selected |= match
    indices = np.flatnonzero(selected)
    rows = rows.select(indices=indices)

    # ----------------------------------------------------------------------
    # The transform plan of the selected rows, with the same arguments as
    # the likelihood's initialize. Angles convert from arcminutes to
    # radians (pi / (180 * 60)); the side-bin edges are log-spaced between
    # theta_min_arcmin and theta_max_arcmin, and the opening angle phi is
    # sampled at the centres of phi_bins equal cells of [0, pi].
    # ----------------------------------------------------------------------
    theta_edges = np.exp(np.linspace(
        start=np.log(likelihood.theta_min_arcmin),
        stop=np.log(likelihood.theta_max_arcmin),
        num=likelihood.ntheta + 1,
    )) * np.pi / (180.0 * 60.0)
    dphi = np.pi / likelihood.phi_bins
    plan = ci.production.TransformPlan.for_aperture(
        bins=np.ascontiguousarray(
            a=rows.bins,
            dtype=np.int32,
        ),
        filters=np.ascontiguousarray(a=rows.angles),
        theta_lower=theta_edges[:-1],
        phi=(np.arange(likelihood.phi_bins) + 0.5) * dphi,
        dlnt=float(np.log(theta_edges[1] / theta_edges[0])),
        dphi=dphi,
        ell_min=likelihood.map_ell_min,
        ell_max=likelihood.map_ell_max,
        minimum_size=likelihood.minimum_fft_size,
        lmax=likelihood.map_lmax,
        mmax=likelihood.map_mmax,
        nmu=likelihood.map_nmu,
        transform_options=likelihood.transform_options,
        projection_mode="prepared",
        projection_settings=likelihood.projection_settings,
        projection_batch_size=likelihood.projection_batch_size,
        accuracy=likelihood.accuracy,
    )

    # ----------------------------------------------------------------------
    # Components. dict(likelihood.ia, ct=..., ct_mode=0) copies the
    # likelihood's IA record and overrides two keys in the copy (the
    # record itself stays as the likelihood scored it). The compiled
    # aperture_components returns one curve per operator and placement,
    # the separately evaluated total, and their difference (closure).
    # ----------------------------------------------------------------------
    ia = dict(likelihood.ia, ct=float(velocity_shear), ct_mode=0)
    ci.set_omp_threads(n=likelihood.omp_threads)
    components = plan.aperture_components(
        model=likelihood.model,
        ia=ia,
        calibration=likelihood.calibration,
    )
    return {
        "components": components,
        "rows": rows,
        "indices": indices,
        "prediction": np.array(likelihood.last_datavector[indices], copy=True),
        "ia": ia,
    }


# ==========================================================================
# 2. bfmt feedback scenarios (EXAMPLE_FEEDBACK_SCENARIOS)
# ==========================================================================
#
# Run the bfmt feedback scenarios of EXAMPLE_FEEDBACK_SCENARIOS through Cobaya.
#
# The notebook compares the change in all 400 DES Y3 two-point rows at one
# cosmology when baryonic feedback modifies the nonlinear matter power. Every
# prediction comes from the project's Cobaya likelihood des_3pcf.joint_shear,
# built with cobaya.model.get_model from a numbered EXAMPLE yaml.
#
# Terms used in this file:
#   bfmt = Cocoa's baryonic-feedback Cobaya theory block (the repository
#       baryon_suppression); given a feedback model and its parameters it
#       returns R_P = P_feedback/P_DMO, the ratio of the matter power
#       spectrum with baryonic feedback to the gravity-only one.
#   DMO = dark-matter-only: the power spectrum without baryonic feedback,
#       the reference every response ratio divides by.
#   R_P = that power-spectrum response ratio, tabulated over [z, k].
#   external_baryon_suppression = the likelihood yaml key that requests R_P
#       from bfmt and multiplies the nonlinear power of the xi rows by it.
#       A power-only model supplies no bispectrum response, so the
#       likelihood then refuses every Map3 row; the scenarios therefore use
#       data/des_y3_xi1.dataset, whose mask cuts all Map3 rows.
#   scenario catalog = configs/feedback_scenarios.yaml: one entry per bfmt
#       model with its selector options, fixed feedback parameters and paper.
#   FLAMINGO = a hydrodynamical simulation suite (one of the bfmt feedback
#       models); BACCO = an emulator calibrated on simulations, whose paired
#       diagnostic below evaluates its power and bispectrum responses
#       together, from its calibrated cold-matter models.
#
# Each scenario changes only the nonlinear matter power, as the LSST Y1
# feedback example does. Intrinsic-alignment loops keep their linear DMO
# input, and no three-point response is inferred from these power-only
# scenarios.
# ==========================================================================

# The dataset whose mask keeps the 400 xi rows and cuts all 80 Map3 rows.
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


# ==========================================================================
# 3. Covariance teaching example (EXAMPLE_COVARIANCE_PME and _MAPS)
# ==========================================================================
#
# Teach covariance corrections with controlled joint Gaussian measurements.
#
# The example is self-contained: the vectors are synthetic Gaussian draws
# with a covariance prescribed in this file, so the exact precision matrix
# and the exact truncation error of every estimator are known in closed
# form. Two-point-like and three-point-like rows get different units and
# correlated fluctuations on purpose, to exercise the same mixed-unit
# bookkeeping as the survey vector. The same precision.py functions used by
# the forecast are called directly; no compiled model, CAMB calculation,
# download or file is needed. Loading external simulation vectors is the
# separate explicit call at the end of the module.
#
# Terms used in this file:
#   precision matrix = the inverse covariance, the matrix that weights the
#       chi2 and the parameter fit.
#   PME = the precision-matrix expansion of Friedrich and Eifler (2018,
#       arXiv:1703.07786): instead of inverting a noisy sample covariance,
#       expand the precision around the inverse of an analytic model
#       covariance M; only M is ever inverted.
#   Wishart correction = the sample covariance of N independent Gaussian
#       draws follows a Wishart distribution, so products of sample
#       covariances are biased; the second-order PME term subtracts exactly
#       that bias (precision.py holds the formula).
#   whitening = the change of coordinates d -> L^-1 d with C = L L^T, after
#       which the covariance is the identity (each covariance eigenvector
#       rescaled by one over the square root of its eigenvalue); errors and
#       ranks measured there are dimensionless and unit-free.
#   coverage = the fraction of repeated experiments whose quoted error
#       interval actually contains the truth; a quoted sigma without
#       coverage can be too optimistic.
# ==========================================================================


def controlled_joint_covariance(ndata=24):
    """Specify C=A+B and an imperfect independent model M=A+Bm.

    A is a diagonal known random contribution and B a dense contribution
    with correlations both within and between the two-point-like and
    three-point-like row groups. The controlled experiment draws the A and
    B random vectors independently, so the total covariance really is the
    sum of the two prescribed pieces and every estimator can be compared
    with the exact answer.

    Arguments:
      ndata = even integer at least 8, split equally into the two row groups.
    Returns:
      Dictionary with true/analytic/component covariances, row labels and a
      three-column illustrative derivative matrix. The row scales 1e-4 and
      1e-8 only expose mixed units; they are not calibrated survey variances.
    Raises:
      ValueError for an invalid dimension.
    """
    # isinstance(ndata, bool) is checked separately because Python booleans
    # are integers (True == 1) and would otherwise slip through.
    if (isinstance(ndata, bool) or int(ndata) != ndata
            or ndata < 8 or ndata % 2):
        raise ValueError("ndata must be an even integer at least 8.")
    ndata = int(ndata)

    # ----------------------------------------------------------------------
    # Dimensionless shapes. B = 0.12 * exp(-|i-j|/3) + 0.025: a short
    # correlation length of 3 rows at amplitude 0.12, plus a small 0.025
    # offset correlating every pair of rows (including the two groups).
    # A is diagonal with variances 0.8 + 0.1 cos(i/4): order one, slowly
    # varying, never close to zero. The model Bm = 0.8 B + 0.01 I is
    # deliberately imperfect: 20 percent low, with a small spurious
    # diagonal.
    # ----------------------------------------------------------------------
    coordinate = np.arange(ndata, dtype=np.float64)
    separation = np.abs(coordinate[:, None] - coordinate[None, :])
    component = 0.12 * np.exp(-separation / 3.0)
    component += 0.025 * np.ones((ndata, ndata))
    known = np.diag(0.8 + 0.1 * np.cos(coordinate / 4.0))
    component_model = 0.8 * component + 0.01 * np.eye(ndata)

    # ----------------------------------------------------------------------
    # Units. The first half of the rows plays the two-point group at scale
    # 1e-4 and the second half the three-point group at scale 1e-8, four
    # orders of magnitude apart, so any estimator that mishandles mixed
    # units fails visibly.
    # ----------------------------------------------------------------------
    scales = np.full(ndata, 1.0e-4)
    scales[ndata // 2:] = 1.0e-8
    units = scales[:, None] * scales[None, :]
    component *= units
    known *= units
    component_model *= units

    # ----------------------------------------------------------------------
    # Three prescribed summary responses, not derivatives of a shear
    # model: a constant offset, two smooth oscillations, and an extra
    # response of the three-point group alone in the last column, so the
    # parameters couple the two groups differently.
    # ----------------------------------------------------------------------
    derivative = np.empty((ndata, 3))
    derivative[:, 0] = scales
    derivative[:, 1] = scales * np.cos(coordinate / 4.0)
    derivative[:, 2] = scales * np.sin(coordinate / 3.0)
    derivative[ndata // 2:, 2] += scales[ndata // 2:]

    # ----------------------------------------------------------------------
    # Row labels in the survey convention: kind "xip" for the two-point
    # half, "map3" for the three-point half.
    # ----------------------------------------------------------------------
    kinds = np.full(ndata, "map3", dtype="U4")
    kinds[:ndata // 2] = "xip"
    labels = []
    for index, kind in enumerate(kinds):
        labels.append(f"controlled_{kind}_{index:03d}")

    return {
        "known": known,
        "component": component,
        "component_model": component_model,
        "truth": known + component,
        "analytic": known + component_model,
        "derivatives": derivative,
        "row_labels": np.asarray(labels),
        "row_kind": kinds,
        "parameter_names": ["amplitude", "alignment", "shape"],
        "scope": "Controlled Gaussian joint vectors; not DES simulations.",
    }


def draw_vectors(covariance, count, generator):
    """Generate independent zero-mean vectors with an explicit covariance.

    With C = L L^T (Cholesky) and n a standard normal vector, L n has
    covariance C exactly; multiplying by L preserves the intentionally
    different observable units without an SVD rank threshold.

    Arguments:
      covariance = positive definite [ndata,ndata] matrix.
      count = number of realizations; generator = supplied NumPy Generator.
    Returns:
      Float64 vectors[count,ndata].
    Raises:
      numpy.linalg.LinAlgError (from the Cholesky factorization) when the
      covariance is not positive definite.
    """
    factor = np.linalg.cholesky(covariance)
    normal = generator.standard_normal(size=(count, covariance.shape[0]))
    return normal @ factor.T


def covariance_of_vectors(vectors):
    """Subtract the estimated mean and divide by N-1.

    Arguments:
      vectors = finite [N,ndata] independent realizations, N>=2.
    Returns:
      Unbiased sample covariance. Rank deficiency is allowed and no inverse
      is attempted; after mean removal its rank is at most N-1.
    Raises:
      ValueError for invalid arrays or insufficient realizations.
    """
    values = np.asarray(vectors, dtype=np.float64)
    if (values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0
            or not np.all(np.isfinite(values))):
        raise ValueError("Vectors must be finite [N,ndata] with N>=2.")
    centered = values - np.mean(values, axis=0)
    return centered.T @ centered / (values.shape[0] - 1)


def precision_diagnostics(precision, truth, derivative, validation):
    """Compare a supplied precision with known truth and held-out vectors.

    All comparisons happen in true-whitened coordinates (module
    docstring): with the true covariance factored as C = Lc Lc^T, a
    perfect precision estimate turns Lc^T Psi Lc into the identity, so
    the eigenvalues of Lc^T Psi Lc - I are dimensionless errors.

    Arguments:
      precision = symmetric candidate [ndata,ndata], possibly indefinite.
      truth = known true covariance; derivative = [ndata,nparameter].
      validation = independent zero-mean truth draws, unused in preparation.
    Returns:
      Eigenvalues of Lc.T precision Lc-I and its relative Frobenius error.
      Positive candidates also yield parameter-error and validation statistics.
      Indefinite candidates are retained as failures, without eigenvalue repair.
    """
    # ----------------------------------------------------------------------
    # Whitened error: symmetrize against rounding, then read off the
    # eigenvalues. A smallest eigenvalue at or below zero means the
    # candidate is not a valid precision matrix.
    # ----------------------------------------------------------------------
    factor = np.linalg.cholesky(truth)
    normalized = factor.T @ precision @ factor
    normalized = 0.5 * (normalized + normalized.T)
    eigenvalues = np.linalg.eigvalsh(normalized)
    error = normalized - np.eye(truth.shape[0])
    result = {
        "positive": bool(eigenvalues[0] > 0.0),
        "minimum_precision_eigenvalue_in_true_units": float(eigenvalues[0]),
        "precision_error_eigenvalues": eigenvalues - 1.0,
        "relative_precision_frobenius_error": float(
            np.linalg.norm(error) / np.sqrt(truth.shape[0])
        ),
    }
    if not result["positive"]:
        result["inference_status"] = "refused_nonpositive_precision"
        return result

    # ----------------------------------------------------------------------
    # Linear parameter fit weighted by the candidate precision: the
    # information matrix D^T Psi D, its inverse as the quoted parameter
    # covariance, and the explicit least-squares estimator matrix.
    # ----------------------------------------------------------------------
    information = derivative.T @ precision @ derivative
    parameter_covariance = np.linalg.solve(
        information, np.eye(information.shape[0]),
    )
    estimator = parameter_covariance @ derivative.T @ precision

    # ----------------------------------------------------------------------
    # Validation: fit every held-out draw and compare the scatter of the
    # fitted parameters with the quoted covariance. The exact frequentist
    # covariance of the linear estimator depends on the true C, not on the
    # covariance whose inverse was used to weight the fit, hence the
    # sandwich product below.
    # ----------------------------------------------------------------------
    fitted = validation @ estimator.T
    empirical = covariance_of_vectors(vectors=fitted)
    sandwich = estimator @ truth @ estimator.T
    result.update({
        "inference_status": "positive_precision",
        "parameter_covariance": parameter_covariance,
        "exact_fit_covariance": sandwich,
        "validation_fit_covariance": empirical,
        "sigma": np.sqrt(np.diag(parameter_covariance)),
        "validation_variance_over_quoted": (
            np.diag(empirical) / np.diag(parameter_covariance)
        ),
        "exact_variance_over_quoted": (
            np.diag(sandwich) / np.diag(parameter_covariance)
        ),
        # The einsum operands read: for each validation vector n, the
        # quadratic form sum_ij validation[n,i] precision[i,j]
        # validation[n,j] — one chi2-like number per vector. Its mean per
        # row is 1 for a perfect precision.
        "validation_mean_quadratic_per_row": float(
            np.mean(np.einsum("ni,ij,nj->n", validation, precision, validation))
            / truth.shape[0]
        ),
    })
    return result


def controlled_precision_study(ndata=24, training_count=18,
                               validation_count=1200, seed=1937):
    """Compare full and component PME with one fixed, independent model.

    Training contains N independent A and B vectors; their sum samples C.
    The B-only sample uses those same N B vectors, so comparisons share the
    available physical realizations. Two disjoint halves supply independent
    B covariance estimates for the split-group polynomial. Validation uses
    a separately seeded stream of fresh joint vectors and never tunes M.

    Arguments:
      ndata = even vector length; training_count = even N>=6 and N<=ndata.
      validation_count = integer number of held-out draws at least 100.
      seed = integer entropy for separate training/validation generators.
    Returns:
      Known matrices, samples, component/full/split estimates, positivity,
      true and estimated residual radii, truncation and validation diagnostics.
    Raises:
      ValueError for invalid counts; no repeated draw selects a nicer result.
    """
    # ----------------------------------------------------------------------
    # Validation. The bool checks keep True/False out (Python booleans are
    # integers); the lower limits keep every sample covariance meaningful.
    # ----------------------------------------------------------------------
    if (isinstance(training_count, bool)
            or int(training_count) != training_count
            or training_count < 6 or training_count % 2
            or training_count > ndata):
        raise ValueError("Use even training_count from 6 through ndata.")
    if (isinstance(validation_count, bool)
            or int(validation_count) != validation_count
            or validation_count < 100):
        raise ValueError("validation_count must be an integer at least 100.")

    # ----------------------------------------------------------------------
    # Random streams. The seed is an arbitrary fixed integer so the
    # notebook is reproducible; SeedSequence.spawn gives two independent
    # child streams, one for training draws and one for validation draws,
    # so changing one count never shifts the other stream.
    # ----------------------------------------------------------------------
    system = controlled_joint_covariance(ndata=ndata)
    streams = np.random.SeedSequence(entropy=seed).spawn(2)
    training_rng = np.random.default_rng(seed=streams[0])
    validation_rng = np.random.default_rng(seed=streams[1])

    # ----------------------------------------------------------------------
    # Draws: N component (B) vectors and N known (A) vectors; their sums
    # are the joint training vectors. The validation draws come from the
    # full true covariance on the separate stream.
    # ----------------------------------------------------------------------
    component_vectors = draw_vectors(
        covariance=system["component"], count=int(training_count),
        generator=training_rng,
    )
    known_vectors = draw_vectors(
        covariance=system["known"], count=int(training_count),
        generator=training_rng,
    )
    joint_vectors = component_vectors + known_vectors
    validation = draw_vectors(
        covariance=system["truth"], count=int(validation_count),
        generator=validation_rng,
    )

    # ----------------------------------------------------------------------
    # Sample covariances: the full joint sample, the B-only sample, and
    # the two disjoint half-samples feeding the split-group estimator.
    # ----------------------------------------------------------------------
    sample_component = covariance_of_vectors(vectors=component_vectors)
    sample_joint = covariance_of_vectors(vectors=joint_vectors)
    half = int(training_count) // 2
    sample_first = covariance_of_vectors(vectors=component_vectors[:half])
    sample_second = covariance_of_vectors(vectors=component_vectors[half:])

    # ----------------------------------------------------------------------
    # Exact references in model-whitened coordinates. The residual is the
    # whitened truth minus the identity; its spectral radius (the largest
    # absolute eigenvalue) controls how fast the PME converges. The
    # second-order polynomial I - R + R^2, evaluated with the exact
    # residual, gives the best answer any second-order estimator could
    # reach; the exact truth precision is the overall reference.
    # ----------------------------------------------------------------------
    model = system["analytic"]
    factor = np.linalg.cholesky(model)
    normalized_truth = whiten_covariance(
        covariance=system["truth"], factor=factor,
    )
    residual = normalized_truth - np.eye(ndata)
    true_radius = float(np.max(np.abs(np.linalg.eigvalsh(residual))))
    inverse_factor = solve_triangular(factor, np.eye(ndata), lower=True)
    analytic_precision = inverse_factor.T @ inverse_factor
    truth_precision = np.linalg.solve(system["truth"], np.eye(ndata))
    polynomial = np.eye(ndata) - residual + residual @ residual
    truncated = inverse_factor.T @ polynomial @ inverse_factor
    estimates = {
        "truth": truth_precision,
        "analytic": analytic_precision,
        "exact_second_order": truncated,
    }

    # ----------------------------------------------------------------------
    # The three estimators under study: the component PME (known A plus a
    # Wishart-corrected B sample), the full-covariance PME (one
    # Wishart-corrected joint sample), and the split-group polynomial
    # (two independent half-samples instead of a Wishart correction).
    # ----------------------------------------------------------------------
    estimator_diagnostics = {}
    estimates["component_wishart"], estimator_diagnostics["component"] = (
        component_precision_expansion(
            analytic=model, component_model=system["component_model"],
            component_sample=sample_component,
            independent_count=training_count,
        )
    )
    estimates["full_wishart"], estimator_diagnostics["full"] = (
        precision_expansion(
            analytic=model, sample=sample_joint,
            independent_count=training_count,
        )
    )
    estimates["component_split"] = split_precision_expansion(
        analytic=model, sample_first=system["known"] + sample_first,
        sample_second=system["known"] + sample_second,
    )

    # ----------------------------------------------------------------------
    # Diagnostics for every estimate against the known truth and the
    # held-out validation draws.
    # ----------------------------------------------------------------------
    diagnostics = {}
    for name, precision in estimates.items():
        diagnostics[name] = precision_diagnostics(
            precision=precision, truth=system["truth"],
            derivative=system["derivatives"], validation=validation,
        )
    normalized_sample = whiten_covariance(
        covariance=sample_joint, factor=factor,
    )

    return {
        "system": system,
        "sample_joint": sample_joint,
        "sample_component": sample_component,
        "estimates": estimates,
        "diagnostics": diagnostics,
        "estimator_diagnostics": estimator_diagnostics,
        "true_residual_spectral_radius": true_radius,
        "true_second_order_operator_error_bound": true_radius**3
        / (1.0 - true_radius),
        "sample_rank_in_model_units": int(
            np.linalg.matrix_rank(normalized_sample)
        ),
        "training_count": int(training_count),
        "validation_count": int(validation_count),
        "seed": seed,
        "training_stream": streams[0].spawn_key,
        "validation_stream": streams[1].spawn_key,
        "sample_inverse_attempted": False,
    }


def plot_precision_errors(report):
    """Draw dimensionless precision errors, including failed estimates.

    Arguments:
      report = controlled_precision_study result.
    Returns:
      Shared figure/axes. Each line sorts the eigenvalues of Lc.T Psi Lc-I;
      sorted positions are not identifications of shared physical eigenvectors.
    """
    # The plotting module is named cosmolike_notebook_utils.3pcf: a module
    # name starting with a digit cannot appear in an import statement, so
    # importlib.import_module loads it from its name as text.
    plots = importlib.import_module(name="cosmolike_notebook_utils.3pcf")
    names = (
        "analytic",
        "component_wishart",
        "component_split",
        "full_wishart",
    )
    labels = [
        "Analytic model",
        "Known A + Wishart B",
        "Known A + independent B groups",
        "Full-covariance Wishart",
    ]
    curves = []
    coordinate = np.arange(report["system"]["truth"].shape[0]) + 1
    for name in names:
        errors = report["diagnostics"][name]["precision_error_eigenvalues"]
        curves.append((coordinate, errors[None, :]))
    return plots.plot_ia_components(
        components=curves, component_labels=["Precision error"],
        coordinate_label="Sorted eigenvalue index",
        quantity_label=r"Eigenvalues of $L_C^{\mathsf{T}}\hat{\Psi} L_C-I$",
        legend=labels, xscale="linear", ncols=1, figsize=(10, 5),
        bintextpos=None, colorbar=None, show=None,
    )


def plot_precision_parameter_errors(report):
    """Plot quoted parameter errors only for positive precision estimates.

    A positive precision alone does not establish coverage (module
    docstring); the validation variance ratios of the report are the
    separate check of the quoted errors.

    Arguments:
      report = controlled_precision_study result with known true precision.
    Returns:
      Shared figure/axis showing sigma/sigma_true.
    """
    # importlib.import_module loads the digit-led module name (see
    # plot_precision_errors).
    plots = importlib.import_module(name="cosmolike_notebook_utils.3pcf")
    errors = []
    labels = []
    methods = (
        ("analytic", "Analytic model"),
        ("component_wishart", "Known A + Wishart B"),
        ("component_split", "Known A + independent B groups"),
        ("full_wishart", "Full-covariance Wishart"),
    )
    for name, label in methods:
        diagnostics = report["diagnostics"][name]
        if diagnostics["positive"]:
            errors.append(diagnostics["sigma"])
            labels.append(label)
        else:
            print(f"No parameter contour for {label}: nonpositive precision.")
    return plots.plot_parameter_errors(
        names=report["system"]["parameter_names"], errors=errors,
        errors_ref=report["diagnostics"]["truth"]["sigma"],
        legend=labels, figsize=(11, 5), show=None,
    )


def load_joint_simulations(path, expected_labels, expected_kind,
                           expected_estimator, excluded_parent_ids=()):
    """Read explicitly labeled independent joint vectors without reordering.

    The NPZ must contain vectors[N,p], row_labels[p], row_kind[p],
    parent_ids[N], and scalar estimator. A parent ID names the independent
    sky realization a vector came from; each must occur exactly once.
    Multiple patches sharing a parent are refused, because averaging them
    here would change the estimator and its covariance. Unique IDs alone
    do not prove independence — that is a scientific input declaration,
    and the caller must establish how the survey realizations were built.

    Arguments:
      path = an explicit NPZ file, read with allow_pickle=False.
      expected_labels, expected_kind = exact ordered row identities of the
          analytic covariance; each group must contain xi and Map3 rows.
      expected_estimator = declared estimator string, e.g. finite_sas (the
          aperture estimator built from finite shear aperture statistics;
          the pipeline's other choice is full_filter). A family name does
          not verify triangle cuts, weights, masks or source calibration;
          compare their input/measurement manifests separately before
          interpreting a survey correction.
      excluded_parent_ids = parents used for analytic tuning or validation;
          any overlap is refused before a sample covariance is formed.
    Returns:
      Owned vectors, labels, kinds, parent IDs and independent count.
    Raises:
      ValueError for shape, finite-value, row-order, estimator or parent
      errors.
    """
    # ----------------------------------------------------------------------
    # Read the required arrays; a missing name fails before anything is
    # interpreted. The copies detach the results from the open archive.
    # ----------------------------------------------------------------------
    filename = Path(path)
    required = (
        "vectors",
        "row_labels",
        "row_kind",
        "parent_ids",
        "estimator",
    )
    arrays = {}
    with np.load(filename, allow_pickle=False) as archive:
        for name in required:
            if name not in archive.files:
                raise ValueError(f"Simulation archive lacks {name}.")
            arrays[name] = archive[name].copy()

    # ----------------------------------------------------------------------
    # Vector and row checks: finite numeric vectors, row labels and kinds
    # matching the analytic covariance exactly, and both row groups
    # present.
    # ----------------------------------------------------------------------
    vectors = arrays["vectors"]
    if (vectors.dtype.kind not in "fiu" or vectors.ndim != 2
            or vectors.shape[0] < 3
            or not np.all(np.isfinite(vectors))):
        raise ValueError("Need finite independent vectors[N,p] with N>2.")
    labels = arrays["row_labels"]
    kinds = arrays["row_kind"]
    parents = arrays["parent_ids"]
    if (labels.shape != (vectors.shape[1],)
            or kinds.shape != labels.shape
            or parents.shape != (vectors.shape[0],)):
        raise ValueError("Simulation rows or parent IDs have wrong shapes.")
    if (not np.array_equal(labels, expected_labels)
            or not np.array_equal(kinds, expected_kind)):
        raise ValueError("Simulation rows do not match the analytic matrix.")
    if (np.unique(labels).size != labels.size
            or np.any(~np.isin(kinds, ("xip", "xim", "map3")))
            or not np.any(kinds == "map3")
            or not np.any(kinds != "map3")):
        raise ValueError("Require unique labeled joint xi and Map3 rows.")

    # ----------------------------------------------------------------------
    # Estimator and parent checks: one declared estimator string matching
    # the analytic side, and parent IDs that are nonempty, unique and
    # disjoint from the tuning/validation parents.
    # ----------------------------------------------------------------------
    if arrays["estimator"].shape != ():
        raise ValueError("Simulation estimator must be one declared string.")
    if str(arrays["estimator"].item()) != expected_estimator:
        raise ValueError("Simulation and analytic estimators differ.")
    if parents.dtype.kind not in "iuUS":
        raise ValueError("Parent IDs must be integer or text identifiers.")
    if parents.dtype.kind in "US" and np.any(parents == ""):
        raise ValueError("Parent IDs must not be empty text identifiers.")
    if np.unique(parents).size != parents.size:
        raise ValueError(
            "Repeated parent IDs: patches are not independent skies."
        )
    if np.intersect1d(parents, np.asarray(excluded_parent_ids)).size:
        raise ValueError("Training parents overlap tuning or validation.")

    return {
        "vectors": vectors.copy(),
        "row_labels": labels.copy(),
        "row_kind": kinds.copy(),
        "parent_ids": parents.copy(),
        "independent_count": int(vectors.shape[0]),
        "estimator": expected_estimator,
    }
