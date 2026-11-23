"""Split the Map3 prediction of an evaluated Cobaya likelihood into IA operators.

The operator decomposition splits one aperture-statistic prediction
(Map3, the third-order aperture-mass moment) into the separate physical
operators that add up to it: the gravity-only term and each
intrinsic-alignment (IA) term, further split by which of the three source
legs carry an intrinsic shape. EXAMPLE_IA_COMPONENTS.ipynb plots each
contribution alone.

Every physical input comes from the Cobaya likelihood des_3pcf.joint_shear
after one evaluation: the Map3 physics model that set_source_related built
from the theory blocks (CAMB distances, growth and power, the BiHalofit
matter bispectrum, the shifted source n(z)), the IA record of the same
point, the shear calibrations and the likelihood's own aperture settings.
This file adds no physics; it selects rows and calls the compiled
decomposition.
"""

import numpy as np

# ci = the compiled cosmolike library of this project; the notebook puts
# interface/ on sys.path before importing this file, and the likelihood
# imports the same module, so both share one extension and its state.
import cosmolike_des_3pcf_interface as ci


# Cobaya's name of the joint xi/Map3 likelihood in the example yamls.
LIKELIHOOD_NAME = "des_3pcf.joint_shear"


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
    if LIKELIHOOD_NAME not in cobaya_model.likelihood:
        raise ValueError(
            f"The Cobaya model has no {LIKELIHOOD_NAME} likelihood; build it "
            "from a numbered EXAMPLE_EVALUATE yaml of des_3pcf.")
    likelihood = cobaya_model.likelihood[LIKELIHOOD_NAME]
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
