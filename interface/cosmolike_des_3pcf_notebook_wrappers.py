"""Notebook wrappers for the des_3pcf Cosmolike interface.

The observables of this project are the shear two-point correlation
functions xi_+ and xi_-, the third-order aperture-mass statistic Map3,
the joint data vector that holds both, and its chi2. This module computes
them at any parameter point, so a notebook imports them instead of
redefining them (EXAMPLE_IA_COMPONENTS does):

    import cosmolike_des_3pcf_notebook_wrappers as nw
    nw.configure()                      # EXAMPLE_EVALUATE1.yaml, one thread
    theta, xi_plus, xi_minus = nw.xi()  # at the project fiducial
    radii, triplets, map3 = nw.map3(sigma8=0.82)
    nw.get_chi2(omegam=0.31)

Every value comes from the project's Cobaya likelihood: configure() builds
the Cobaya model of a numbered EXAMPLE yaml once (CAMB, the BiHalofit
matter bispectrum, the joint likelihood des_3pcf.joint_shear), and every
call evaluates that model at the requested point. Nothing here computes
physics on its own; to change the theory (for example to add baryonic
feedback), call configure() with the changed yaml blocks.

Use these functions wherever a notebook needs the observables at a few
parameter points: plotting a prediction against the measurement, accuracy
and component checks, or the model evaluations behind Fisher derivatives.
A sampling run does not come through here; it runs the same yaml with
cobaya-run.

Two layers of state matter here:

- The Cobaya model lives in this module (one at a time). configure()
  replaces it; every observable function evaluates it from scratch at
  its point (Cobaya's cache is bypassed), so no call depends on which
  function ran before it.
- The project fiducial point is the sampler.evaluate.override point of
  EXAMPLE_EVALUATE1.yaml, read once when the module is imported into the
  plain constants below (omegam, sigma8, M, ...). The functions use them
  as default argument values, which Python evaluates once, at import:
  assigning nw.omegam = 0.31 later does not change the defaults; pass
  omegam=0.31 to the call instead.

Parameter names follow the lsst_y1 wrappers where the physics is the
same: omegam, omegab, ns; A1 = [a1, eta1] and A2 = [a2, eta2] are the
tidal-alignment and tidal-torquing amplitudes and redshift slopes of the
TATT intrinsic-alignment model, BTA its density weighting; M = [m_1..m_4]
the multiplicative shear calibrations and shear_photoz_bias =
[dz_1..dz_4] the source photo-z shifts. des_3pcf samples h and sigma8
directly (lsst_y1 samples H0 and As_1e9).
"""

from pathlib import Path

import numpy as np
from cobaya.model import get_model
from cobaya.yaml import yaml_load_file


# ----------------------------------------------------------------------
# Paths and the compiled interface.
# ----------------------------------------------------------------------
# PROJECT = projects/des_3pcf; COCOA = the Cocoa root.
PROJECT = Path(__file__).resolve().parents[1]
COCOA = PROJECT.parents[1]

# ci = the compiled cosmolike library of this project; the notebook puts
# interface/ on sys.path before importing this file, and the likelihood
# imports the same module, so both share one extension and its state.
import cosmolike_des_3pcf_interface as ci

# Cobaya's name of the joint xi/Map3 likelihood in the example yamls.
LIKELIHOOD = "des_3pcf.joint_shear"

# Radians to arcminutes.
RADIAN_TO_ARCMIN = 180.0 * 60.0 / np.pi


# ----------------------------------------------------------------------
# The project fiducial point (EXAMPLE_EVALUATE1.yaml, sampler.evaluate).
# ----------------------------------------------------------------------
_FIDUCIAL = yaml_load_file(str(PROJECT / "EXAMPLE_EVALUATE1.yaml"))
_OVERRIDE = _FIDUCIAL["sampler"]["evaluate"]["override"]

omegam = _OVERRIDE["omega_m"]
omegab = _OVERRIDE["omega_b"]
h = _OVERRIDE["h"]
ns = _OVERRIDE["ns"]
sigma8 = _OVERRIDE["sigma8"]

# NLA at the fiducial: a2, eta2 and bta are fixed to zero in the yaml.
A1_FID = [_OVERRIDE["a1"], _OVERRIDE["eta1"]]
A2_FID = [_FIDUCIAL["params"]["a2"], _FIDUCIAL["params"]["eta2"]]
BTA_FID = _FIDUCIAL["params"]["bta"]
M_FID = [_OVERRIDE[f"m{i}"] for i in range(1, 5)]
SHEAR_PHOTOZ_FID = [_OVERRIDE[f"dz{i}"] for i in range(1, 5)]


# ----------------------------------------------------------------------
# The configured Cobaya model (one at a time).
# ----------------------------------------------------------------------
_STATE = {"model": None, "yaml": None}


def configure(yaml="EXAMPLE_EVALUATE1.yaml", likelihood=None, theory=None,
              params=None, threads=1):
    """Build the Cobaya model that every observable function evaluates.

    Call it once at the top of a notebook, and again whenever the theory
    or the dataset changes (it replaces the previous model).

    Arguments:
      yaml = a numbered example yaml of this project (EXAMPLE_EVALUATE1 is
        NLA, EXAMPLE_EVALUATE2 TATT; see the README for the others).
      likelihood = keys that replace the yaml's des_3pcf.joint_shear
        settings, for example {"data_file": "des_y3_xi1.dataset",
        "external_baryon_suppression": True}.
      theory = theory blocks to add or replace, {name: options}; None as
        options removes a block (for example {"bfmt": {...}} for baryonic
        feedback on the two-point rows).
      params = parameter entries to add or replace, in yaml form (a fixed
        number, or a dict with a prior), for example the parameters of an
        added theory block.
      threads = OpenMP workers of the compiled projection (the
        likelihood's omp_threads yaml key); the BLAS worker count is
        separate and belongs to the notebook's setup cell.
    Side effects:
      closes the previous model and builds the new one (CAMB and the
      theory blocks initialize; no point is evaluated yet).
    """
    info = yaml_load_file(str(PROJECT / yaml))
    info.pop("sampler", None)
    info.pop("output", None)

    # The yaml's paths are relative to Cocoa/; give CAMB an absolute one
    # so the notebook may run from any working directory.
    if "camb" in info["theory"]:
        info["theory"]["camb"]["path"] = str(COCOA / "external_modules/code/CAMB")

    info["likelihood"][LIKELIHOOD].update(likelihood or {})
    info["likelihood"][LIKELIHOOD]["omp_threads"] = threads
    for name, options in (theory or {}).items():
        if options is None:
            info["theory"].pop(name, None)
        else:
            info["theory"][name] = options
    info["params"].update(params or {})

    if _STATE["model"] is not None:
        _STATE["model"].close()
    _STATE["model"] = get_model(info_or_yaml_or_file=info)
    _STATE["yaml"] = yaml


def _point(omegam, omegab, h, ns, sigma8, A1, A2, BTA, shear_photoz_bias,
           M, extra):
    """Translate the wrapper arguments into the Cobaya point of the model.

    A Cobaya point is the {sampled parameter name: value} dictionary that
    logposterior evaluates. Only the sampled parameters of the configured
    model enter the point.
    A parameter the yaml fixes must be requested at its fixed value: a
    different value raises, because the model could not honour it.
    """
    model = _STATE["model"]
    if model is None:
        raise ValueError("Call nw.configure() before computing an observable.")

    requested = {
        "omega_m": omegam,
        "omega_b": omegab,
        "h": h,
        "ns": ns,
        "sigma8": sigma8,
        "a1": A1[0],
        "eta1": A1[1],
        "a2": A2[0],
        "eta2": A2[1],
        "bta": BTA,
    }
    for index in range(4):
        requested[f"m{index + 1}"] = M[index]
        requested[f"dz{index + 1}"] = shear_photoz_bias[index]
    requested.update(extra)

    # Split the request along the model's own classification: a sampled
    # parameter is forwarded, a fixed one is accepted only at its yaml
    # value. A standard argument the yaml does not carry is dropped (it
    # keeps the same call working across the numbered yamls), but an extra
    # keyword the model does not know is a typo or a missing theory block,
    # so it raises instead of silently leaving the point unchanged.
    sampled = model.parameterization.sampled_params()
    fixed = model.parameterization.constant_params()
    unknown = [name for name in extra if name not in sampled and name not in fixed]
    if unknown:
        raise ValueError(
            f"{_STATE['yaml']} has no parameter {unknown}; check the name, "
            "or configure the theory block that defines it.")
    point = {}
    for name, value in requested.items():
        if name in sampled:
            point[name] = float(value)
        elif name in fixed and float(value) != float(fixed[name]):
            raise ValueError(
                f"{name} = {value}, but {_STATE['yaml']} fixes it to "
                f"{fixed[name]}; configure a yaml that samples {name}.")
    missing = [name for name in sampled if name not in point]
    if missing:
        raise ValueError(f"No value for the sampled parameters {missing}; "
                         "pass them as keyword arguments.")
    return point


def _evaluate(point):
    """Evaluate the configured model at one point; return its likelihood.

    Every observable function funnels through here. cached=False forces
    Cobaya to call the likelihood even at a point it has already seen:
    served from Cobaya's state cache, logp would not run, and the
    likelihood's last_datavector would still hold the previous point's
    prediction.

    Returns:
      the des_3pcf.joint_shear likelihood component, whose rows and
      last_datavector attributes the caller reads.
    Raises:
      ValueError when the posterior is not finite (the theory or the
      likelihood rejected the point).
    """
    model = _STATE["model"]
    posterior = model.logposterior(params_values=point, cached=False)
    if not np.isfinite(posterior.logpost):
        raise ValueError(f"The model returned logpost = {posterior.logpost} "
                         f"at {point}.")
    return model.likelihood[LIKELIHOOD]


def datavector(omegam=omegam, omegab=omegab, h=h, ns=ns, sigma8=sigma8,
               A1=A1_FID, A2=A2_FID, BTA=BTA_FID,
               shear_photoz_bias=SHEAR_PHOTOZ_FID, M=M_FID, **extra):
    """Compute the joint xi_+, xi_- and Map3 data vector in measured order.

    This is the prediction the likelihood scores, one entry per dataset
    row, in the dataset's own row order (the xi_+ block, then xi_-, then
    Map3). Use it when the whole vector is needed at once, for example
    for Fisher derivatives or for residuals against the measured values;
    xi() and map3() return the same numbers arranged on their angular
    grids.

    Arguments:
      omegam, omegab, h, ns, sigma8 = cosmology; A1, A2, BTA = TATT
        amplitudes (module docstring); shear_photoz_bias, M = the four
        source photo-z shifts and shear calibrations; extra = parameters
        of added theory blocks (configure(params=...)).
    Returns:
      (labels, values): labels = [ndata] row names built by the survey
        reader from each row's identity: the kind (xip, xim or map3),
        the zero-based source bins (third entry -1 for a xi row, which
        has only a source pair) and the angles in radians written as
        exact hexadecimal floats, joined with colons. values = [ndata]
        dimensionless predictions in the same order. With
        external_baryon_suppression the Map3 entries are NaN (the
        likelihood refuses them for a power-only feedback model).
    """
    likelihood = _evaluate(_point(omegam, omegab, h, ns, sigma8, A1, A2, BTA,
                                  shear_photoz_bias, M, extra))
    return (np.array(likelihood.rows.labels, copy=True),
            np.array(likelihood.last_datavector, copy=True))


def xi(omegam=omegam, omegab=omegab, h=h, ns=ns, sigma8=sigma8,
       A1=A1_FID, A2=A2_FID, BTA=BTA_FID,
       shear_photoz_bias=SHEAR_PHOTOZ_FID, M=M_FID, **extra):
    """Compute the binned shear correlation functions xi_+ and xi_-.

    Use it to plot or compare the two-point part of the prediction: the
    values are the xi entries of datavector() arranged on the dataset's
    angular grid, one [nsource, nsource] table of source-bin pairs per
    angular bin. The angular bins are the dataset's measured bins (their
    edges come from its two-point file), and each value is averaged over
    its bin as the measurement is.

    Arguments:
      as in datavector.
    Returns:
      (theta, xi_plus, xi_minus): theta = [ntheta] bin centres in arcmin;
        xi_plus, xi_minus = [ntheta, nsource, nsource], dimensionless,
        both bin orderings filled; NaN where the dataset has no row.
    """
    likelihood = _evaluate(_point(omegam, omegab, h, ns, sigma8, A1, A2, BTA,
                                  shear_photoz_bias, M, extra))
    rows = likelihood.rows
    values = likelihood.last_datavector

    # The common angular grid: a xi row stores (lower edge, upper edge,
    # centre) in its angles columns, so column 2 of the non-Map3 rows
    # holds the bin centres, and np.unique sorts and deduplicates them.
    theta = np.unique(rows.angles[rows.kind != "map3", 2])

    nsource = likelihood.source_ntomo
    result = {}
    for kind in ("xip", "xim"):
        # One [ntheta, nsource, nsource] table per kind; NaN marks the
        # entries for which the dataset has no row.
        table = np.full((theta.size, nsource, nsource), np.nan)
        for row in np.flatnonzero(rows.kind == kind):
            # searchsorted locates the row's centre on the sorted grid;
            # the match is exact, because the grid was built from these
            # same values.
            itheta = np.searchsorted(theta, rows.angles[row, 2])
            i, j = rows.bins[row, 0], rows.bins[row, 1]
            # xi is symmetric in the source pair and the dataset stores
            # each pair once, so one row fills both orderings.
            table[itheta, i, j] = values[row]
            table[itheta, j, i] = values[row]
        result[kind] = table
    return theta * RADIAN_TO_ARCMIN, result["xip"], result["xim"]


def map3(omegam=omegam, omegab=omegab, h=h, ns=ns, sigma8=sigma8,
         A1=A1_FID, A2=A2_FID, BTA=BTA_FID,
         shear_photoz_bias=SHEAR_PHOTOZ_FID, M=M_FID, **extra):
    """Compute the third-order aperture-mass statistic Map3.

    Use it to plot or compare the three-point part of the prediction:
    the values are the Map3 entries of datavector() arranged on the
    dataset's grid of filter radii and source triplets.

    Map3 is the third moment of the aperture mass, the shear field
    filtered by a compensated filter of radius R; the dataset uses equal
    radii on the three legs. Like the measurement, it is built from the
    shear three-point function on the dataset's side-angle bins (the
    finite-SAS estimator).

    Arguments:
      as in datavector.
    Returns:
      (radii, triplets, values): radii = [nradius] filter radii in arcmin;
        triplets = [ntriplet, 3] zero-based source bins, in dataset order;
        values = [nradius, ntriplet], dimensionless.
    Raises:
      ValueError if a Map3 row has unequal radii (use datavector).
    """
    likelihood = _evaluate(_point(omegam, omegab, h, ns, sigma8, A1, A2, BTA,
                                  shear_photoz_bias, M, extra))
    rows = likelihood.rows
    selected = np.flatnonzero(rows.kind == "map3")
    angles = rows.angles[selected]

    # The [nradius, ntriplet] layout below assumes one radius per row; a
    # row whose three legs differ has no place on that grid, so it is
    # refused rather than silently dropped.
    if np.any(angles[:, 1] != angles[:, 0]) or np.any(angles[:, 2] != angles[:, 0]):
        raise ValueError("Some Map3 rows have unequal radii; read them with "
                         "nw.datavector().")
    radii = np.unique(angles[:, 0])

    # Collect the source triplets in first-appearance (dataset) order;
    # np.unique would sort them and silently permute the value columns.
    triplets = []
    for row in selected:
        triplet = tuple(int(b) for b in rows.bins[row])
        if triplet not in triplets:
            triplets.append(triplet)

    # Fill the grid; searchsorted is an exact match, as in xi(), and NaN
    # marks a radius/triplet pair without a dataset row.
    values = np.full((radii.size, len(triplets)), np.nan)
    for row in selected:
        iradius = np.searchsorted(radii, rows.angles[row, 0])
        itriplet = triplets.index(tuple(int(b) for b in rows.bins[row]))
        values[iradius, itriplet] = likelihood.last_datavector[row]
    return radii * RADIAN_TO_ARCMIN, np.array(triplets, dtype=np.int32), values


def get_chi2(omegam=omegam, omegab=omegab, h=h, ns=ns, sigma8=sigma8,
             A1=A1_FID, A2=A2_FID, BTA=BTA_FID,
             shear_photoz_bias=SHEAR_PHOTOZ_FID, M=M_FID, **extra):
    """Compute chi2 of the configured dataset at one parameter point.

    chi2 = (d - t)^T C^-1 (d - t) over the rows the dataset's mask keeps,
    with d the measurement, t the prediction and C the dataset's
    covariance. The measurement, covariance and mask were loaded into
    the compiled interface when configure() built the likelihood, so the
    native call below only has to score the prediction against them.

    Arguments:
      as in datavector.
    Returns:
      chi2 as a float.
    """
    likelihood = _evaluate(_point(omegam, omegab, h, ns, sigma8, A1, A2, BTA,
                                  shear_photoz_bias, M, extra))
    return float(ci.compute_joint_chi2(prediction=likelihood.last_datavector))


def map3_components(triplets, velocity_shear=0.0, omegam=omegam,
                    omegab=omegab, h=h, ns=ns, sigma8=sigma8,
                    A1=A1_FID, A2=A2_FID, BTA=BTA_FID,
                    shear_photoz_bias=SHEAR_PHOTOZ_FID, M=M_FID, **extra):
    """Split the Map3 rows of chosen source triplets into IA operators.

    Use it to see which intrinsic-alignment term drives a Map3 prediction
    or its change with a parameter. The Map3 of a source triplet is a sum
    of five operators (gravity-only lensing; the tidal-alignment, density-
    weighted and tidal-torquing IA terms; the velocity-shear term) times
    eight placements, which say which of the three legs carry an
    intrinsic shape (GGG = all three lensed, ..., III = all three
    intrinsic). Configure a TATT yaml (EXAMPLE_EVALUATE2) so every IA
    operator has a nonzero amplitude.

    The projection is repeated once per operator and placement, so this
    builds a second aperture plan restricted to the requested rows, with
    the same settings as the likelihood; each row keeps the value it has
    in the full plan, which "prediction" lets the caller check.

    Arguments:
      triplets = [ntriplet, 3] zero-based source bins; [0, 2, 3] is the
        triplet labelled 134.
      velocity_shear = the velocity-shear amplitude C_t in units of the
        linear tidal-alignment coefficient C_1 (the likelihood itself
        always uses 0); the rest as in datavector.
    Returns:
      dictionary with
        components = "operators" (5 names), "placements" (8 names),
          "values" [5, 8, nrow], "total" [nrow] (computed separately) and
          "closure" [nrow] (sum of values minus total);
        rows = the selected SurveyRows (labels, bins, radii in radians);
        indices = their positions in the joint data vector;
        prediction = the likelihood's own Map3 rows there (with C_t = 0);
        ia = the IA dictionary used.
    Raises:
      ValueError for a triplet with no Map3 row or a non-finite
      velocity_shear.
    """
    likelihood = _evaluate(_point(omegam, omegab, h, ns, sigma8, A1, A2, BTA,
                                  shear_photoz_bias, M, extra))
    requested = np.asarray(triplets)
    # dtype.kind is numpy's one-letter type-class code: "i" = signed and
    # "u" = unsigned integers. A float array is refused, because a
    # fractional source bin has no meaning and silent truncation could
    # select the wrong rows.
    if (requested.ndim != 2 or requested.shape[1] != 3
            or requested.dtype.kind not in "iu"):
        raise ValueError(
            f"triplets has shape {requested.shape} and dtype {requested.dtype}; "
            "supply an integer array of shape [ntriplet, 3].")
    if not np.isfinite(velocity_shear):
        raise ValueError(f"velocity_shear = {velocity_shear}; supply a finite number.")

    # ------------------------------------------------------------------
    # The Map3 rows whose bin triplet matches a requested one, in the
    # likelihood's row order.
    # ------------------------------------------------------------------
    rows = likelihood.rows
    selected = np.zeros(shape=rows.values.size, dtype=bool)
    for triplet in requested:
        match = (rows.kind == "map3") & np.all(rows.bins == triplet, axis=1)
        if not np.any(match):
            raise ValueError(
                f"No Map3 row has the source triplet {triplet}; the bins run "
                f"from 0 to {likelihood.source_ntomo - 1}.")
        selected |= match
    indices = np.flatnonzero(selected)
    # rows is rebound to a new record holding only the selected rows, in
    # data-vector order; the likelihood's full table is left untouched.
    rows = rows.select(indices=indices)

    # ------------------------------------------------------------------
    # The aperture plan of those rows, with the likelihood's settings: the
    # side bins are the likelihood's own (the measurement's bins after its
    # scale cut, lower edges in radians, log width dlnt) and the opening
    # angle phi takes the centres of phi_bins equal cells of [0, pi].
    # ------------------------------------------------------------------
    dphi = np.pi / likelihood.phi_bins
    plan = ci.production.TransformPlan.for_aperture(
        bins=np.ascontiguousarray(a=rows.bins, dtype=np.int32),
        filters=np.ascontiguousarray(a=rows.angles),
        theta_lower=likelihood.theta_lower,
        phi=(np.arange(likelihood.phi_bins) + 0.5) * dphi,
        dlnt=likelihood.dlnt,
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

    # ------------------------------------------------------------------
    # The decomposition: a copy of the likelihood's IA record with the
    # requested velocity shear (ct_mode = 0 means C_t = C_1 * ct).
    # ------------------------------------------------------------------
    ia = dict(likelihood.ia, ct=float(velocity_shear), ct_mode=0)
    # This projection runs outside the likelihood's own evaluation path,
    # so the OpenMP worker count is set here to the likelihood's value.
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
