"""Join cosmology, source uncertainties and the full survey mean vector.

This module gives the Fisher driver one deterministic prediction function.
Cosmological changes rebuild power tables when CAMB mode is selected.
Photo-z changes rebuild the source-weighted C model. IA changes reuse that
model's prepared matter tables, while shear calibration acts on the final
vector. Exact value keys distinguish these physically different costs.

The aperture estimator is an explicit choice: full Fourier filters for
comparison with the analytic covariance, or the finite shear-correlation
integral for comparison with catalogue measurements. They are never
identified merely because their rows have the same aperture-radius labels.
"""


import numpy as np
from copy import deepcopy

from des3pcf.physical_provider import validate_tables
from des3pcf.survey import shifted_sources
from des3pcf.physical_provider import camb_tables, _camb_power_limits
from des3pcf.survey import reconstructed_sources
from des3pcf.physical_provider import camb_accuracy_settings
from cosmolike_des_3pcf_interface.production import TransformPlan
from cosmolike_des_3pcf_interface import shear_power_rows
from theory.bacco_bispectrum.response import FeedbackPair
from theory.bacco_bispectrum.response import feedback_power_tables
from theory.bacco_bispectrum.response import BaccoFeedback, BACCO_PARAMETERS
from cosmolike_des_3pcf_interface import quadrature


# Only these parameters require a new Boltzmann calculation. Shape and
# redshift nuisance parameters are applied at their own later stages.
COSMOLOGY_NAMES = (
    "omega_m", "omega_b", "h", "ns", "sigma8", "w0", "wa",
    "mnu", "num_massive_neutrinos", "nnu",
)
IA_NAMES = (
    "a1", "eta1", "a2", "eta2", "bta", "ct", "zpivot", "c1rho",
    "power_mode", "ct_mode", "fb", "suppress_only",
    "feedback_amplitude",
)


import cosmolike_des_3pcf_interface as ci


def parameter_key(parameters, names):
    """Represent exactly the named finite parameter values for cache comparison.

    Arguments:
      parameters = dictionary with every requested name.
      names = stable sequence defining the key's order.
    Returns:
      tuple of float hexadecimal strings; no approximate equality is used.
    Raises:
      ValueError for a missing or nonfinite parameter.
    """
    values = []
    for name in names:
        if name not in parameters or not np.isfinite(parameters[name]):
            raise ValueError(f"Prediction requires a finite parameter named {name}.")
        values.append(float(parameters[name]).hex())
    return tuple(values)


class JointPrediction:
    """Evaluate all selected xi± and Map3 rows in their original order.

    Arguments:
      compiled = project extension.
      rows = explicit full or selected SurveyRows.
      provider = PhysicalProvider supplying cosmology and source tables.
      ia_defaults = explicit physical IA choices, overridden by named parameters.
      aperture_estimator = 'full_filter' or 'finite_sas'.
      aperture_settings = full_filter ell/phi quadrature dictionaries, or
          finite_sas keyword arguments for TransformPlan.for_aperture.
      xi_settings = ell quadrature dictionary, theta_count, pair_weight.
      two_point_core = ordinary-core module and settings for CFASTPT spectra.
      accuracy = geometry and quadrature boosts, resolved from baseline settings.
          Construct another prediction to change these prepared settings.
    Side effects:
      Retains geometry plans; uses the provider's exact physical-input cache.
    """

    def __init__(
        self, compiled, rows, provider, ia_defaults, aperture_estimator,
        aperture_settings, xi_settings=None, two_point_core=None,
        accuracy=None,
    ):
        """Prepare fixed geometry separately from all varied physical parameters.

        Arguments:
          All arguments follow the class contract; no data compression occurs.
        Returns:
          nothing; only finite_sas mode constructs FFT plans at this point.
        Raises:
          ValueError for an unsupported estimator or missing xi settings.
        """
        if aperture_estimator not in ("full_filter", "finite_sas"):
            raise ValueError("Aperture estimator must be full_filter or finite_sas.")
        self.compiled = compiled
        # Own the row definition used by the retained geometry and mean.
        # Editing the caller's arrays must not change only half of a plan.
        self.rows = rows.select(indices=np.arange(rows.values.size, dtype=np.int64))
        self.provider = provider
        self.ia_defaults = dict(ia_defaults)
        self.estimator = aperture_estimator
        self.two_point_core = two_point_core
        if self.two_point_core is not None:
            core = self.two_point_core["core"]
            nbin = self.two_point_core["nbin"]
            threads = self.two_point_core.get("threads", 1)
            interpolation_type = self.two_point_core.get("interpolation_type", 0)
            accuracy_boost = self.two_point_core.get("accuracy_boost", 1.0)
            integration_accuracy = self.two_point_core.get("integration_accuracy", 0)
            fpt_internal_boost = self.two_point_core.get("fpt_internal_boost", 0.5)
            ell_internal = self.two_point_core.get("ell_internal", 192)
            core.initial_setup()
            core.set_omp_threads(n=threads)
            core.init_accuracy_boost(
                accuracy_boost=accuracy_boost,
                integration_accuracy=integration_accuracy,
            )
            # Apply dedicated overrides after the common boost, whose core
            # implementation also scales the coarse ell and CFASTPT controls.
            core.init_fpt_internal_boost(internal_boost=fpt_internal_boost)
            core.init_ntable_ell_internal(nell_internal=ell_internal)
            core.init_probes(possible_probes="xi")
            core.init_cosmo_runmode(is_linear=False)
            core.init_include_halo_IA(include_halo_IA=0)
            core.init_photoz_histogram_akima(
                enabled=0,
                preparation_factor=3,
            )
            core.init_photoz_conventions(
                interpolation_type=interpolation_type,
                zmid_convention=1,
            )
            core.init_source_sample_size(Ntomo=nbin)
            # One unused lens bin permits the shared pair bookkeeping to be
            # initialized. Only source-source spectra are ever requested here.
            core.init_lens_sample_size(Ntomo=1)
            core.init_IA(
                ia_model=1,
                ia_redshift_evolution=3,
                ia_code=0,
            )
            core.reset_bary_struct()
        self.accuracy = ci.init_3pcf_accuracy(value=accuracy)
        self.map_indices = np.flatnonzero(rows.kind == "map3")
        self.xi_indices = np.flatnonzero(rows.kind != "map3")
        self.map_engine = None
        self.groups = {}
        self._cached_model = None
        self._cached_ia_key = None
        self._cached_mean = None
        aperture_settings = deepcopy(aperture_settings)
        if aperture_estimator == "full_filter":
            projection = aperture_settings.get("projection_settings", {})
            aperture_settings["projection_settings"] = ci.accuracy_projection_settings(
                accuracy=self.accuracy,
                baseline=projection,
            )
        if self.map_indices.size:
            self.map_rows = rows.select(indices=self.map_indices)
            if aperture_estimator == "finite_sas":
                self.map_engine = TransformPlan.for_aperture(
                    bins=np.ascontiguousarray(self.map_rows.bins, dtype=np.int32),
                    filters=np.ascontiguousarray(self.map_rows.angles),
                    **aperture_settings,
                    accuracy=self.accuracy,
                )
            else:
                ell_settings = ci.accuracy_quadrature_settings(
                    accuracy=self.accuracy,
                    baseline=aperture_settings["ell"],
                    stage="aperture_ell_boost",
                )
                phi_settings = ci.accuracy_quadrature_settings(
                    accuracy=self.accuracy,
                    baseline=aperture_settings["phi"],
                    stage="aperture_phi_boost",
                )
                self.ell, self.dell = quadrature(**ell_settings)
                self.phi, self.dphi = quadrature(**phi_settings)
                self.projection_mode = aperture_settings.get(
                    "projection_mode", "prepared",
                )
                if self.projection_mode not in ("prepared", "direct"):
                    raise ValueError("Projection mode must be prepared or direct.")
                batch_size = aperture_settings.get("projection_batch_size", 4)
                if int(batch_size) != batch_size or batch_size < 1:
                    raise ValueError("projection_batch_size must be a positive integer.")
                self.projection_batch_size = int(batch_size)
                self.projection_settings = dict(
                    aperture_settings.get("projection_settings", {}),
                )
                self.projection_settings.setdefault(
                    "ell_min", float(aperture_settings["ell"]["lower"]),
                )
                self.projection_settings.setdefault(
                    "ell_max", float(aperture_settings["ell"]["upper"]),
                )
                self.projection_settings.setdefault("prepare_multipoles", 0)
                if self.projection_settings["prepare_multipoles"] != 0:
                    raise ValueError(
                        "Full-filter projection does not require multipole tables."
                    )
                for row in range(self.map_rows.values.size):
                    triplet = tuple(int(value) for value in self.map_rows.bins[row])
                    if triplet not in self.groups:
                        self.groups[triplet] = []
                    self.groups[triplet].append(row)
        if self.xi_indices.size:
            if xi_settings is None:
                raise ValueError(
                    "Xi rows require explicit Fourier and angular-bin quadratures."
                )
            selected = rows.select(indices=self.xi_indices)
            ell_settings = ci.accuracy_quadrature_settings(
                accuracy=self.accuracy,
                baseline=xi_settings["ell"],
                stage="xi_ell_boost",
            )
            ell, dell = quadrature(**ell_settings)
            theta_count = ci.accuracy_count(
                accuracy=self.accuracy,
                baseline=xi_settings["theta_count"],
                stage="xi_theta_boost",
                integration=True,
            )
            if (theta_count != xi_settings["theta_count"]
                    and not isinstance(xi_settings["pair_weight"], str)):
                raise ValueError(
                    "Refining xi theta nodes requires new measured pair weights; "
                    "supply them at the desired baseline count or use uniform_area."
                )
            self.xi_rows = selected
            self.xi_ell = np.array(ell, copy=True)
            self.xi_dell = np.array(dell, copy=True)
            self.xi_theta, self.xi_weight = compiled.xi_bin_weights(
                kind=self.xi_rows.kind.tolist(),
                angles=self.xi_rows.angles,
                count=theta_count,
                pair_weight=xi_settings["pair_weight"],
            )
            self.xi_sign = np.where(self.xi_rows.kind == "xim", -1, 1).astype(np.int32)

    def cache_token(self):
        """Identify the mutable numerical choices before an outer mean-cache read.

        Physical parameter values belong to the outer cache's parameter key.
        This token adds settings that a notebook can change independently.
        Geometry, reference tables and default IA values are immutable owned
        inputs, so their owning object's identity is sufficient here.
        """
        provider = self.provider
        return (
            id(self), provider.mode, provider.halofit_version,
            provider.numerical_key(),
        )

    def __call__(self, parameters):
        """Evaluate the mean used by exact-key derivatives and Fisher forecasts.

        Arguments:
          parameters = cosmology, IA and optional m1..mN/dz1..dzN values.
        Returns:
          dimensionless [nrow] joint vector, with all selected two-point entries.
        Raises:
          ValueError for unknown parameter names or unavailable physical models.
        Side effects:
          May prepare new physical tables and the ordinary two-point core.
        """
        allowed = set(COSMOLOGY_NAMES + IA_NAMES)
        if isinstance(self.provider.feedback, BaccoFeedback):
            allowed.update(BACCO_PARAMETERS)
        for index in range(self.provider.nbin):
            allowed.add(f"m{index + 1}")
            allowed.add(f"dz{index + 1}")
        for name in parameters:
            if name not in allowed:
                raise ValueError(
                    f"Unknown prediction parameter {name}; check its spelling."
                )
        ia = dict(self.ia_defaults)
        for name in IA_NAMES:
            if name in parameters:
                ia[name] = parameters[name]
        feedback_amplitude = ia.get("feedback_amplitude", 0.0)
        if not np.isfinite(feedback_amplitude):
            raise ValueError("feedback_amplitude must be finite.")
        if feedback_amplitude != 0.0:
            if self.provider.feedback is None:
                raise ValueError("Nonzero feedback_amplitude needs paired response tables.")
            if ia.get("fb", 0.0) != 0.0:
                raise ValueError("Paired feedback cannot be combined with legacy fb.")
        if isinstance(self.provider.feedback, BaccoFeedback) and feedback_amplitude != 1.0:
            raise ValueError("Dynamic BACCO requires fixed feedback_amplitude=1.")
        model, tables, metadata = self.provider.get(parameters=parameters)
        calibration = np.zeros(self.provider.nbin, dtype=np.float64)
        for index in range(self.provider.nbin):
            calibration[index] = parameters.get(f"m{index + 1}", 0.0)
        if not np.all(np.isfinite(calibration)):
            raise ValueError("Shear calibration values must be finite.")
        ia_key = parameter_key(parameters=ia, names=sorted(ia))
        # The C model owns immutable copies of its cosmology and n(z).
        # Object identity is sufficient only under that ownership contract;
        # all mutable IA choices are compared by exact numerical value.
        if self._cached_model is model and self._cached_ia_key == ia_key:
            return self.calibrated_mean(values=self._cached_mean, calibration=calibration)
        result = np.empty(self.rows.values.size, dtype=np.float64)
        if self.map_indices.size:
            if self.estimator == "finite_sas":
                aperture = self.map_engine.aperture_vector(
                    model=model, ia=ia,
                )
            else:
                aperture = self.full_aperture_mean(model=model, ia=ia)
            result[self.map_indices] = aperture
        if self.xi_indices.size:
            if self.two_point_core is None:
                even = self.compiled.nla_power_rows(
                    model=model,
                    ell=self.xi_ell,
                    bins=np.ascontiguousarray(self.xi_rows.bins[:, :2], dtype=np.int32),
                    ia=ia,
                )
                xi = self.compiled.production.xi_from_power(
                    ell=self.xi_ell,
                    dell=self.xi_dell,
                    theta=self.xi_theta,
                    theta_weight=self.xi_weight,
                    spectra_e=even,
                    spectra_b=np.zeros_like(even),
                    sign=self.xi_sign,
                )
            else:
                physical_parameters = dict(self.provider.reference)
                physical_parameters.update(parameters)
                two_point_metadata = dict(metadata)
                two_point_metadata["paired_feedback_amplitude"] = feedback_amplitude
                core = self.two_point_core["core"]
                nbin = self.two_point_core["nbin"]
                threads = self.two_point_core.get("threads", 1)
                parameters = physical_parameters
                tables = feedback_power_tables(
                    model=model,
                    tables=tables,
                    amplitude=feedback_amplitude,
                )
                metadata = two_point_metadata
                ell = self.xi_ell
                bins = self.xi_rows.bins[:, :2]
                checked = validate_tables(tables=tables)
                amplitude = ia.get("feedback_amplitude", 0.0)
                if metadata.get("paired_feedback_amplitude", 0.0) != amplitude:
                    raise ValueError(
                        "Two-point nonlinear P and IA dictionary name different "
                        "paired_feedback_amplitude values; prepare matching tables."
                    )
                if "p_linear" not in checked:
                    raise ValueError(
                        "The TATT two-point route requires full p_linear[z,k] tables."
                    )
                if checked["sources"].shape[0] != nbin:
                    raise ValueError("Physical source count changed; create a new core adapter.")
                h = float(parameters["h"])
                omega_b = float(parameters["omega_b"])
                omega_nu = float(metadata.get("omega_nu_h2", 0.0))
                if parameters.get("mnu", 0.0) > 0.0 and "omega_nu_h2" not in metadata:
                    raise ValueError("Massive-neutrino tables require recorded omega_nu_h2.")
                if omega_nu > 0.0 and "p_linear_cb" not in checked:
                    raise ValueError(
                        "Massive-neutrino core preparation requires linear P_cb[z,k]."
                    )
                # The YAML uses the ordinary core convention directly. There is no
                # amplitude or pivot conversion in the production likelihood.
                first = np.zeros(nbin, dtype=np.float64)
                second = np.zeros(nbin, dtype=np.float64)
                density = np.zeros(nbin, dtype=np.float64)
                first[:2] = [ia.get("a1", 0.0), ia.get("eta1", 0.0)]
                second[:2] = [ia.get("a2", 0.0), ia.get("eta2", 0.0)]
                density[0] = ia.get("bta", 0.0)
                source_z = checked.get("source_z", checked["z"])
                # The ordinary midpoint reader reconstructs uniform coordinates
                # from the endpoint spacing. Refuse an irregular independent axis
                # instead of silently passing coordinates that reader would move.
                uniform = np.linspace(start=source_z[0], stop=source_z[-1],
                                      num=source_z.size)
                tolerance = 32.0 * np.finfo(float).eps * source_z[-1]
                if not np.allclose(source_z, uniform, rtol=0.0, atol=tolerance):
                    raise ValueError("The two-point bridge requires a uniform source_z grid.")
                source_table = np.column_stack((source_z, checked["sources"].T))
                lens_table = source_table[:, :2].copy()
                core.set_source_sample(nofz=source_table)
                core.set_lens_sample(nofz=lens_table)
                core.init_ntomo_powerspectra()
                cold = []
                if "p_linear_cb" in checked:
                    cold = np.log(checked["p_linear_cb"]).flatten(order="F")
                # The old core stores [k,z], whereas the new tables store [z,k].
                # Fortran flattening of the latter gives exactly the former order.
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
                core.set_nuisance_shear_photoz(bias=np.zeros(nbin))
                core.set_nuisance_shear_calib(M=np.zeros(nbin))
                core.set_nuisance_ia(
                    A1=first,
                    A2=second,
                    B_TA=density,
                )
                core.set_omp_threads(n=threads)
                even, odd = shear_power_rows(
                    ell=ell,
                    bins=bins,
                )
                xi = self.compiled.production.xi_from_power(
                    ell=self.xi_ell,
                    dell=self.xi_dell,
                    theta=self.xi_theta,
                    theta_weight=self.xi_weight,
                    sign=self.xi_sign,
                    bins=np.ascontiguousarray(self.xi_rows.bins[:, :2], dtype=np.int32),
                    spectra_e=even,
                    spectra_b=odd,
                )
            result[self.xi_indices] = xi
        if not np.all(np.isfinite(result)):
            raise ValueError("Joint prediction contains a nonfinite value.")
        self._cached_model = model
        self._cached_ia_key = ia_key
        self._cached_mean = deepcopy(x=result)
        return self.calibrated_mean(values=result, calibration=calibration)

    def full_aperture_mean(self, model, ia):
        """Integrate full Fourier filters with bounded batches of source tables.

        Arguments:
          model = immutable physical C model for this cosmology and sources.
          ia = exact intrinsic-alignment and matter-response choices.
        Returns:
          uncalibrated aperture rows in their declared order.
        Side effects:
          Prepared mode allocates at most projection_batch_size projected
          tables together; fixed Fourier quadrature then uses linear reads.
          Direct mode evaluates every line-of-sight integral as a reference.
        """
        aperture = np.empty(self.map_indices.size, dtype=np.float64)
        triplets = list(self.groups)
        for start in range(0, len(triplets), self.projection_batch_size):
            batch = triplets[start:start + self.projection_batch_size]
            grids = None
            if self.projection_mode == "prepared":
                grids = model.projection_grids(
                    bins=np.ascontiguousarray(batch, dtype=np.int32),
                    parameters=ia, settings=dict(self.projection_settings),
                )
            for index, triplet in enumerate(batch):
                indices = self.groups[triplet]
                # The retained group is an immutable tuple. A list selects
                # rows; a tuple would instead index separate NumPy axes.
                filters = np.ascontiguousarray(self.map_rows.angles[list(indices)])
                arguments = {
                    "ell": self.ell, "dell": self.dell, "phi": self.phi,
                    "dphi": self.dphi, "filters": filters,
                }
                if grids is None:
                    values = model.aperture_full(
                        **arguments, bins=np.ascontiguousarray(triplet, dtype=np.int32),
                        parameters=ia,
                    )
                else:
                    values = grids[index].aperture_full(**arguments)
                aperture[list(indices)] = values
            grids = None
        return aperture

    def calibrated_mean(self, values, calibration):
        """Apply one multiplicative factor for every observed shear field.

        A (1,1,2) aperture row contains two fields from sample 1, so its
        factor is (1+m1)^2(1+m2). Calibration does not change any physical
        projection or Fourier transform. Retaining their uncalibrated
        result avoids repeating those calculations in calibration-only
        Fisher displacements.

        Arguments:
          values = retained, uncalibrated [nrow] prediction.
          calibration = finite [nbin] shear offsets.
        Returns:
          an owned [nrow] vector; the retained prediction is unchanged.
        """
        result = values.copy()
        for row in range(result.size):
            for source in self.rows.bins[row]:
                if source >= 0:
                    result[row] *= 1.0 + calibration[source]
        return result
