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

from .inputs import validate_tables, shifted_sources, camb_tables, _camb_power_limits
from .mean import physics_model, Map3Engine, XiEngine
from .covariance import quadrature
from .accuracy import accuracy_settings
from .ownership import immutable_copy, PreparedGeometry


# Only these parameters require a new Boltzmann calculation. Shape and
# redshift nuisance parameters are applied at their own later stages.
COSMOLOGY_NAMES = (
    "omega_m", "omega_b", "h", "ns", "sigma8", "w0", "wa",
    "mnu", "num_massive_neutrinos", "nnu",
)
IA_NAMES = (
    "a1", "eta1", "a2", "eta2", "bta", "ct", "zpivot", "c1rho",
    "power_mode", "ct_mode", "fb", "suppress_only",
)


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


class PhysicalProvider:
    """Prepare one owned physical model at each distinct cosmology and photo-z point.

    Arguments:
      compiled = project extension containing PhysicsModel.
      reference_tables = physical arrays, including the original unshifted n(z).
      reference_metadata = provenance and optional CAMB/neutrino information.
      reference_parameters = cosmological values corresponding to those tables.
      cosmology_mode = 'fixed_tables' or 'camb'.
      kernel_nz = baseline source-kernel interpolation node count.
      halofit_version = explicit nonlinear-power prescription for CAMB calls.
      kmax_boltzmann = upper CAMB evolution wavenumber in 1/Mpc, default 5.
      extrap_kmax = power-interpolator extension in 1/Mpc; None selects
          max(250, 1.01*h*k[-1]) for each requested cosmology.
      source_readers = use upstream spline readers only for reference comparisons.
      table_boost = uniform read-table refinement; cubic preparation occurs once.
      fit_nz = coarse uniform redshift anchors for the matter-fit coefficients.
      accuracy = immutable global/stage controls; replace provider.accuracy to
          prepare a new resolution from the same baseline in this interpreter.
    Side effects:
      Retains the latest cosmology table and source-weighted C model.
      CAMB runs only after a physical parameter changes in camb mode.
    """

    def __init__(
        self, compiled, reference_tables, reference_metadata, reference_parameters,
        cosmology_mode="fixed_tables", kernel_nz=101, halofit_version="takahashi",
        source_readers=False, table_boost=8, fit_nz=33,
        accuracy=None, kmax_boltzmann=5.0, extrap_kmax=None,
    ):
        """Store the physical baseline and validate the requested producer mode.

        Arguments:
          All arguments have the units and meanings in the class contract.
        Returns:
          nothing; no Boltzmann calculation or C projection occurs here.
        Raises:
          ValueError for inconsistent reference parameters or unsupported mode.
        """
        if cosmology_mode not in ("fixed_tables", "camb"):
            raise ValueError("Cosmology mode must be fixed_tables or camb.")
        self.compiled = compiled
        self.base = immutable_copy(value=validate_tables(tables=reference_tables))
        self.base_metadata = immutable_copy(value=reference_metadata)
        self.accuracy = accuracy_settings(value=accuracy)
        self.mode = cosmology_mode
        if int(kernel_nz) != kernel_nz or kernel_nz < 4:
            raise ValueError("kernel_nz must be an integer of at least four.")
        self.kernel_nz = int(kernel_nz)
        self.halofit_version = halofit_version
        self.kmax_boltzmann = kmax_boltzmann
        self.extrap_kmax = extrap_kmax
        if not isinstance(source_readers, (bool, np.bool_)):
            raise ValueError("source_readers must be an explicit Boolean choice.")
        if int(table_boost) != table_boost or table_boost < 1:
            raise ValueError("table_boost must be a positive integer.")
        self.source_readers = bool(source_readers)
        self.table_boost = int(table_boost)
        if int(fit_nz) != fit_nz or fit_nz < 4:
            raise ValueError("fit_nz must be an integer of at least four.")
        self.fit_nz = int(fit_nz)
        self.reference = dict(reference_parameters)
        self.reference.setdefault("mnu", 0.0)
        self.reference.setdefault("num_massive_neutrinos", 1)
        self.reference.setdefault("nnu", 3.046)
        self.reference_cosmology_key = parameter_key(
            self.reference, names=COSMOLOGY_NAMES,
        )
        self.reference = immutable_copy(value=self.reference)
        self.reference_producer_settings = self._producer_settings(
            parameters=self.reference,
        )
        for name in ("omega_m", "ns", "sigma8", "w0", "wa"):
            expected = self.base[name]
            if not np.isclose(self.reference[name], expected, rtol=1.0e-8, atol=1.0e-12):
                raise ValueError(
                    f"Reference parameter {name} differs from its physical table."
                )
        # Power tables also depend on h, baryon density and neutrino
        # choices, which are absent from the minimal C table schema.
        # Compare any recorded producer values before reusing the baseline;
        # otherwise a derivative can mix two different physical models.
        recorded = reference_metadata.get("parameters", {})
        if not isinstance(recorded, dict):
            raise ValueError("Physical-table metadata parameters must be a dictionary.")
        for name in COSMOLOGY_NAMES:
            if name in recorded:
                if not np.isclose(
                    self.reference[name], recorded[name], rtol=1.0e-10, atol=1.0e-12,
                ):
                    raise ValueError(
                        f"Reference parameter {name} differs from table provenance."
                    )
        recorded_halofit = self.base_metadata.get("halofit_version")
        if (self.mode == "camb" and recorded_halofit is not None
                and recorded_halofit != self.halofit_version):
            raise ValueError(
                "CAMB nonlinear prescription differs from the baseline table."
            )
        # Older tables may lack the distinction between the CAMB evolution
        # range and the extrapolated read range. They remain usable as fixed
        # numerical arrays, but cannot silently seed a changed-CAMB calculation.
        limits_recorded = all(
            name in self.base_metadata for name in ("kmax_boltzmann", "extrap_kmax")
        )
        self.reference_producer_verified = (
            limits_recorded and recorded_halofit is not None
        )
        if limits_recorded:
            expected = self.reference_producer_settings[1:]
            recorded_limits = (
                self.base_metadata["kmax_boltzmann"],
                self.base_metadata["extrap_kmax"],
            )
            if recorded_limits != expected:
                raise ValueError(
                    "CAMB wavenumber limits differ from table provenance; "
                    "regenerate tables with the declared limits."
                )
        if self.mode == "camb" and not self.reference_producer_verified:
            raise ValueError(
                "CAMB reference tables lack kmax_boltzmann/extrap_kmax provenance; "
                "regenerate the reference tables before varying cosmology."
            )
        self.nbin = self.base["sources"].shape[0]
        self.cosmology_key = self.reference_cosmology_key
        self.producer_key = (self.mode,) + self.reference_producer_settings
        self.reference_halofit = self.halofit_version
        self.cosmology_tables = self.base
        self.cosmology_metadata = self.base_metadata
        self.model_key = None
        self.model = None
        self.model_tables = None
        self._reference_locked = True

    def __setattr__(self, name, value):
        """Protect the physical baseline while keeping numerical controls changeable.

        Replacing tables or source counts requires a new provider. That new
        object can coexist with the old one in the same notebook kernel.
        """
        reference_names = (
            "base", "base_metadata", "reference", "reference_cosmology_key",
            "reference_halofit", "nbin", "compiled",
            "reference_producer_settings", "reference_producer_verified",
        )
        if getattr(self, "_reference_locked", False) and name in reference_names:
            raise AttributeError(
                f"PhysicalProvider.{name} defines its baseline; construct a new "
                "provider with the replacement tables in this kernel."
            )
        if name == "accuracy":
            value = accuracy_settings(value=value)
        object.__setattr__(self, name, value)

    def _producer_settings(self, parameters):
        """Resolve and validate both physical wavenumber limits for one cosmology.

        CAMB's two limits are in 1/Mpc; the saved k axis is in h/Mpc. Multiplying
        its last value by h compares physical wavenumbers without conflating
        the output sampling range with the range evolved by the Boltzmann code.

        Arguments:
          parameters = cosmology containing h.
        Returns:
          Tuple of halofit name, evolution limit and resolved extrapolation limit.
        Raises:
          ValueError for nonpositive/nonfinite limits or insufficient read support.
        """
        for name in ("kmax_boltzmann", "extrap_kmax"):
            if isinstance(getattr(self, name), np.bool_):
                raise ValueError(f"{name} must be a physical wavenumber, not Boolean.")
        limit, extrapolation = _camb_power_limits(
            k=self.base["k"], h=float(parameters["h"]),
            kmax_boltzmann=self.kmax_boltzmann, extrap_kmax=self.extrap_kmax,
        )
        return self.halofit_version, limit, extrapolation

    def numerical_key(self):
        """Validate mutable controls before comparing any retained prediction.

        Returns:
          exact numerical settings, including the immutable boost object.
        Raises:
          ValueError for an invalid notebook setting, even on a cache hit.
        """
        # Validate mutable notebook controls before testing cache equality.
        # Otherwise True==1 could reuse a model despite an invalid count.
        for name, minimum in (("kernel_nz", 4), ("table_boost", 1), ("fit_nz", 4)):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not np.isfinite(value)
                    or int(value) != value or value < minimum):
                raise ValueError(f"{name}={value!r} must be an integer >= {minimum}.")
        if not isinstance(self.source_readers, (bool, np.bool_)):
            raise ValueError("source_readers must be an explicit Boolean choice.")
        self._producer_settings(parameters=self.reference)
        return (
            self.kernel_nz, self.source_readers, self.table_boost, self.fit_nz,
            accuracy_settings(value=self.accuracy),
            self.kmax_boltzmann, self.extrap_kmax,
        )

    def get(self, parameters):
        """Return the model and exact input tables for this physical point.

        Arguments:
          parameters = cosmology dictionary plus optional dz1,...,dzN shifts.
        Returns:
          PhysicsModel, source-shifted tables, and preparation metadata.
        Raises:
          ValueError if fixed tables are asked to represent a new cosmology.
        Side effects:
          Replaces the latest owned model when a physical input changes.
        """
        numerical_key = self.numerical_key()
        cosmology = dict(self.reference)
        for name in COSMOLOGY_NAMES:
            if name in parameters:
                cosmology[name] = parameters[name]
        key = parameter_key(cosmology, names=COSMOLOGY_NAMES)
        if self.mode not in ("fixed_tables", "camb"):
            raise ValueError("Cosmology mode must be fixed_tables or camb.")
        producer_settings = self._producer_settings(parameters=cosmology)
        producer_key = (self.mode,) + producer_settings
        if key != self.cosmology_key or producer_key != self.producer_key:
            if self.mode == "fixed_tables" and (
                key != self.reference_cosmology_key
                or producer_settings != self.reference_producer_settings
            ):
                raise ValueError(
                    "A cosmological parameter changed; select camb or supply new tables."
                )
            if (key == self.reference_cosmology_key
                    and producer_settings == self.reference_producer_settings
                    and (self.mode == "fixed_tables"
                         or self.reference_producer_verified)):
                tables = self.base
                metadata = self.base_metadata
            else:
                tables, metadata = camb_tables(
                    parameters=cosmology, k=self.base["k"], z=self.base["z"],
                    sources=self.base["sources"], los_z=self.base["los_z"],
                    halofit_version=self.halofit_version,
                    kmax_boltzmann=self.kmax_boltzmann,
                    extrap_kmax=self.extrap_kmax,
                )
            self.cosmology_tables = immutable_copy(value=tables)
            self.cosmology_metadata = immutable_copy(value=metadata)
            self.cosmology_key = key
            self.producer_key = producer_key
        shifts = np.zeros(self.nbin, dtype=np.float64)
        for index in range(self.nbin):
            shifts[index] = parameters.get(f"dz{index + 1}", 0.0)
        if not np.all(np.isfinite(shifts)):
            raise ValueError("All source redshift shifts must be finite.")
        source_key = tuple(float(value).hex() for value in shifts)
        combined = (
            key, producer_key, source_key, numerical_key,
        )
        if combined != self.model_key:
            tables = dict(self.cosmology_tables)
            tables["sources"] = shifted_sources(
                z=self.base["z"], sources=self.base["sources"], shifts=shifts,
            )
            self.model = physics_model(
                compiled=self.compiled, tables=tables, kernel_nz=self.kernel_nz,
                source_readers=self.source_readers, table_boost=self.table_boost,
                fit_nz=self.fit_nz,
                accuracy=self.accuracy,
            )
            self.model_tables = immutable_copy(value=tables)
            self.model_key = combined
        return self.model, self.model_tables, self.cosmology_metadata


class JointPrediction(PreparedGeometry):
    """Evaluate all selected xi± and Map3 rows in their original order.

    Arguments:
      compiled = project extension.
      rows = explicit full or selected SurveyRows.
      provider = PhysicalProvider supplying cosmology and source tables.
      ia_defaults = explicit physical IA choices, overridden by named parameters.
      aperture_estimator = 'full_filter' or 'finite_sas'.
      aperture_settings = full_filter ell/phi quadrature dictionaries, or
          finite_sas keyword arguments for Map3Engine.
      xi_settings = ell quadrature dictionary, theta_count, pair_weight.
      two_point_core = optional CoreTwoPoint for the unchanged CFASTPT model.
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
        self.accuracy = accuracy_settings(value=accuracy)
        self.map_indices = np.flatnonzero(rows.kind == "map3")
        self.xi_indices = np.flatnonzero(rows.kind != "map3")
        self.map_engine = None
        self.xi_engine = None
        self.groups = {}
        self._cached_model = None
        self._cached_ia_key = None
        self._cached_mean = None
        aperture_settings = deepcopy(aperture_settings)
        if aperture_estimator == "full_filter":
            projection = aperture_settings.get("projection_settings", {})
            aperture_settings["projection_settings"] = self.accuracy.projection_settings(
                baseline=projection,
            )
        if self.map_indices.size:
            self.map_rows = rows.select(indices=self.map_indices)
            if aperture_estimator == "finite_sas":
                self.map_engine = Map3Engine(
                    compiled=compiled, rows=self.map_rows, **aperture_settings,
                    accuracy=self.accuracy,
                )
            else:
                ell_settings = self.accuracy.quadrature_settings(
                    baseline=aperture_settings["ell"], stage="aperture_ell_boost",
                )
                phi_settings = self.accuracy.quadrature_settings(
                    baseline=aperture_settings["phi"], stage="aperture_phi_boost",
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
            ell_settings = self.accuracy.quadrature_settings(
                baseline=xi_settings["ell"], stage="xi_ell_boost",
            )
            ell, dell = quadrature(**ell_settings)
            theta_count = self.accuracy.count(
                xi_settings["theta_count"], stage="xi_theta_boost", integration=True,
            )
            if (theta_count != xi_settings["theta_count"]
                    and not isinstance(xi_settings["pair_weight"], str)):
                raise ValueError(
                    "Refining xi theta nodes requires new measured pair weights; "
                    "supply them at the desired baseline count or use uniform_area."
                )
            self.xi_engine = XiEngine(
                compiled=compiled, rows=selected, ell=ell, dell=dell,
                theta_count=theta_count,
                pair_weight=xi_settings["pair_weight"],
            )
        self._seal_geometry()

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
        for index in range(self.provider.nbin):
            allowed.add(f"m{index + 1}")
            allowed.add(f"dz{index + 1}")
        for name in parameters:
            if name not in allowed:
                raise ValueError(
                    f"Unknown prediction parameter {name}; check its spelling."
                )
        model, tables, metadata = self.provider.get(parameters=parameters)
        ia = dict(self.ia_defaults)
        for name in IA_NAMES:
            if name in parameters:
                ia[name] = parameters[name]
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
                aperture = self.map_engine.evaluate(
                    model=model, ia=ia,
                )
            else:
                aperture = self.full_aperture_mean(model=model, ia=ia)
            result[self.map_indices] = aperture
        if self.xi_indices.size:
            if self.two_point_core is None:
                xi = self.xi_engine.evaluate(model=model, ia=ia)
            else:
                physical_parameters = dict(self.provider.reference)
                physical_parameters.update(parameters)
                even, odd = self.two_point_core.spectra(
                    parameters=physical_parameters, tables=tables,
                    metadata=metadata, ia=ia,
                    ell=self.xi_engine.ell, bins=self.xi_engine.rows.bins[:, :2],
                )
                xi = self.xi_engine.evaluate_spectra(
                    spectra_e=even, spectra_b=odd,
                )
            result[self.xi_indices] = xi
        if not np.all(np.isfinite(result)):
            raise ValueError("Joint prediction contains a nonfinite value.")
        self._cached_model = model
        self._cached_ia_key = ia_key
        self._cached_mean = immutable_copy(value=result)
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
