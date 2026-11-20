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

import json
from pathlib import Path

import numpy as np
from des3pcf.survey import shifted_sources, reconstructed_sources, _source_edge_arrays
from theory.bihalofit.model import uniform_segments
from copy import deepcopy

from theory.bacco_bispectrum.response import FeedbackPair
from theory.bacco_bispectrum.response import BaccoFeedback, BACCO_PARAMETERS



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
      source_nz = optional fixed source sampling count, independent of CAMB z.
          Changing it rebuilds source kernels without a new Boltzmann solve.
          Sampling accuracy_boost and source_boost refine its intervals;
          they never enable an independent grid or reconstruction implicitly.
      source_interpolation, source_padding = explicit reconstruction choices;
          None reads saved provenance. They define an immutable source model.
      accuracy = immutable global/stage controls; replace provider.accuracy to
          prepare a new resolution from the same baseline in this interpreter.
      feedback = optional immutable FeedbackPair or BaccoFeedback producer.
          A fixed pair retains its ratios across cosmology. BACCO prepares
          new ratios after any cosmology or physical feedback change.
      pixel_window = optional scalar W(ell), sampled at integers from zero.
          Owned and fixed for this provider; changing it needs a new provider.
          Only three-point angular observables are smoothed, not ordinary xi.
      bispectrum_source = 'supplied'; the Python theory producer always owns B.
      bispectrum_table = explicit uniform z/logk/logu/v axes and an optional
          dense-reader boost, required only for the supplied route. logk is
          the natural logarithm of the middle triangle side in h/Mpc.
      variance_source = 'local' retains the reference smoothing calculation;
          'shared' supplies owned core FFTLog quantities and trusts input P.
      variance_growth = 'factorized' retains P(k,0)D(z)^2; 'evolving' requires
          the full supplied linear power and the shared variance source.
      camb_accuracy_boost, camb_k_per_logint = independent CAMB integration
          and transfer sampling controls; defaults one/zero preserve the
          previous producer. Both enter saved provenance and exact cache keys.
    Side effects:
      Retains the latest cosmology table and source-weighted C model.
      CAMB runs only after a physical parameter changes in camb mode.
    """

    def __init__(
        self, compiled, reference_tables, reference_metadata, reference_parameters,
        cosmology_mode="fixed_tables", kernel_nz=101, halofit_version="takahashi",
        source_readers=False, table_boost=8, fit_nz=33,
        accuracy=None, kmax_boltzmann=5.0, extrap_kmax=None,
        source_nz=None, source_interpolation=None, source_padding=None,
        feedback=None, pixel_window=None,
        bispectrum_source="supplied", bispectrum_table=None,
        variance_source="local", variance_growth="factorized",
        camb_accuracy_boost=1.0, camb_k_per_logint=0,
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
        if bispectrum_source != "supplied":
            raise ValueError("bispectrum_source must be supplied; C BiHalofit is removed.")
        if variance_source not in ("local", "shared"):
            raise ValueError("variance_source must be local or shared.")
        if variance_growth not in ("factorized", "evolving"):
            raise ValueError("variance_growth must be factorized or evolving.")
        if variance_source == "local" and variance_growth != "factorized":
            raise ValueError("Evolving variance growth requires variance_source='shared'.")
        self.variance_source = variance_source
        self.variance_growth = variance_growth
        self._variance_engine = None
        self._reference_variance_engine = None
        self._active_variance = None
        from theory.bihalofit.model import BiHalofitProducer, bispectrum_axes

        if bispectrum_table is None:
            bispectrum_table = bispectrum_axes(zmax=reference_tables["z"][-1])
        if not isinstance(bispectrum_table, dict):
            raise ValueError("The supplied route requires a bispectrum_table dictionary.")
        required_axes = {"z", "logk", "logu", "v"}
        if (not required_axes.issubset(bispectrum_table)
                or set(bispectrum_table) - required_axes - {"boost"}):
            raise ValueError(
                "bispectrum_table requires z, logk, logu, v and only optional boost."
            )
        BiHalofitProducer(**bispectrum_table)
        self.bispectrum_source = bispectrum_source
        self.bispectrum_table = deepcopy(x=bispectrum_table)
        self._baseline_producer = None
        self._baseline_resolution = None
        self.compiled = compiled
        if feedback is not None and not isinstance(feedback, (FeedbackPair, BaccoFeedback)):
            raise ValueError("feedback must be a FeedbackPair, BaccoFeedback or None.")
        self.feedback = feedback
        self._active_feedback = None
        self.pixel_window = None
        if pixel_window is not None:
            from .pixel_window import checked_window

            self.pixel_window = deepcopy(x=checked_window(values=pixel_window))
        self.base = deepcopy(x=validate_tables(tables=reference_tables))
        if "omega_m_z" not in self.base:
            raise ValueError(
                "The supplied baseline requires provider omega_m_z; regenerate "
                "the reference tables with include_bispectrum_background=True."
            )
        if float(reference_parameters.get("mnu", 0.0)) > 0.0:
            if "omega_m_z" not in self.base:
                raise ValueError(
                    "Massive-neutrino tables require provider omega_m_z; "
                    "regenerate with include_bispectrum_background=True."
                )
        self.base_metadata = deepcopy(x=reference_metadata)
        recorded_source = reference_metadata.get(
            "source_reconstruction", "linear_midpoint",
        )
        if source_interpolation is None:
            source_interpolation = recorded_source
        if source_interpolation not in ("linear_midpoint", "pchip_midpoint"):
            raise ValueError(
                "Source reconstruction must be linear_midpoint or pchip_midpoint."
            )
        if ("source_reconstruction" in reference_metadata
                and source_interpolation != recorded_source):
            raise ValueError(
                "Source reconstruction differs from the saved physical tables."
            )
        if source_padding is None:
            source_padding = reference_metadata.get(
                "source_padding", "zero_outside_native",
            )
        allowed_padding = ("zero_outside_native",)
        if source_interpolation == "pchip_midpoint":
            allowed_padding = ("zero_at_observer_and_background_edge",
                               "zero_at_measured_edges")
        if source_padding not in allowed_padding:
            raise ValueError(
                "Select the explicit padding policy for this reconstruction."
            )
        if ("source_padding" in reference_metadata
                and source_padding != reference_metadata["source_padding"]):
            raise ValueError("Source padding differs from the saved physical tables.")
        self.source_interpolation = source_interpolation
        self.source_padding = source_padding
        if source_padding == "zero_at_measured_edges":
            for name in ("source_native_low", "source_native_high"):
                if name not in self.base:
                    raise ValueError(
                        "Measured-edge sources require saved native FITS edges."
                    )
        if source_nz is None and "source_z" in self.base:
            source_nz = self.base["source_z"].size
        if source_interpolation == "pchip_midpoint" and source_nz is None:
            raise ValueError("PCHIP preparation requires an independent source_nz count.")
        self.source_nz = source_nz
        self.accuracy = self.compiled.init_3pcf_accuracy(value=accuracy)
        self.mode = cosmology_mode
        if int(kernel_nz) != kernel_nz or kernel_nz < 4:
            raise ValueError("kernel_nz must be an integer of at least four.")
        self.kernel_nz = int(kernel_nz)
        self.halofit_version = halofit_version
        self.kmax_boltzmann = kmax_boltzmann
        self.extrap_kmax = extrap_kmax
        self.camb_accuracy_boost = camb_accuracy_boost
        self.camb_k_per_logint = camb_k_per_logint
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
        self.reference = deepcopy(x=self.reference)
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
            expected = self.reference_producer_settings[1:3]
            recorded_limits = (
                self.base_metadata["kmax_boltzmann"],
                self.base_metadata["extrap_kmax"],
            )
            if recorded_limits != expected:
                raise ValueError(
                    "CAMB wavenumber limits differ from table provenance; "
                    "regenerate tables with the declared limits."
                )
        # Tables predating these controls were produced with one/zero.
        # A new precision request cannot silently reuse those old arrays.
        recorded_accuracy = camb_accuracy_settings(
            accuracy_boost=self.base_metadata.get("camb_accuracy_boost", 1.0),
            k_per_logint=self.base_metadata.get("camb_k_per_logint", 0),
        )
        if recorded_accuracy != self.reference_producer_settings[3:]:
            raise ValueError(
                "CAMB accuracy differs from table provenance; regenerate tables "
                "with the declared accuracy settings."
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
        self.model_metadata = None
        self._active_baseline = None

    def _producer_settings(self, parameters):
        """Resolve and validate both physical wavenumber limits for one cosmology.

        CAMB's two limits are in 1/Mpc; the saved k axis is in h/Mpc. Multiplying
        its last value by h compares physical wavenumbers without conflating
        the output sampling range with the range evolved by the Boltzmann code.

        Arguments:
          parameters = cosmology containing h.
        Returns:
          Tuple of halofit name, evolution limit, resolved extrapolation limit,
          CAMB accuracy boost and transfer sampling count.
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
        boost, sampling = camb_accuracy_settings(
            accuracy_boost=self.camb_accuracy_boost,
            k_per_logint=self.camb_k_per_logint,
        )
        return self.halofit_version, limit, extrapolation, boost, sampling

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
        if self.source_nz is not None:
            value = self.source_nz
            if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                    or not np.isfinite(value) or int(value) != value
                    or value < 4 or value > np.iinfo(np.int32).max):
                raise ValueError("source_nz must be an integer from four to INT_MAX.")
        elif "source_z" in self.base or self.source_interpolation == "pchip_midpoint":
            raise ValueError("An independent source model requires a source_nz count.")
        controls = self.compiled.init_3pcf_accuracy(value=self.accuracy)
        if self.variance_source == "local":
            for name in ("variance_mass_boost", "variance_a_boost",
                         "variance_radius_boost", "variance_inverse_boost"):
                if controls[name] != 1.0:
                    raise ValueError(f"{name} requires variance_source='shared'.")
        if self.source_nz is None and controls["source_boost"] != 1.0:
            raise ValueError("source_boost requires an explicit independent source_nz.")
        self._producer_settings(parameters=self.reference)
        if isinstance(self.feedback, BaccoFeedback):
            self.feedback.verify_weights()
        return (
            self.kernel_nz, self.source_readers, self.table_boost, self.fit_nz,
            self.compiled.init_3pcf_accuracy(value=self.accuracy),
            self.kmax_boltzmann, self.extrap_kmax,
            self.camb_accuracy_boost, self.camb_k_per_logint,
            self.source_nz, self.source_interpolation, self.source_padding,
        )

    def shared_variance(self, parameters):
        """Copy shared fit anchors for this cosmology and declared resolution.

        Source distributions, IA amplitudes and feedback responses do not
        enter the smoothing calculation. The owned result may therefore
        survive source reconstruction or another provider's core changes.
        """
        if self.variance_source == "local":
            return None, None
        from theory.bihalofit.model import fit_redshifts
        from theory.bihalofit.variance import SharedVariance

        if self._variance_engine is None:
            self._variance_engine = SharedVariance(compiled=self.compiled)
        controls = self.compiled.init_3pcf_accuracy(value=self.accuracy)
        settings = self.compiled.accuracy_model_settings(accuracy=controls,
            kernel_nz=self.kernel_nz, table_boost=self.table_boost,
            fit_nz=self.fit_nz,
        )
        anchors = fit_redshifts(zmax=self.cosmology_tables["z"][-1],
                                fit_nz=settings["fit_nz"],
                                source_readers=self.source_readers)
        snapshot, record = self._variance_engine.prepare(
            tables=self.cosmology_tables, parameters=parameters,
            metadata=self.cosmology_metadata, z=anchors, accuracy=controls,
            growth_mode=self.variance_growth,
        )
        fit = {}
        for name in ("z", "rnl", "neff", "sigma8"):
            fit[name] = snapshot[name]
        return deepcopy(x=fit), record

    def reference_variance(self):
        """Prepare the explicit legacy reference once per spectrum/resolution.

        This returns the old scalar-normalized fit for source comparisons.
        Its multiplier applies only to the three-point model's linear power;
        the ordinary two-point prediction retains the original provider arrays.
        """
        from theory.bihalofit.variance import ReferenceVariance

        if self._reference_variance_engine is None:
            self._reference_variance_engine = ReferenceVariance()
        controls = self.compiled.init_3pcf_accuracy(value=self.accuracy)
        settings = self.compiled.accuracy_model_settings(accuracy=controls,
            kernel_nz=self.kernel_nz, table_boost=self.table_boost,
            fit_nz=self.fit_nz,
        )
        return self._reference_variance_engine.prepare(
            tables=self.cosmology_tables, fit_nz=settings["fit_nz"],
            table_boost=settings["table_boost"], source_readers=self.source_readers,
        )

    def supplied_baseline(self, variance=None, tables=None):
        """Prepare a baseline table only when its physical inputs change.

        The coarse axes refine with accuracy_boost*bispectrum_boost. The
        dense-reader ratio is independent, so a global boost is not applied
        twice to the same intervals. The producer hashes power, growth,
        background and fit controls; source shifts and IA amplitudes do not
        rebuild its cosmology-only table. Previously returned owners remain
        valid when the provider replaces its latest model.

        Returns:
          a dictionary containing the bispectrum arrays and theory metadata.
        """
        from theory.bihalofit.model import BiHalofitProducer

        controls = self.compiled.init_3pcf_accuracy(value=self.accuracy)
        resolution = (controls["accuracy_boost"], controls["bispectrum_boost"])
        if resolution != self._baseline_resolution:
            axes = {}
            for name in ("z", "logk", "logu", "v"):
                original = self.bispectrum_table[name]
                count = self.compiled.accuracy_count(accuracy=controls,
                    baseline=len(original), stage="bispectrum_boost", endpoints=True,
                )
                axes[name] = np.linspace(
                    start=original[0], stop=original[-1], num=count,
                )
            axes["boost"] = self.bispectrum_table.get("boost", 2)
            self._baseline_producer = BiHalofitProducer(**axes)
            self._baseline_resolution = resolution
        settings = self.compiled.accuracy_model_settings(accuracy=controls,
            kernel_nz=self.kernel_nz, table_boost=self.table_boost,
            fit_nz=self.fit_nz,
        )
        if tables is None:
            tables = self.cosmology_tables
        return self._baseline_producer.prepare(
            tables=tables, fit_nz=settings["fit_nz"],
            table_boost=settings["table_boost"], source_readers=self.source_readers,
            variance=variance,
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
        self.accuracy = self.compiled.init_3pcf_accuracy(value=self.accuracy)
        numerical_key = self.numerical_key()
        cosmology = dict(self.reference)
        for name in COSMOLOGY_NAMES:
            if name in parameters:
                cosmology[name] = parameters[name]
        key = parameter_key(cosmology, names=COSMOLOGY_NAMES)
        physical_point = dict(self.reference)
        physical_point.update(parameters)
        feedback_key = None
        if isinstance(self.feedback, BaccoFeedback):
            feedback_key = self.feedback.parameter_key(parameters=physical_point)
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
                    "A cosmological parameter changed or a CAMB producer setting changed; "
                    "select camb or supply new tables."
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
                    camb_accuracy_boost=self.camb_accuracy_boost,
                    camb_k_per_logint=self.camb_k_per_logint,
                    source_z=self.base.get("source_z"),
                    include_bispectrum_background=True,
                )
            self.cosmology_tables = deepcopy(x=tables)
            self.cosmology_metadata = deepcopy(x=metadata)
            self.cosmology_key = key
            self.producer_key = producer_key
        shifts = np.zeros(self.nbin, dtype=np.float64)
        for index in range(self.nbin):
            shifts[index] = parameters.get(f"dz{index + 1}", 0.0)
        if not np.all(np.isfinite(shifts)):
            raise ValueError("All source redshift shifts must be finite.")
        source_key = tuple(float(value).hex() for value in shifts)
        combined = (
            key, producer_key, source_key, numerical_key, feedback_key,
        )
        if combined != self.model_key:
            active_feedback = self.feedback
            if isinstance(self.feedback, BaccoFeedback):
                active_feedback = self.feedback.prepare(
                    parameters=physical_point, tables=self.cosmology_tables,
                    metadata=self.cosmology_metadata, accuracy=self.accuracy,
                )
            tables = dict(self.cosmology_tables)
            metadata = dict(self.cosmology_metadata)
            if self.variance_source == "local":
                variance, multiplier, variance_record = self.reference_variance()
            else:
                variance, variance_record = self.shared_variance(parameters=cosmology)
                multiplier = 1.0
            model_tables = dict(tables)
            if multiplier != 1.0:
                model_tables["p_linear0"] = tables["p_linear0"] * multiplier
            baseline = self.supplied_baseline(variance=variance, tables=model_tables)
            if variance_record is not None:
                metadata["variance"] = variance_record
            if baseline is not None:
                metadata["bispectrum_source"] = "supplied"
                metadata["baseline_bispectrum"] = baseline["metadata"]
            if active_feedback is not None:
                # A plain JSON copy records the response of this point,
                # rather than exposing the producer's next cached table.
                metadata["feedback_response"] = active_feedback.record()
            if self.source_nz is None:
                # This path preserves the historical finite-grid operation
                # exactly. Merely storing native samples does not select a
                # different source reconstruction or sampling resolution.
                tables["sources"] = shifted_sources(
                    z=self.base["z"], sources=self.base["sources"], shifts=shifts,
                )
            else:
                count = self.compiled.accuracy_count(accuracy=self.accuracy,
                    baseline=self.source_nz, stage="source_boost", endpoints=True,
                )
                nodes = np.linspace(start=0.0, stop=self.base["z"][-1],
                                    num=count)
                native_z = self.base.get("source_native_z",
                                        self.base.get("source_z", self.base["z"]))
                native = self.base.get("source_native", self.base["sources"])
                tables["sources"], source_record = reconstructed_sources(
                    native_z=native_z, native_sources=native, source_z=nodes,
                    shifts=shifts, reconstruction=self.source_interpolation,
                    padding=self.source_padding,
                    native_low=self.base.get("source_native_low"),
                    native_high=self.base.get("source_native_high"),
                )
                tables["source_z"] = nodes
                metadata.update(source_record)
            # Source reconstruction is common to both spectra. Only the
            # legacy three-point reference receives the power multiplier;
            # returning raw power below keeps ordinary xi exactly unchanged.
            model_tables["sources"] = tables["sources"]
            if "source_z" in tables:
                model_tables["source_z"] = tables["source_z"]
            resolved_model = self.compiled.accuracy_model_settings(
                accuracy=self.accuracy,
                kernel_nz=self.kernel_nz,
                table_boost=self.table_boost,
                fit_nz=self.fit_nz,
            )
            response = {}
            if active_feedback is not None:
                response = dict(active_feedback.native)
                response["boost"] = self.compiled.accuracy_count(
                    accuracy=self.accuracy,
                    baseline=response["boost"],
                    stage="feedback_table_boost",
                )
            self.model = self.compiled.production.PhysicsModelWithBispectrum(
                k=model_tables["k"],
                p_linear0=model_tables["p_linear0"],
                z=model_tables["z"],
                p_nonlinear=model_tables["p_nonlinear"],
                chi=model_tables["chi"],
                dz_dchi=model_tables["dz_dchi"],
                growth=model_tables["growth"],
                sources=model_tables["sources"],
                omega_m=model_tables["omega_m"],
                sigma8=model_tables["sigma8"],
                omega_m_z=model_tables["omega_m_z"],
                source_z=model_tables.get("source_z"),
                los_z=self.compiled.accuracy_los_nodes(accuracy=self.accuracy, baseline=model_tables["los_z"]),
                baseline=baseline["native"],
                kernel_nz=resolved_model["kernel_nz"],
                table_boost=resolved_model["table_boost"],
                fit_nz=resolved_model["fit_nz"],
                source_readers=self.source_readers,
                feedback=response,
                pixel_window=self.pixel_window,
            )
            self.model_tables = deepcopy(x=tables)
            self.model_metadata = deepcopy(x=metadata)
            self.model_key = combined
            self._active_feedback = active_feedback
            self._active_baseline = baseline
            self._active_variance = variance
        return self.model, self.model_tables, self.model_metadata

    def covariance_model(self, parameters, inputs):
        """Return one model with halo variances from the same physical inputs.

        The shared route smooths total-matter power at the covariance's
        exact mass and LOS nodes. It preserves the existing M200m density
        convention and leaves Sheth-Tormen and Duffy coefficients with the
        caller. The returned input mapping owns its variance arrays; it
        remains usable with this model after another provider evaluation.
        The local reference route retains its declared k quadrature.
        """
        model, tables, metadata = self.get(parameters=parameters)
        prepared = dict(inputs)
        record = dict(metadata)
        if self.variance_source == "local":
            return model, prepared, record
        cosmology = dict(self.reference)
        for name in COSMOLOGY_NAMES:
            if name in parameters:
                cosmology[name] = parameters[name]
        # This is the covariance's existing critical-density constant in
        # (Msun/h)/(Mpc/h)^3. The shared reader converts query masses to
        # preserve these radii when its own density constant differs.
        density = 2.77536627e11 * tables["omega_m"]
        radial = model.radial()
        snapshot, variance_record = self._variance_engine.prepare(
            tables=self.cosmology_tables, parameters=cosmology,
            metadata=self.cosmology_metadata, z=radial["z"],
            mass=prepared["mass"], reference_rho=density,
            include_fit=False, accuracy=self.accuracy,
            growth_mode=self.variance_growth,
        )
        prepared["halo_z"] = snapshot["z"]
        prepared["halo_mass"] = snapshot["mass"]
        prepared["halo_sigma2"] = snapshot["halo_sigma2"]
        prepared["halo_dlnsigma_dlnm"] = snapshot["halo_dlnsigma_dlnm"]
        prepared["halo_reference_rho"] = snapshot["reference_rho"]
        record["halo_variance"] = variance_record
        return model, prepared, record




def load_tables(path):
    """Read a numerical NPZ file without executing serialized Python objects.

    Arguments:
      path = file written by save_tables or following the same schema.
    Returns:
      validated table dictionary and JSON metadata dictionary.
    Raises:
      ValueError for a wrong format version or inconsistent physical arrays.
      OSError when the file cannot be read.
    """
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("format") != "cosmolike-3pcf-tables-v1":
            raise ValueError("Unknown physics-table format; supply the v1 NPZ schema.")
        tables = {}
        for name in ('k', 'p_linear0', 'z', 'p_nonlinear', 'chi', 'dz_dchi', 'growth', 'sources', 'los_z'):
            tables[name] = archive[name]
        for name in ('omega_m', 'omega_de', 'w0', 'wa', 'ns', 'sigma8'):
            tables[name] = metadata[name]
        for name in ('p_linear', 'p_linear_cb', 'p_linear_nu') + ('source_z', 'source_native_z', 'source_native', 'source_native_low', 'source_native_high') + ('omega_m_z',):
            if name in archive.files:
                tables[name] = archive[name]
        validated = validate_tables(tables=tables)
    return validated, metadata


def save_tables(path, tables, metadata, overwrite=False):
    """Write validated physics tables and their preparation choices together.

    Arguments:
      path = destination ending in .npz.
      tables = physical arrays and scalars in the documented h units.
      metadata = JSON-compatible preparation settings and source information.
      overwrite = explicitly permit replacing an existing output.
    Returns:
      absolute output Path.
    Raises:
      ValueError for an invalid destination or input tables.
      FileExistsError when overwrite is False and output already exists.
    Side effects:
      Writes the requested file, after all validation succeeds.
    """
    destination = Path(path).resolve()
    if destination.suffix != ".npz":
        raise ValueError("Physics-table output must end in .npz.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {destination}; choose another path.")
    checked = validate_tables(tables=tables)
    record = dict(metadata)
    record["format"] = "cosmolike-3pcf-tables-v1"
    arrays = {}
    for name in ('k', 'p_linear0', 'z', 'p_nonlinear', 'chi', 'dz_dchi', 'growth', 'sources', 'los_z'):
        arrays[name] = checked[name]
    for name in ('omega_m', 'omega_de', 'w0', 'wa', 'ns', 'sigma8'):
        record[name] = checked[name]
    for name in ('p_linear', 'p_linear_cb', 'p_linear_nu') + ('source_z', 'source_native_z', 'source_native', 'source_native_low', 'source_native_high') + ('omega_m_z',):
        if name in checked:
            arrays[name] = checked[name]
    arrays["metadata_json"] = np.array(json.dumps(record, sort_keys=True))
    # An open file prevents NumPy from silently appending another suffix.
    mode = "wb" if overwrite else "xb"
    with destination.open(mode=mode) as stream:
        np.savez_compressed(stream, **arrays)
    return destination


def _camb_power_limits(k, h, kmax_boltzmann, extrap_kmax):
    """Separate the solved transfer range from the spectrum's extrapolated tail.

    The output k array uses h/Mpc, while both limits use physical 1/Mpc,
    matching the LSST likelihood. For example k=1000 h/Mpc at h=0.7 is a
    query at 700/Mpc, not a request to integrate Boltzmann modes to 700/Mpc.
    The default solver cutoff is supplied independently by the caller.
    """
    if (isinstance(kmax_boltzmann, (bool, np.bool_)) or not np.isfinite(kmax_boltzmann)
            or kmax_boltzmann <= 0.0):
        raise ValueError("kmax_boltzmann must be positive finite in 1/Mpc.")
    required = float(k[-1] * h)
    if extrap_kmax is None:
        extrap_kmax = max(250.0, 1.01 * required)
    if (isinstance(extrap_kmax, (bool, np.bool_)) or not np.isfinite(extrap_kmax)
            or extrap_kmax < required):
        raise ValueError(
            f"extrap_kmax must cover the table's {required} 1/Mpc endpoint; "
            "increase this interpolation limit, not the Boltzmann cutoff."
        )
    return float(kmax_boltzmann), float(extrap_kmax)


def camb_accuracy_settings(accuracy_boost=1.0, k_per_logint=0):
    """Validate CAMB's independent integration and transfer-sampling controls.

    Increasing output-table density cannot repair an inaccurate transfer
    solution. These controls act inside CAMB; they do not enlarge its
    maximum solved wavenumber or multiply the CosmoLike accuracy controls.

    Arguments:
      accuracy_boost = finite CAMB AccuracyBoost, at least one.
      k_per_logint = integer transfer samples per log-k interval; zero uses
          CAMB's existing automatic sampling, not zero transfer samples.
    Returns:
      float boost and integer sampling count suitable for exact cache keys.
    Raises:
      ValueError for Boolean, nonfinite or out-of-domain controls.
    """
    if (isinstance(accuracy_boost, (bool, np.bool_))
            or not isinstance(accuracy_boost, (int, float, np.integer, np.floating))
            or not np.isfinite(accuracy_boost) or accuracy_boost < 1.0):
        raise ValueError("camb_accuracy_boost must be finite and at least one.")
    if (isinstance(k_per_logint, (bool, np.bool_))
            or not isinstance(k_per_logint, (int, float, np.integer, np.floating))
            or not np.isfinite(k_per_logint)
            or int(k_per_logint) != k_per_logint or k_per_logint < 0
            or k_per_logint > np.iinfo(np.int32).max):
        raise ValueError("camb_k_per_logint must be an integer from zero to INT_MAX.")
    return float(accuracy_boost), int(k_per_logint)


def camb_tables(
    parameters, k, z, sources, los_z, halofit_version="takahashi",
    kmax_boltzmann=5.0, extrap_kmax=None, source_z=None,
    include_bispectrum_background=True,
    camb_accuracy_boost=1.0, camb_k_per_logint=0,
):
    """Calculate the physical tables with the installed CAMB library.

    The requested amplitude is sigma8. An initial linear calculation fixes
    As. The matter transfer functions are then reused to calculate power
    at that As, including its nonlinear correction; a second Boltzmann
    evolution is unnecessary because the background and spectral shape
    did not change. D(z)=sigma8_linear(z)/sigma8_linear(0)
    supplies the scale-independent growth used by the port. With massive
    neutrinos this is a chosen effective growth, not a replacement for
    scale-dependent transfer functions. The full supplied nonlinear power
    retains CAMB's neutrino dependence; the IA prescription and BiHalofit
    still inherit their stated scale-independent-growth approximation.

    Arguments:
      parameters = omega_m, omega_b, h, ns, sigma8, w0, wa; optional mnu=0
          in eV, num_massive_neutrinos=1 and nnu=3.046.
      k = positive increasing output nodes in h/Mpc, including extrapolated tails.
      z = increasing nodes beginning at zero, at most 150 transfer redshifts.
      sources = [nbin,nsource] densities per unit redshift.
      source_z = optional source nodes, independent of the <=150 power nodes.
      los_z = positive integration nodes within the z range.
      halofit_version = an installed CAMB nonlinear model name.
      kmax_boltzmann = largest solved transfer wavenumber in 1/Mpc; default 5.
      extrap_kmax = interpolation tail endpoint in 1/Mpc. None covers at least
          250/Mpc and 1.01 times the output table's physical endpoint.
      include_bispectrum_background = also save CAMB's Omega_m(z), defined
          by CDM, baryons and massive neutrinos relative to critical density.
          The supplied-bispectrum route requires this provider quantity;
          the required bispectrum producer and halo covariance use this background.
      camb_accuracy_boost = CAMB integration accuracy, independent of the
          CosmoLike controls; default one preserves existing calculations.
      camb_k_per_logint = CAMB transfer sampling per log-k interval; default
          zero preserves its automatic grid. Output k is a separate grid.
    Returns:
      validated physical tables and metadata recording CAMB and model choices.
    Raises:
      ValueError for unavailable cosmology choices or invalid grids.
      ImportError when CAMB is absent from the selected environment.
    Side effects:
      Runs CAMB serially; does not write files or alter an existing core state.
    """
    import camb

    camb_accuracy_boost, camb_k_per_logint = camb_accuracy_settings(
        accuracy_boost=camb_accuracy_boost, k_per_logint=camb_k_per_logint,
    )
    if not isinstance(include_bispectrum_background, (bool, np.bool_)):
        raise ValueError("include_bispectrum_background must be a Boolean.")
    required = ("omega_m", "omega_b", "h", "ns", "sigma8", "w0", "wa")
    for name in required:
        if name not in parameters or not np.isfinite(parameters[name]):
            raise ValueError(f"CAMB input requires finite {name}.")
    neutrino_mass = float(parameters.get("mnu", 0.0))
    neutrino_count = int(parameters.get("num_massive_neutrinos", 1))
    nnu = float(parameters.get("nnu", 3.046))
    if not np.isfinite(neutrino_mass) or neutrino_mass < 0.0 or neutrino_count < 1:
        raise ValueError(
            "Require finite mnu>=0 and at least one massive-neutrino species."
        )
    h = float(parameters["h"])
    omega_m = float(parameters["omega_m"])
    omega_b = float(parameters["omega_b"])
    if not 0.0 < omega_b < omega_m < 1.0 or h <= 0.0:
        raise ValueError("CAMB producer requires 0<omega_b<omega_m<1 and h>0.")
    k = np.asarray(k, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if (k.ndim != 1 or k.size < 2 or k[0] <= 0.0 or np.any(np.diff(k) <= 0.0)
            or z.ndim != 1 or z.size < 2 or z.size > 150 or z[0] != 0.0
            or np.any(np.diff(z) <= 0.0) or not np.all(np.isfinite(k))
            or not np.all(np.isfinite(z)) or parameters["sigma8"] <= 0.0):
        raise ValueError(
            "Require increasing positive k and 2..150 z nodes beginning at 0."
        )
    requested_extrapolation = extrap_kmax
    kmax_boltzmann, extrap_kmax = _camb_power_limits(
        k=k, h=h, kmax_boltzmann=kmax_boltzmann, extrap_kmax=extrap_kmax,
    )
    settings = camb.CAMBparams()
    settings.set_cosmology(
        H0=100.0 * h, ombh2=omega_b * h**2,
        omch2=(omega_m - omega_b) * h**2, mnu=neutrino_mass, omk=0.0,
        num_massive_neutrinos=neutrino_count, nnu=nnu,
    )
    # CAMB computes Omega_nu h^2 from its temperature/species convention.
    # Subtract that exact value so omega_m includes neutrinos only once.
    settings.omch2 = (omega_m - omega_b) * h**2 - settings.omnuh2
    if settings.omch2 <= 0.0:
        raise ValueError(
            "The requested matter density leaves no positive cold dark matter."
        )
    settings.set_dark_energy(
        w=parameters["w0"], wa=parameters["wa"], dark_energy_model="ppf",
    )
    # This seed amplitude is rescaled to the requested measured sigma8.
    amplitude_seed = 2.1e-9
    settings.InitPower.set_params(As=amplitude_seed, ns=parameters["ns"])
    settings.WantCls = False
    # Like the LSST likelihood, solve a bounded transfer range and let the
    # power interpolator extend its high-k tail. Output table resolution and
    # extent must not silently enlarge the Boltzmann differential equations.
    if camb_accuracy_boost != 1.0:
        settings.set_accuracy(AccuracyBoost=camb_accuracy_boost)
    settings.set_matter_power(redshifts=z[::-1], kmax=kmax_boltzmann,
                              k_per_logint=camb_k_per_logint, silent=True)
    # Keep the same transfer sampling in both calls. Switching the nonlinear
    # mode only after the amplitude calibration changes CAMB's k sampling
    # and shifts its numerical sigma8 even though the linear scaling is exact.
    settings.NonLinearModel.set_params(halofit_version=halofit_version)
    settings.NonLinear = camb.model.NonLinear_both
    first = camb.get_results(params=settings)
    initial_sigma8 = float(first.get_sigma8()[-1])
    amplitude = amplitude_seed * (parameters["sigma8"] / initial_sigma8)**2
    settings.InitPower.set_params(As=amplitude, ns=parameters["ns"])
    # This matter-only calculation has WantCls=False. CAMB can recalculate
    # primordial powers, sigma8 and nonlinear corrections from the retained
    # transfers; multiplying nonlinear P by an amplitude ratio would be wrong.
    first.power_spectra_from_transfer(initial_power_params=settings.InitPower)
    results = first
    # Keep both CAMB interpolation limits in physical 1/Mpc. Convert queries
    # and returned powers explicitly at the boundary, as in the LSST helper.
    physical_k = k * h
    power_conversion = h**3
    linear = results.get_matter_power_interpolator(
        nonlinear=False, hubble_units=False, k_hunit=False, extrap_kmax=extrap_kmax,
    )
    nonlinear = results.get_matter_power_interpolator(
        nonlinear=True, hubble_units=False, k_hunit=False, extrap_kmax=extrap_kmax,
    )
    cold_linear = results.get_matter_power_interpolator(
        nonlinear=False, var1="delta_nonu", var2="delta_nonu",
        hubble_units=False, k_hunit=False, extrap_kmax=extrap_kmax,
    )
    sigma_z = np.asarray(results.get_sigma8(), dtype=np.float64)[::-1]
    tables = {
        "k": k,
        "p_linear0": linear.P(z=0.0, kh=physical_k) * power_conversion,
        "z": z,
        "p_nonlinear": nonlinear.P(z=z, kh=physical_k) * power_conversion,
        "p_linear": linear.P(z=z, kh=physical_k) * power_conversion,
        "p_linear_cb": cold_linear.P(z=z, kh=physical_k) * power_conversion,
        "chi": results.comoving_radial_distance(z=z) * h,
        "dz_dchi": results.hubble_parameter(z=z) / (299792.458 * h),
        "growth": sigma_z / sigma_z[0],
        "sources": sources,
        "los_z": los_z,
        "omega_m": omega_m,
        "omega_de": 1.0 - omega_m,
        "w0": parameters["w0"],
        "wa": parameters["wa"],
        "ns": parameters["ns"],
        "sigma8": float(sigma_z[0]),
    }
    if source_z is not None:
        tables["source_z"] = source_z
    if neutrino_mass > 0.0:
        # The optional response conversion uses linear neutrino auto-power.
        # Keep it explicitly: inferring a tiny neutrino transfer
        # by subtracting two nearly equal cold/total amplitudes loses accuracy.
        neutrino_linear = results.get_matter_power_interpolator(
            nonlinear=False, var1="delta_nu", var2="delta_nu",
            hubble_units=False, k_hunit=False, extrap_kmax=extrap_kmax,
        )
        tables["p_linear_nu"] = (
            neutrino_linear.P(z=z, kh=physical_k) * power_conversion
        )
    # 299792.458 above is c in km/s; chi carries the extra h conversion.
    metadata = {
        "producer": "CAMB",
        "camb_version": camb.__version__,
        "halofit_version": halofit_version,
        "kmax_boltzmann": kmax_boltzmann,
        "camb_accuracy_boost": camb_accuracy_boost,
        "camb_k_per_logint": camb_k_per_logint,
        "extrap_kmax": extrap_kmax,
        "extrap_kmax_requested": requested_extrapolation,
        "camb_wavenumber_limit_units": "1/Mpc",
        "power_extrapolation": "CAMB log-power tail",
        "amplitude_method": "reuse_matter_transfers",
        "parameters": dict(parameters),
        "As": float(amplitude),
        "omega_nu_h2": float(settings.omnuh2),
        "sigma8_cb": float(np.asarray(results.get_sigmaR(
            R=8.0, z_indices=-1, var1="delta_nonu", var2="delta_nonu",
            hubble_units=True,
        )).item()),
        "sigma8_cb_definition": "CAMB linear delta_nonu top-hat R=8 Mpc/h at z=0",
        "growth_definition": "linear sigma8(z)/sigma8(0)",
        "massive_neutrino_effective_growth": neutrino_mass > 0.0,
        "flat_background": True,
    }
    if include_bispectrum_background:
        # Read each density from this CAMB solution. In particular, the
        # neutrino background is not replaced with a pressureless a^-3 law.
        matter = results.get_Omega(var="cdm", z=z)
        matter = matter + results.get_Omega(var="baryon", z=z)
        matter = matter + results.get_Omega(var="nu", z=z)
        tables["omega_m_z"] = np.asarray(matter, dtype=np.float64)
        metadata["omega_m_z_definition"] = (
            "CAMB Omega_cdm(z)+Omega_baryon(z)+Omega_nu_massive(z)"
        )
    return validate_tables(tables=tables), metadata



def validate_tables(tables):
    """Copy and check the unit-specific arrays needed by PhysicsModel.

    Rejecting a mismatched axis here prevents a Python exception from
    becoming a fatal C validation message. The C++ layer repeats its own
    checks because callers can also use the extension directly.

    Arguments:
      tables = dictionary containing ('k', 'p_linear0', 'z', 'p_nonlinear', 'chi', 'dz_dchi', 'growth', 'sources', 'los_z') and ('omega_m', 'omega_de', 'w0', 'wa', 'ns', 'sigma8').
    Returns:
      independent C-contiguous float64 arrays and scalar values.
    Raises:
      ValueError for missing fields, shapes, domains or normalization.
    """
    result = {}
    for name in ('k', 'p_linear0', 'z', 'p_nonlinear', 'chi', 'dz_dchi', 'growth', 'sources', 'los_z'):
        if name not in tables:
            raise ValueError(f"Physics tables lack {name}; regenerate the input file.")
        array = np.array(tables[name], dtype=np.float64, order="C", copy=True)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"Physics table {name} must be nonempty and finite.")
        result[name] = array
    for name in ('omega_m', 'omega_de', 'w0', 'wa', 'ns', 'sigma8'):
        if name not in tables or not np.isfinite(tables[name]):
            raise ValueError(f"Physics metadata requires finite scalar {name}.")
        result[name] = float(tables[name])
    for name in ("k", "z", "los_z"):
        axis = result[name]
        if axis.ndim != 1 or axis.size < 2 or np.any(np.diff(axis) <= 0.0):
            raise ValueError(f"{name} must be a strictly increasing 1D grid.")
    k = result["k"]
    z = result["z"]
    if k[0] <= 0.0 or z[0] != 0.0:
        raise ValueError(
            "k must be positive and the physical redshift grid must start at 0."
        )
    log10k = np.log10(k)
    uniform_segments(coordinates=log10k, name="log10(k)", maximum=1)
    uniform_segments(coordinates=z, name="background z")
    if result["los_z"][0] <= 0.0 or result["los_z"][-1] > z[-1]:
        raise ValueError("LOS nodes must lie within 0<z<=the physical table maximum.")
    for name in ("chi", "dz_dchi", "growth"):
        if result[name].shape != z.shape:
            raise ValueError(f"{name} must have one value per redshift node.")
    if result["p_linear0"].shape != k.shape:
        raise ValueError("p_linear0 must have one value per k node.")
    if result["p_nonlinear"].shape != (z.size, k.size):
        raise ValueError("p_nonlinear must have shape [nz,nk], with k contiguous.")
    for name in ('source_z', 'source_native_z', 'source_native', 'source_native_low', 'source_native_high'):
        if name in tables:
            array = np.array(tables[name], dtype=np.float64, order="C", copy=True)
            if array.size == 0 or not np.all(np.isfinite(array)):
                raise ValueError(f"Source table {name} must be nonempty and finite.")
            result[name] = array
    source_z = result.get("source_z", z)
    if (source_z.ndim != 1 or source_z.size < 2
            or np.any(np.diff(source_z) <= 0.0) or source_z[0] < 0.0
            or source_z[-1] > z[-1]):
        raise ValueError("source_z requires increasing nodes within the background.")
    uniform_segments(coordinates=source_z, name="source z")
    source = result["sources"]
    if source.ndim != 2 or source.shape[1] != source_z.size:
        raise ValueError("sources must have shape [nbin,nsource] on source_z (or z).")
    for name in ("p_linear0", "p_nonlinear", "dz_dchi", "growth"):
        if np.any(result[name] <= 0.0):
            raise ValueError(f"{name} must contain strictly positive values.")
    if (result["chi"][0] != 0.0 or np.any(np.diff(result["chi"]) <= 0.0)
            or abs(result["growth"][0] - 1.0) > 1.0e-12):
        raise ValueError("Require increasing chi with chi(0)=0 and D(0)=1.")
    if np.any(source < 0.0) or np.any(np.trapz(source, x=source_z, axis=1) <= 0.0):
        raise ValueError("Each n(z) must be nonnegative with positive integral.")
    if source_z[0] == 0.0 and np.any(source[:, 0] != 0.0):
        raise ValueError("Each source curve must have n(0)=0 at the observer.")
    native_fields = ("source_native_z" in result, "source_native" in result)
    if native_fields[0] != native_fields[1]:
        raise ValueError("Native sources require both source_native_z and source_native.")
    if native_fields[0]:
        native_z = result["source_native_z"]
        native = result["source_native"]
        if (native_z.ndim != 1 or native_z.size < 2 or native_z[0] < 0.0
                or np.any(np.diff(native_z) <= 0.0)
                or native.shape != (source.shape[0], native_z.size)
                or np.any(native < 0.0)
                or np.any(np.trapz(native, x=native_z, axis=1) <= 0.0)):
            raise ValueError(
                "Native source samples need increasing z and positive areas."
            )
    edge_fields = ("source_native_low" in result, "source_native_high" in result)
    if edge_fields[0] != edge_fields[1] or (edge_fields[0] and not native_fields[0]):
        raise ValueError(
            "Native source edges require both edges and native midpoint arrays."
        )
    if edge_fields[0]:
        _source_edge_arrays(native_z=result["source_native_z"],
                            native_low=result["source_native_low"],
                            native_high=result["source_native_high"])
    if result["omega_m"] <= 0.0 or result["sigma8"] <= 0.0:
        raise ValueError("omega_m and sigma8 must be positive.")
    for name in ('p_linear', 'p_linear_cb', 'p_linear_nu'):
        if name in tables:
            values = np.array(tables[name], dtype=np.float64, order="C", copy=True)
            if (values.shape != (z.size, k.size) or np.any(values <= 0.0)
                    or not np.all(np.isfinite(values))):
                raise ValueError(f"Optional {name} must be positive finite [nz,nk].")
            result[name] = values
    for name in ('omega_m_z',):
        if name in tables:
            values = np.array(tables[name], dtype=np.float64, order="C", copy=True)
            if (values.shape != z.shape or np.any(values <= 0.0)
                    or not np.all(np.isfinite(values))):
                raise ValueError(
                    f"Optional {name} must be positive finite [nz] on the background z grid."
                )
            result[name] = values
    return result
