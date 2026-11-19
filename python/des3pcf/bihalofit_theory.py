"""Optional Cobaya Theory adapter for the supplied BiHalofit baseline.

Loading this module does not enable the route. A Cobaya configuration must
explicitly add BiHalofitTheory and another component must request
bihalofit_baseline. The existing project's direct-CAMB pipeline can use the
same BiHalofitProducer without going through this adapter.

The provider supplies linear total-matter power, sigma8(z), H0 and the
baryon/CDM/massive-neutrino density fractions. Summing the three fractions
uses the provider's expansion and neutrino treatment; no radiation-free
w0-wa Friedmann equation is reconstructed. h-units are converted exactly
once at this boundary: k_physical=h*k, P_h=h^3*P_physical.
"""

import numpy as np
from cobaya.theory import Theory

from .bihalofit import BiHalofitProducer, _axis, _integer, fit_redshifts


class BiHalofitTheory(Theory):
    """Compute an explicitly requested uniform ln B table from provider arrays.

    The four baseline axes are required configuration values, with the same
    meanings as BiHalofitProducer. background_z defaults to baseline_z;
    supply it separately when the C model uses a different background grid.
    power_k is an optional h/Mpc sampling axis; the default covers 1e-5..1e3
    with 256 logarithmic samples. kmax_boltzmann=5 is the provider's actual
    transfer-computation limit in 1/Mpc, independent of that output grid.
    Higher k is the power interpolator's explicit power-law extension.

    Refinement controls retain the Python/C comparison defaults. None is a
    declaration that a survey data vector or Fisher derivative has converged.
    """

    baseline_z = None
    baseline_logk = None
    baseline_logu = None
    baseline_v = None
    background_z = None
    power_k = None
    baseline_boost = 2
    baseline_domain = "error"
    block_size = 65536
    fit_nz = 33
    table_boost = 8
    source_readers = False
    variance_nodes = 1000
    slope_nodes = 2000
    radius_rtol = 1.0e-5
    kmax_boltzmann = 5.0
    variance_source = "local"
    variance_growth = "factorized"
    variance_accuracy = None

    def initialize(self):
        """Own explicit geometry before asking the provider for any products."""
        if self.variance_source not in ("local", "shared"):
            raise ValueError("variance_source must be local or shared.")
        if self.variance_growth not in ("factorized", "evolving"):
            raise ValueError("variance_growth must be factorized or evolving.")
        if self.variance_source == "local" and self.variance_growth != "factorized":
            raise ValueError("Evolving variance requires the shared engine.")
        if self.variance_source == "local" and self.variance_accuracy is not None:
            raise ValueError("variance_accuracy requires the shared engine.")
        self._variance_engine = None
        names = ("baseline_z", "baseline_logk", "baseline_logu", "baseline_v")
        for name in names:
            if getattr(self, name) is None:
                raise ValueError(f"BiHalofitTheory requires explicit {name}.")
        self._producer = BiHalofitProducer(
            z=self.baseline_z, logk=self.baseline_logk,
            logu=self.baseline_logu, v=self.baseline_v,
            boost=self.baseline_boost, domain=self.baseline_domain,
            block_size=self.block_size)
        redshift = self.background_z
        if redshift is None:
            redshift = self.baseline_z
        self._background_z = _axis(values=redshift, name="background_z")
        if self._background_z[0] != 0.0:
            raise ValueError("BiHalofitTheory background_z must begin at zero.")
        if self._background_z[-1] < self._producer.z[-1]:
            raise ValueError("Background must cover all baseline redshifts.")
        if self.power_k is None:
            log10k = np.linspace(start=-5.0, stop=3.0, num=256)
            self._power_k = 10.0**log10k
        else:
            self._power_k = _axis(values=self.power_k, name="power_k")
        if self._power_k[0] <= 0.0:
            raise ValueError("BiHalofitTheory power_k must be positive.")
        self.fit_nz = _integer(value=self.fit_nz, name="fit_nz", minimum=4)
        if not np.isfinite(self.kmax_boltzmann) or self.kmax_boltzmann <= 0.0:
            raise ValueError("kmax_boltzmann must be finite and positive.")
        fit_z = fit_redshifts(zmax=self._background_z[-1],
                             fit_nz=self.fit_nz,
                             source_readers=self.source_readers)
        # Omega requests include exact coefficient anchors and the C model's
        # background nodes. This avoids inserting an extra interpolation of
        # the provider's density fractions before fitting coefficients.
        self._omega_z = np.unique(np.concatenate((self._background_z, fit_z)))
        self._fit_z = fit_z

    def get_requirements(self):
        """Declare the cosmology dependencies for Cobaya cache validation."""
        return {
            "ns": None,
            "Hubble": {"z": [0.0]},
            "sigma8_z": {"z": self._background_z.tolist()},
            "Omega_b": {"z": self._omega_z.tolist()},
            "Omega_cdm": {"z": self._omega_z.tolist()},
            "Omega_nu_massive": {"z": self._omega_z.tolist()},
            "Pk_interpolator": {
                "z": self._background_z.tolist(),
                "k_max": float(self.kmax_boltzmann),
                "nonlinear": False,
                "vars_pairs": [["delta_tot", "delta_tot"]],
            },
        }

    def get_can_provide(self):
        """Name the baseline and matching background products for consumers."""
        return ["bihalofit_baseline", "bihalofit_background"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        """Build from one provider state and save independently owned arrays."""
        hubble = np.asarray(self.provider.get_Hubble(
            z=np.asarray([0.0]), units="km/s/Mpc"), dtype=np.float64)
        if hubble.size != 1 or not np.isfinite(hubble[0]) or hubble[0] <= 0.0:
            raise ValueError("The provider must supply one positive H0.")
        little_h = float(hubble[0]) / 100.0
        physical_k = little_h * self._power_k
        interpolator = self.provider.get_Pk_interpolator(
            var_pair=("delta_tot", "delta_tot"), nonlinear=False,
            extrap_kmin=float(physical_k[0]),
            extrap_kmax=max(float(self.kmax_boltzmann),
                            1.01 * float(physical_k[-1])))
        linear0 = np.asarray(interpolator.P(z=0.0, k=physical_k),
                             dtype=np.float64) * little_h**3
        sigma = np.asarray(self.provider.get_sigma8_z(z=self._background_z),
                           dtype=np.float64)
        if (sigma.shape != self._background_z.shape
                or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0)):
            raise ValueError("The provider must supply positive sigma8(z).")
        omega_b = np.asarray(self.provider.get_Omega_b(z=self._omega_z),
                             dtype=np.float64)
        omega_cdm = np.asarray(self.provider.get_Omega_cdm(z=self._omega_z),
                               dtype=np.float64)
        omega_nu = np.asarray(
            self.provider.get_Omega_nu_massive(z=self._omega_z),
            dtype=np.float64)
        for name, fraction in (("baryons", omega_b), ("CDM", omega_cdm),
                               ("massive neutrinos", omega_nu)):
            if (fraction.shape != self._omega_z.shape
                    or np.any(~np.isfinite(fraction))
                    or np.any(fraction < 0.0)):
                raise ValueError(f"Invalid provider {name} fractions.")
        tables = {
            "k": self._power_k,
            "p_linear0": linear0,
            "z": self._background_z,
            "growth": sigma / sigma[0],
            "omega_m_redshift": self._omega_z,
            "omega_m_z": omega_b + omega_cdm + omega_nu,
            "ns": float(self.provider.get_param("ns")),
            "sigma8": float(sigma[0]),
        }
        variance = None
        variance_record = None
        if self.variance_source == "shared":
            from .variance import SharedVariance

            if self._variance_engine is None:
                # The project setup places its compiled interface on the
                # import path, as it does for the likelihood. This optional
                # dependency belongs only to the shared-smoothing route.
                import cosmolike_des_3pcf_interface as compiled

                self._variance_engine = SharedVariance(compiled=compiled)
            tables["omega_m"] = float(tables["omega_m_z"][0])
            if self.variance_growth == "evolving":
                tables["p_linear"] = np.asarray(
                    interpolator.P(z=self._background_z, k=physical_k),
                    dtype=np.float64,
                ) * little_h**3
            parameters = {
                "omega_b": float(omega_b[0]),
                "h": little_h,
            }
            metadata = {"omega_nu_h2": float(omega_nu[0]) * little_h**2}
            snapshot, variance_record = self._variance_engine.prepare(
                tables=tables, parameters=parameters, metadata=metadata,
                z=self._fit_z, accuracy=self.variance_accuracy,
                growth_mode=self.variance_growth,
            )
            variance = {}
            for name in ("z", "rnl", "neff", "sigma8"):
                variance[name] = snapshot[name]
        baseline = self._producer.prepare(
            tables=tables, fit_nz=self.fit_nz, table_boost=self.table_boost,
            source_readers=self.source_readers,
            variance_nodes=self.variance_nodes, slope_nodes=self.slope_nodes,
            radius_rtol=self.radius_rtol, variance=variance)
        state["bihalofit_baseline"] = baseline
        state["bihalofit_variance_record"] = variance_record
        state["bihalofit_background"] = {
            "z": baseline.background_z,
            "omega_m_z": baseline.background_omega_m_z,
        }
        return True

    def get_bihalofit_baseline(self):
        """Return the immutable full-B table for the active cached state."""
        return self.current_state["bihalofit_baseline"]

    def get_bihalofit_background(self):
        """Return a new mapping of the independently owned background arrays."""
        return dict(self.current_state["bihalofit_background"])
