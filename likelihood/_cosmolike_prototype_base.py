"""Joint DES shear likelihood, using the same call sequence as LSST Y1.

The YAML theory blocks supply distances, growth, power and the matter
bispectrum. This class passes those arrays to CosmoLike, sets the source
nuisance parameters, computes xi+/xi-/Map3 and calls the native likelihood.
It neither starts CAMB nor selects a matter-bispectrum fitting formula.
"""

from pathlib import Path
import sys

import numpy as np
from cobaya.likelihoods.base_classes import DataSetLikelihood
from getdist import IniFile

PROJECT = Path(__file__).resolve().parents[1]
CORE = PROJECT.parents[1] / "external_modules/code/cosmolike_core"
sys.path.insert(0, str(PROJECT / "interface"))
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(CORE))

import cosmolike_des_3pcf_interface as ci
from des3pcf.survey import (
    read_survey,
    read_joint_covariance,
    read_source_distributions,
    reconstructed_sources,
)


class _cosmolike_prototype_base(DataSetLikelihood):
    """Set native state once; evaluate the provider's current cosmology."""

    def initialize(self, probe):
        """Read fixed inputs and prepare numerical grids, without a cosmology."""
        directory = PROJECT / "data" if self.path is None else Path(self.path)
        ini = IniFile(str(directory / self.data_file))
        self.probe = probe
        self.source_ntomo = ini.int("source_ntomo")
        self.rows = read_survey(
            two_point_path=ini.relativeFileName("two_point_file"),
            aperture_path=ini.relativeFileName("aperture_file"),
            nbin=self.source_ntomo,
        )
        covariance = read_joint_covariance(
            path=ini.relativeFileName("cov_file"),
            rows=self.rows,
        )
        data = self.rows.values
        if ini.string("data_file", ""):
            data = np.loadtxt(fname=ini.relativeFileName("data_file"))
        indices = np.arange(self.rows.values.size, dtype=np.int64)
        if ini.string("mask_file", ""):
            mask = np.loadtxt(fname=ini.relativeFileName("mask_file"))
            if mask.shape != data.shape or not np.all((mask == 0) | (mask == 1)):
                raise ValueError("The mask must have one 0/1 value per data row.")
            indices = np.flatnonzero(mask)
        compression = np.empty((0, 0), dtype=np.float64)
        if ini.string("compression_file", ""):
            # Labels fix the input ordering. Numerical matrix checks and
            # application to data/covariance belong to init_joint_data.
            with np.load(
                file=ini.relativeFileName("compression_file"),
                allow_pickle=False,
            ) as saved:
                if not np.array_equal(
                    a1=saved["input_labels"],
                    a2=self.rows.labels[indices],
                ):
                    raise ValueError("Compression input labels differ from selected rows.")
                compression = saved["matrix"]

        ci.initial_setup()
        ci.set_omp_threads(n=self.omp_threads)
        ci.init_accuracy_boost(
            accuracy_boost=self.accuracyboost,
            integration_accuracy=self.integration_accuracy,
        )
        ci.init_fpt_internal_boost(internal_boost=self.fpt_internal_boost)
        ci.init_ntable_ell_internal(nell_internal=self.ell_internal)
        ci.init_probes(possible_probes="xi")
        ci.init_cosmo_runmode(is_linear=False)
        ci.init_include_halo_IA(include_halo_IA=0)
        ci.init_photoz_histogram_akima(
            enabled=0,
            preparation_factor=3,
        )
        ci.init_photoz_conventions(
            interpolation_type=self.interpolation_type,
            zmid_convention=1,
        )
        ci.init_source_sample_size(Ntomo=self.source_ntomo)
        # Pair bookkeeping expects a lens sample. No lens observable is
        # evaluated; the first source curve supplies this unused sample.
        ci.init_lens_sample_size(Ntomo=1)
        ci.init_IA(
            ia_model=1,
            ia_redshift_evolution=3,
            ia_code=0,
        )
        ci.reset_bary_struct()
        ci.init_joint_data(
            data=data,
            covariance=covariance,
            indices=np.asarray(indices, dtype=np.int32),
            compression=compression,
            mock_count=ini.int("mock_count", 0),
            parameter_count=ini.int("parameter_count", 0),
        )
        self.accuracy = ci.init_3pcf_accuracy(value=self.threepcf_accuracy)
        self.z_interp_2D = np.linspace(
            start=0.0,
            stop=self.zmax,
            num=self.power_nz,
        )
        self.log10k_interp_2D = np.linspace(
            start=self.log10k_min,
            stop=self.log10k_max,
            num=self.power_nk,
        )
        self.source_z = np.linspace(
            start=0.0,
            stop=self.zmax,
            num=ci.accuracy_count(
                accuracy=self.accuracy,
                baseline=self.source_nz,
                stage="source_boost",
                endpoints=True,
            ),
        )
        self.los_z = ci.accuracy_los_nodes(
            accuracy=self.accuracy,
            baseline=np.linspace(
                start=self.los_zmin,
                stop=self.zmax,
                num=self.los_nz,
            ),
        )
        (self.source_native_z, self.source_native,
         self.source_native_low, self.source_native_high) = read_source_distributions(
            path=ini.relativeFileName("nz_source_file"),
            nbin=self.source_ntomo,
            with_edges=True,
        )
        self.map_indices = np.flatnonzero(self.rows.kind == "map3")
        self.xi_indices = np.flatnonzero(self.rows.kind != "map3")
        theta_edges = np.exp(np.linspace(
            start=np.log(self.theta_min_arcmin),
            stop=np.log(self.theta_max_arcmin),
            num=self.ntheta + 1,
        )) * np.pi / (180.0 * 60.0)
        dphi = np.pi / self.phi_bins
        map_rows = self.rows.select(indices=self.map_indices)
        self.map_engine = ci.production.TransformPlan.for_aperture(
            bins=np.ascontiguousarray(
                a=map_rows.bins,
                dtype=np.int32,
            ),
            filters=np.ascontiguousarray(a=map_rows.angles),
            theta_lower=theta_edges[:-1],
            phi=(np.arange(self.phi_bins) + 0.5) * dphi,
            dlnt=float(np.log(theta_edges[1] / theta_edges[0])),
            dphi=dphi,
            ell_min=self.map_ell_min,
            ell_max=self.map_ell_max,
            minimum_size=self.minimum_fft_size,
            lmax=self.map_lmax,
            mmax=self.map_mmax,
            nmu=self.map_nmu,
            transform_options=self.transform_options,
            projection_mode="prepared",
            projection_settings=self.projection_settings,
            projection_batch_size=self.projection_batch_size,
            accuracy=self.accuracy,
        )
        ell_settings = ci.accuracy_quadrature_settings(
            accuracy=self.accuracy,
            baseline=self.xi_ell,
            stage="xi_ell_boost",
        )
        ell, dell = ci.quadrature(
            lower=ell_settings["lower"],
            upper=ell_settings["upper"],
            count=ell_settings["count"],
            logarithmic=ell_settings.get("logarithmic", False),
        )
        self.xi_rows = self.rows.select(indices=self.xi_indices)
        self.xi_ell = np.array(ell, copy=True)
        self.xi_dell = np.array(dell, copy=True)
        self.xi_theta, self.xi_weight = ci.xi_bin_weights(
            kind=self.xi_rows.kind.tolist(),
            angles=self.xi_rows.angles,
            count=ci.accuracy_count(
                accuracy=self.accuracy,
                baseline=self.xi_theta_count,
                stage="xi_theta_boost",
                integration=True,
            ),
            pair_weight="uniform_area",
        )
        self.xi_sign = np.where(self.xi_rows.kind == "xim", -1, 1).astype(np.int32)

    def get_requirements(self):
        """Ask Cobaya for theory products on the declared sampling grids."""
        return {
            "matter_bispectrum": None,
            "Hubble": {"z": self.z_interp_2D},
            "comoving_radial_distance": {"z": self.z_interp_2D},
            "sigma8_z": {"z": self.z_interp_2D},
            "Omega_b": {"z": self.z_interp_2D},
            "Omega_cdm": {"z": self.z_interp_2D},
            "Omega_nu_massive": {"z": self.z_interp_2D},
            "Pk_interpolator": {
                "z": self.z_interp_2D,
                "k_max": self.kmax_boltzmann,
                "nonlinear": (False, True),
                "vars_pairs": [("delta_tot", "delta_tot"),
                               ("delta_nonu", "delta_nonu")],
            },
        }

    def set_cosmo_related(self):
        """Pass the current provider arrays to the native cosmology setters.

        Cobaya uses k in 1/Mpc and P in Mpc^3. CosmoLike uses h/Mpc and
        (Mpc/h)^3, hence k_physical=h*k and P_h=h^3*P_physical. The requested
        CAMB transfer limit stays at kmax_boltzmann; its interpolator extends
        the supplied spectrum to the much larger projection-grid endpoint.
        """
        ci.set_omp_threads(n=self.omp_threads)
        hubble = self.provider.get_Hubble(
            z=self.z_interp_2D,
            units="km/s/Mpc",
        )
        self.h = float(hubble[0]) / 100.0
        k = self.h * 10.0**self.log10k_interp_2D
        linear = self.provider.get_Pk_interpolator(
            var_pair=("delta_tot", "delta_tot"),
            nonlinear=False,
            extrap_kmin=float(k[0]),
            extrap_kmax=max(self.kmax_boltzmann, 1.01 * float(k[-1])),
        )
        nonlinear = self.provider.get_Pk_interpolator(
            var_pair=("delta_tot", "delta_tot"),
            nonlinear=True,
            extrap_kmin=float(k[0]),
            extrap_kmax=max(self.kmax_boltzmann, 1.01 * float(k[-1])),
        )
        cold = self.provider.get_Pk_interpolator(
            var_pair=("delta_nonu", "delta_nonu"),
            nonlinear=False,
            extrap_kmin=float(k[0]),
            extrap_kmax=max(self.kmax_boltzmann, 1.01 * float(k[-1])),
        )
        self.p_linear = linear.P(
            z=self.z_interp_2D,
            k=k,
        ) * self.h**3
        self.p_nonlinear = nonlinear.P(
            z=self.z_interp_2D,
            k=k,
        ) * self.h**3
        p_linear_cb = cold.P(
            z=self.z_interp_2D,
            k=k,
        ) * self.h**3
        self.chi = self.provider.get_comoving_radial_distance(
            z=self.z_interp_2D,
        ) * self.h
        self.dz_dchi = hubble / (299792.458 * self.h)
        sigma = self.provider.get_sigma8_z(z=self.z_interp_2D)
        self.sigma8 = float(sigma[0])
        self.growth = sigma / self.sigma8
        omega_b = self.provider.get_Omega_b(z=self.z_interp_2D)
        omega_c = self.provider.get_Omega_cdm(z=self.z_interp_2D)
        omega_nu = self.provider.get_Omega_nu_massive(z=self.z_interp_2D)
        self.omega_m_z = omega_b + omega_c + omega_nu
        self.baseline = self.provider.get_result("matter_bispectrum")
        # The native constructor validates the four axes and their sizes.
        # A supplied B uses (Mpc/h)^6 on z, ln(k), ln(short/middle), v.
        if "native" not in self.baseline:
            raise ValueError("matter_bispectrum must supply its native B table.")
        ci.set_cosmology(
            omegam=float(self.omega_m_z[0]),
            H0=100.0 * self.h,
            log10k_2D=self.log10k_interp_2D,
            z_2D=self.z_interp_2D,
            lnP_linear=np.log(self.p_linear).flatten(order="F"),
            lnP_nonlinear=np.log(self.p_nonlinear).flatten(order="F"),
            G=self.growth * (1.0 + self.z_interp_2D),
            z_1D=self.z_interp_2D,
            chi=self.chi,
            omegab=float(omega_b[0]),
            z_G=self.z_interp_2D,
            omegan2=float(omega_nu[0]) * self.h**2,
            lnP_linear_cb=np.log(p_linear_cb).flatten(order="F"),
        )

    def set_source_related(self, **params):
        """Use CosmoLike IA amplitudes and the same shifted sources in both probes."""
        shifts = np.zeros(self.source_ntomo)
        self.calibration = np.zeros(self.source_ntomo)
        for index in range(self.source_ntomo):
            shifts[index] = params.get(f"dz{index + 1}", 0.0)
            self.calibration[index] = params.get(f"m{index + 1}", 0.0)
        self.sources, _ = reconstructed_sources(
            native_z=self.source_native_z,
            native_sources=self.source_native,
            source_z=self.source_z,
            shifts=shifts,
            reconstruction=self.source_interpolation,
            padding=self.source_padding,
            native_low=self.source_native_low,
            native_high=self.source_native_high,
        )
        source_table = np.column_stack((self.source_z, self.sources.T))
        ci.set_source_sample(nofz=source_table)
        ci.set_lens_sample(nofz=source_table[:, :2].copy())
        ci.init_ntomo_powerspectra()
        # These sources already contain the shift. Final xi and Map3
        # operators apply calibration once per shear, after projection.
        ci.set_nuisance_shear_photoz(bias=np.zeros(self.source_ntomo))
        ci.set_nuisance_shear_calib(M=np.zeros(self.source_ntomo))
        a1 = np.zeros(self.source_ntomo)
        a2 = np.zeros(self.source_ntomo)
        bta = np.zeros(self.source_ntomo)
        a1[:2] = [params.get("a1", 0.0), params.get("eta1", 0.0)]
        a2[:2] = [params.get("a2", 0.0), params.get("eta2", 0.0)]
        bta[0] = params.get("bta", 0.0)
        ci.set_nuisance_ia(
            A1=a1,
            A2=a2,
            B_TA=bta,
        )
        self.ia = {
            "a1": a1[0],
            "eta1": a1[1],
            "a2": a2[0],
            "eta2": a2[1],
            "bta": bta[0],
            "c1rho": 0.01389,
            "zpivot": 0.62,
            "power_mode": self.power_mode,
            "ct": 0.0,
            "fb": 0.0,
        }
        # One supplied spectrum normalization belongs to the legacy
        # bispectrum producer's IA convention. Other producers may omit
        # it (one). It never rescales ordinary two-point CAMB power.
        normalization = self.baseline.get("metadata", {}).get(
            "linear_power_normalization", 1.0,
        )
        settings = ci.accuracy_model_settings(
            accuracy=self.accuracy,
            kernel_nz=self.kernel_nz,
            table_boost=self.table_boost,
            fit_nz=self.fit_nz,
        )
        self.model = ci.production.PhysicsModelWithBispectrum(
            k=10.0**self.log10k_interp_2D,
            p_linear0=self.p_linear[0] * normalization,
            z=self.z_interp_2D,
            p_nonlinear=self.p_nonlinear,
            chi=self.chi,
            dz_dchi=self.dz_dchi,
            growth=self.growth,
            sources=self.sources,
            source_z=self.source_z,
            los_z=self.los_z,
            omega_m=float(self.omega_m_z[0]),
            omega_m_z=self.omega_m_z,
            sigma8=self.sigma8,
            baseline=self.baseline["native"],
            kernel_nz=settings["kernel_nz"],
            table_boost=settings["table_boost"],
            fit_nz=settings["fit_nz"],
            source_readers=False,
        )

    def compute_logp(self, datavector):
        """Evaluate the native joint density; Cobaya adds the parameter priors.

        The normalized Gaussian differs from -chi2/2 by a fixed constant.
        Retaining it permits evidence comparisons with the optional
        normalized finite-simulation likelihood chosen by the dataset.
        """
        return ci.compute_joint_loglike(prediction=datavector)

    def logp(self, **params):
        """Return the likelihood of the supplied nuisance-parameter point."""
        return self.compute_logp(datavector=self.get_datavector(**params))

    def get_datavector(self, **params):
        """Return every xi+/xi-/Map3 row in the input measurement order."""
        return np.asarray(self.internal_get_datavector(**params))

    def internal_get_datavector(self, **params):
        """Set physical inputs, project both probes and assemble the full vector."""
        self.set_cosmo_related()
        self.set_source_related(**params)
        ci.set_omp_threads(n=self.omp_threads)
        result = np.empty(self.rows.values.size)
        even, odd = ci.shear_power_rows(
            ell=self.xi_ell,
            bins=self.xi_rows.bins[:, :2],
        )
        result[self.xi_indices] = ci.production.xi_from_power(
            ell=self.xi_ell,
            dell=self.xi_dell,
            theta=self.xi_theta,
            theta_weight=self.xi_weight,
            sign=self.xi_sign,
            bins=np.ascontiguousarray(self.xi_rows.bins[:, :2], dtype=np.int32),
            spectra_e=even,
            spectra_b=odd,
            calibration=self.calibration,
        )
        result[self.map_indices] = self.map_engine.aperture_vector(
            model=self.model,
            ia=self.ia,
            calibration=self.calibration,
        )
        return result
