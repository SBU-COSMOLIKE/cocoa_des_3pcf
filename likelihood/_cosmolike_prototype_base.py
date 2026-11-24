"""Defines the Cobaya likelihood base class of the joint DES shear statistics.

Cobaya, the sampler framework, builds one likelihood object for each entry of
the likelihood block of a yaml file (here des_3pcf.joint_shear) and calls its
logp method at every sampled point. The subclass only names the probe; this
base class does all the work. The physics runs in the compiled cosmolike
library, imported below as ci (cosmolike_des_3pcf_interface, built from
interface/interface.cpp), plus its Python production layer ci.production for
the third-order aperture statistic.

Terms used in this file:
  data vector  = the 1D array of every measured statistic, in the measured row
                 order: the shear two-point correlation functions xi_+ and
                 xi_- (one entry per source-source bin pair and angular bin)
                 and the third-order aperture-mass moment Map3 (one entry per
                 source bin triple and aperture filter scale).
  rows         = the labeled row table of the data vector (one label, row
                 kind "xip"/"xim"/"map3", bin indices and angle per entry),
                 read by read_survey from the measurement files.
  mask         = one 0/1 flag per data-vector entry; entries flagged 0 (scale
                 cuts) are excluded from the chi2.
  .dataset     = small text file with one "key = value" line per setting: the
                 measurement, covariance, mask and n(z) file names and the
                 source bin count.
  compression  = an optional fixed matrix T that multiplies data and theory
                 (and the covariance by T on the left and T transpose on the
                 right) so the chi2 runs over a few linear modes instead of
                 every row; likelihood/compression.py trains and saves it.
  nuisance parameters = survey systematics sampled together with cosmology,
                 read from Cobaya with the DES_ prefix of the des_y3
                 project: photo-z shifts DES_DZ_S1..DES_DZ_S4, shear
                 calibrations DES_M1..DES_M4, and the intrinsic-alignment
                 slots DES_A1_1, DES_A1_2 (tidal-alignment amplitude and its
                 redshift exponent), DES_A2_1, DES_A2_2 (tidal torquing) and
                 DES_BTA_1 (density weighting); likelihood/params_source.yaml
                 gives their default priors.

What happens at one sampled point (logp):
  get_datavector
    -> set_cosmo_related : P(k, z), growth, distances and the matter
                           bispectrum table from the provider to cosmolike
    -> set_source_related: source n(z) and nuisance parameters to cosmolike,
                           plus the Python-side Map3 physics model
    -> ci.shear_power_rows + ci.production.xi_from_power: the xi rows
    -> map_engine.aperture_vector: the Map3 rows
  compute_logp: the native joint Gaussian log-density of the full vector

The yaml theory blocks supply distances, growth, power and the matter
bispectrum. This class neither starts CAMB nor selects a matter-bispectrum
fitting formula.
"""

from pathlib import Path
import sys

import numpy as np
from cobaya.likelihoods.base_classes import DataSetLikelihood
from getdist import IniFile

# PROJECT = projects/des_3pcf; CORE = the cosmolike core checkout. The two
# sys.path entries make the compiled interface (interface/) and
# cosmolike_notebook_utils importable without an installation step.
PROJECT = Path(__file__).resolve().parents[1]
CORE = PROJECT.parents[1] / "external_modules/code/cosmolike_core"
sys.path.insert(0, str(PROJECT / "interface"))
sys.path.insert(0, str(CORE))

# ci = the compiled cosmolike library of this project
# (interface/cosmolike_des_3pcf_interface.so, built with pybind11); every
# ci.<name> call runs C/C++ code and changes or reads cosmolike's global
# state. ci.production is its Python production layer (the Map3 transform
# plan and physics model built on the same compiled kernels).
import cosmolike_des_3pcf_interface as ci
# survey = likelihood/survey.py, the readers of the measurement, the
# covariance and the source n(z), imported relative to this package.
from .survey import (
    read_survey,
    read_joint_covariance,
    read_source_distributions,
    reconstructed_sources,
)

# Prefix of every nuisance-parameter name this likelihood reads from Cobaya,
# e.g. DES_DZ_S1 (photo-z shift of source bin 1) or DES_M1 (shear
# calibration of source bin 1), as in the des_y3 project. It must match the
# names in params_source.yaml.
survey = "DES"


class _cosmolike_prototype_base(DataSetLikelihood):
    """Evaluates the joint DES shear likelihood of one probe choice.

    The subclass names the probe and passes it to initialize. Cobaya then
    calls, once, get_requirements (the theory quantities to compute at every
    point) and, at every sampled point, logp(**params) with the
    nuisance-parameter values.

    Options come from the likelihood yaml (likelihood/joint_shear.yaml,
    overridden by the likelihood block of an example yaml); Cobaya stores
    each key as an attribute of self before initialize runs:
      path, data_file  = directory (default: the project's data/) and name
                         of the .dataset file
      omp_threads      = OpenMP thread count of cosmolike's parallel loops
      accuracyboost    = multiplies cosmolike's internal sampling (1 = default)
      integration_accuracy = cosmolike's integration accuracy level (0 = default)
      fpt_internal_boost = sampling boost of the FAST-PT intrinsic-alignment
                         tables
      ell_internal     = internal multipole table size of the two-point core
      interpolation_type = n(z) interpolation convention of the photo-z code
      threepcf_accuracy = named accuracy preset of the three-point stages
                         (null = the default preset); init_3pcf_accuracy
                         resolves it into the stage boosts used below
      zmax             = largest redshift of every grid built here
      power_nz, power_nk = z and k node counts of the P(k, z) tables
      log10k_min, log10k_max = log10 k range [h/Mpc] of those tables
      kmax_boltzmann   = k_max [1/Mpc] requested from CAMB's interpolator
      source_nz        = node count of the common source n(z) grid
      kernel_nz        = node count of the Map3 lensing-kernel tables
      los_zmin, los_nz = range start and node count of the line-of-sight
                         integration grid of the Map3 projection
      table_boost, fit_nz = Map3 physics-table refinement and the node count
                         of its fitted n(z) representation
      source_interpolation, source_padding = n(z) reconstruction method and
                         its behavior outside the measured redshift range
      theta_min_arcmin, theta_max_arcmin, ntheta = the log-spaced side
                         bins of the Map3 measurement's three-point
                         function (DES Y3: 20 bins from 0.5 to 80 arcmin)
      theta_lower_cut_arcmin = the measurement's scale cut: only side bins
                         whose lower edge is at least this value enter
                         the Map3 rows (DES Y3: 8 arcmin, 9 bins kept)
      phi_bins         = azimuthal node count of the aperture transform
      map_ell_min, map_ell_max = multipole range of the aperture transform
      minimum_fft_size = smallest FFT length the transform plan may choose
      map_lmax, map_mmax, map_nmu = multipole, azimuthal and angular node
                         caps of the bispectrum projection
      transform_options, projection_settings, projection_batch_size =
                         dictionaries and batch size passed through to the
                         aperture transform plan (meanings in the yaml)
      xi_ell           = lower/upper/count/logarithmic of the multipole
                         quadrature that turns C_ell into xi
      xi_theta_count   = angular nodes per bin of the xi bin average
      power_mode       = IA power-spectrum prescription selector of the
                         Map3 model (meanings in the yaml)
      external_baryon_response = False: gravity-only P and B; True: request
                         the paired baryonic responses R_P and R_B (the
                         ratios feedback/gravity-only of the power spectrum
                         and bispectrum) from a theory block
      external_baryon_suppression = False: gravity-only P; True: request the
                         power-only response R_P = P_feedback/P_DMO from the
                         bfmt theory block and apply it to the nonlinear
                         power of the xi rows. A power-only model supplies
                         no bispectrum response R_B, so the dataset mask
                         must cut every Map3 row; the two feedback keys
                         cannot be on together
    """

    def initialize(self, probe):
        """Read the fixed inputs and configure cosmolike, without a cosmology.

        Cobaya calls initialize once, when it builds the likelihood. In
        order: read the .dataset file and the measurement, covariance, mask
        and compression inputs; configure cosmolike and load the joint data;
        build the z and k grids of the tables sent to cosmolike at every
        point; read the source n(z); prepare the Map3 aperture engine and
        the xi quadrature. Cosmolike keeps its configuration in C global
        variables shared by every likelihood object of the Python process,
        so each option is set here explicitly, even at its default value.

        Arguments:
          probe = "xi_map3", the joint xi_+/xi_-/Map3 measurement

        Raises:
          ValueError when the mask or the compression labels do not match
          the selected data rows; file readers raise on malformed inputs.

        Side effects: changes cosmolike's global state (configuration, data,
        covariance, mask, compression); sets the grid, row-index and
        quadrature attributes used by every later method.
        """
        # ------------------------------------------------------------------
        # Fixed inputs. ini = the .dataset file parsed by getdist's IniFile;
        # relativeFileName returns the file named by a key, with a relative
        # name resolved against the directory of the .dataset file. A null
        # yaml path selects the project's own data/ directory.
        # ------------------------------------------------------------------
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

        # The scored vector defaults to the measured rows themselves; an
        # optional data_file key replaces it (for example with a synthetic
        # vector) while keeping the measured covariance and row layout.
        data = self.rows.values
        if ini.string("data_file", ""):
            data = np.loadtxt(fname=ini.relativeFileName("data_file"))

        # Without a mask_file key every row enters the chi2; with one, the
        # rows flagged 1 are kept and the rows flagged 0 are cut.
        indices = np.arange(self.rows.values.size, dtype=np.int64)
        if ini.string("mask_file", ""):
            mask = np.loadtxt(fname=ini.relativeFileName("mask_file"))
            if mask.shape != data.shape or not np.all((mask == 0) | (mask == 1)):
                raise ValueError("The mask must have one 0/1 value per data row.")
            indices = np.flatnonzero(mask)

        # Optional fixed compression (the matrix T of the module docstring),
        # trained and saved by likelihood/compression.py. Only the row
        # labels are checked here, so a transform trained on different rows
        # is refused before any native state changes; the numerical matrix
        # checks and the application to data and covariance happen inside
        # ci.init_joint_data. An empty matrix means no compression.
        compression = np.empty((0, 0), dtype=np.float64)
        if ini.string("compression_file", ""):
            # The with-statement closes the npz archive when the block ends,
            # even on an exception.
            with np.load(
                file=ini.relativeFileName("compression_file"),
                allow_pickle=False,
            ) as saved:
                if not np.array_equal(
                    a1=saved["input_labels"],
                    a2=self.rows.labels[indices],
                ):
                    raise ValueError(
                        "Compression input labels differ from selected rows.")
                compression = saved["matrix"]

        # ------------------------------------------------------------------
        # Feedback options, checked before any native state changes. The
        # two keys describe different physics: external_baryon_response
        # applies a paired power/bispectrum response (R_P and R_B from one
        # model), external_baryon_suppression a power-only response R_P.
        # Applying both would multiply the power by two feedback models.
        # A power-only model says nothing about the bispectrum, and taking
        # R_B = 1 would silently drop feedback from the Map3 rows while
        # keeping it in the xi rows. With external_baryon_suppression no
        # Map3 row may therefore enter the chi2: the dataset mask must cut
        # all of them, and internal_get_datavector never computes them.
        # ------------------------------------------------------------------
        if self.external_baryon_response and self.external_baryon_suppression:
            raise ValueError(
                "external_baryon_response and external_baryon_suppression are "
                "both True. Choose one feedback model: the paired BACCO "
                "response (R_P and R_B) or the power-only bfmt response (R_P)."
            )
        if self.external_baryon_suppression:
            kept_map3 = int(np.count_nonzero(self.rows.kind[indices] == "map3"))
            if kept_map3 > 0:
                raise ValueError(
                    "external_baryon_suppression supplies only the power "
                    "response R_P; it has no bispectrum response R_B for the "
                    f"Map3 rows, and {kept_map3} Map3 rows of {self.data_file} "
                    "enter the chi2. Cut every Map3 row with the dataset "
                    "mask_file, or use external_baryon_response with the "
                    "bacco_bispectrum theory block."
                )

        # ------------------------------------------------------------------
        # Native configuration. Some imported libraries reset the
        # process-wide OpenMP thread count without saying so, hence the
        # explicit ci.set_omp_threads here and before every heavy native
        # call. initial_setup resets cosmolike's global configuration to its
        # defaults; every option below is then set from its yaml key.
        # ------------------------------------------------------------------
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
        # Cosmolike sizes its tomographic bin-pair tables assuming at least
        # one lens bin exists. This likelihood evaluates no lens observable,
        # so set_source_related later fills the mandatory slot with the
        # first source n(z); the shear statistics never read it.
        ci.init_lens_sample_size(Ntomo=1)
        # Intrinsic alignments: ia_model = 1 selects the TATT expansion
        # (tidal alignment and tidal torquing), ia_redshift_evolution = 3
        # the power-law A(z) = a * ((1 + z)/(1 + zpivot))^eta, and
        # ia_code = 0 the C FAST-PT implementation.
        ci.init_IA(
            ia_model=1,
            ia_redshift_evolution=3,
            ia_code=0,
        )
        ci.reset_bary_struct()
        # The native side owns the data, covariance, mask and compression
        # from here on; it validates the matrices and computes the joint
        # chi2. mock_count and parameter_count select the optional
        # finite-simulation (simulation-estimated covariance) likelihood;
        # both 0 keeps the plain Gaussian.
        ci.init_joint_data(
            data=data,
            covariance=covariance,
            indices=np.asarray(indices, dtype=np.int32),
            compression=compression,
            mock_count=ini.int("mock_count", 0),
            parameter_count=ini.int("parameter_count", 0),
        )
        # self.accuracy = the resolved three-point accuracy preset; the
        # ci.accuracy_* helpers below turn it into node counts and
        # quadrature settings for each numerical stage.
        self.accuracy = ci.init_3pcf_accuracy(value=self.threepcf_accuracy)

        # ------------------------------------------------------------------
        # Interpolation grids. z_interp_2D and log10k_interp_2D are the
        # nodes of the P(k, z) tables sent to cosmolike at every point;
        # source_z is the common grid of the reconstructed source n(z);
        # los_z the line-of-sight grid of the Map3 projection.
        # ------------------------------------------------------------------
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
        # ------------------------------------------------------------------
        # Measured source distributions. The call returns a four-value
        # tuple, in this order: the measured redshift grid, the n(z) curve
        # of every bin [nbin, nz], and the lower and upper edges of the
        # measured redshift cells. set_source_related reconstructs shifted
        # n(z) from these at every sampled point.
        # ------------------------------------------------------------------
        (self.source_native_z, self.source_native,
         self.source_native_low, self.source_native_high) = read_source_distributions(
            path=ini.relativeFileName("nz_source_file"),
            nbin=self.source_ntomo,
            with_edges=True,
        )

        # ------------------------------------------------------------------
        # The Map3 aperture engine: the fixed transform plan that turns the
        # physics model of set_source_related into the Map3 rows. It
        # integrates the shear three-point function over the same
        # side-angle-side (SAS) bins as the measurement: ntheta log bins
        # between theta_min and theta_max, of which the scale cut keeps
        # those whose lower edge is at least theta_lower_cut_arcmin. The
        # kept lower edges (radians, pi / (180 * 60) per arcmin) and the
        # common log width dlnt are stored for notebooks that rebuild the
        # plan on a subset of rows. The aperture filter radii come from the
        # measurement file (the Map3 rows' angles).
        # ------------------------------------------------------------------
        self.map_indices = np.flatnonzero(self.rows.kind == "map3")
        self.xi_indices = np.flatnonzero(self.rows.kind != "map3")
        edges = np.exp(np.linspace(
            start=np.log(self.theta_min_arcmin),
            stop=np.log(self.theta_max_arcmin),
            num=self.ntheta + 1,
        ))
        kept = edges[:-1][edges[:-1] >= self.theta_lower_cut_arcmin]
        if 0 == kept.size:
            raise ValueError(
                f"No side bin has a lower edge >= theta_lower_cut_arcmin = "
                f"{self.theta_lower_cut_arcmin}; the bins end at {edges[-1]}.")
        self.theta_lower = kept * np.pi / (180.0 * 60.0)
        self.dlnt = float(np.log(edges[1] / edges[0]))
        dphi = np.pi / self.phi_bins
        map_rows = self.rows.select(indices=self.map_indices)
        self.map_engine = ci.production.TransformPlan.for_aperture(
            bins=np.ascontiguousarray(
                a=map_rows.bins,
                dtype=np.int32,
            ),
            filters=np.ascontiguousarray(a=map_rows.angles),
            theta_lower=self.theta_lower,
            phi=(np.arange(self.phi_bins) + 0.5) * dphi,
            dlnt=self.dlnt,
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

        # ------------------------------------------------------------------
        # The xi quadrature: fixed multipole nodes and weights that turn the
        # shear power spectra C_ell into the binned xi_+ and xi_- rows, plus
        # the angular nodes of the bin average. xi_sign is +1 for xi_+ rows
        # and -1 for xi_- rows (they add and subtract the B-mode spectrum).
        # ------------------------------------------------------------------
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
        """Return the theory products Cobaya must compute at every point.

        Cobaya reads this dictionary once and routes each entry to the
        theory block that can supply it (CAMB or a custom block). Each key
        names a product; the value fixes the redshift grid and options:
        Hubble, comoving_radial_distance, sigma8_z and the three Omega_X(z)
        density fractions on z_interp_2D; Pk_interpolator, the linear and
        nonlinear P(k, z) for both the total matter and the cold
        (baryon + CDM, "delta_nonu") fields; matter_bispectrum, the custom
        block's bispectrum table (set_cosmo_related documents its axes);
        when external_baryon_response is on, the paired baryonic
        responses R_P and R_B; and, when external_baryon_suppression is
        on, the bfmt block's baryon_suppression R_P on fixed z and k
        (1/Mpc) nodes.

        Returns:
          dictionary {product name: options or None}, in Cobaya's
          requirements format.
        """
        requirements = {
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
        if self.external_baryon_response:
            requirements["baryon_response"] = None

        # The bfmt block evaluates R_P at fixed wavenumbers in 1/Mpc, while
        # this likelihood's tables use h/Mpc and h changes from point to
        # point. The request therefore reuses the numbers of the table grid
        # read as 1/Mpc, a fixed grid over the same eight decades;
        # baryon_suppression_table moves R_P onto the h/Mpc nodes.
        if self.external_baryon_suppression:
            requirements["baryon_suppression"] = {
                "z": self.z_interp_2D,
                "k": 10.0**self.log10k_interp_2D,
            }
        return requirements

    def set_cosmo_related(self):
        """Pass the current provider arrays to the native cosmology setters.

        Hot path: Cobaya calls this at every sampled point, through
        internal_get_datavector. Cobaya uses k in 1/Mpc and P in Mpc^3.
        CosmoLike uses h/Mpc and (Mpc/h)^3, hence k_physical=h*k and
        P_h=h^3*P_physical. The requested CAMB transfer limit stays at
        kmax_boltzmann; its interpolator extends the supplied spectrum to
        the much larger projection-grid endpoint.

        Side effects: sets cosmolike's global cosmology (distances, growth
        and the ln P tables) and the attributes h, p_linear, p_nonlinear,
        chi, dz_dchi, sigma8, growth, omega_m_z, baseline and feedback that
        set_source_related reads at the same point. With
        external_baryon_suppression the native nonlinear power is
        P_DMO * R_P, while the attribute p_nonlinear stays P_DMO.

        Raises:
          ValueError when the bispectrum product lacks its native table,
          the baryon response does not cover the redshift range, or the
          bfmt R_P fails the checks of baryon_suppression_table.
        """
        ci.set_omp_threads(n=self.omp_threads)

        # ------------------------------------------------------------------
        # Provider reads: expansion and the three P(k, z) interpolators.
        # h converts the units below; k is the table grid in 1/Mpc. The
        # extrap_kmin/extrap_kmax bounds tell the interpolator how far it
        # may extrapolate beyond CAMB's computed range.
        # ------------------------------------------------------------------
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
        # cold = the linear power of the cold field (baryons + CDM, no
        # massive neutrinos); cosmolike's massive-neutrino runs need it.
        cold = self.provider.get_Pk_interpolator(
            var_pair=("delta_nonu", "delta_nonu"),
            nonlinear=False,
            extrap_kmin=float(k[0]),
            extrap_kmax=max(self.kmax_boltzmann, 1.01 * float(k[-1])),
        )

        # ------------------------------------------------------------------
        # Power tables [z, k] in (Mpc/h)^3 (hence the h^3), distances in
        # Mpc/h (hence the h), and the growth factor normalized to one
        # today, from sigma8(z)/sigma8(0).
        # ------------------------------------------------------------------
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
        # dz/dchi = H(z)/c in h/Mpc (chi is in Mpc/h); 299792.458 km/s is
        # the speed of light, matching the km/s/Mpc unit of hubble above.
        self.dz_dchi = hubble / (299792.458 * self.h)
        sigma = self.provider.get_sigma8_z(z=self.z_interp_2D)
        self.sigma8 = float(sigma[0])
        self.growth = sigma / self.sigma8

        # ------------------------------------------------------------------
        # Density fractions Omega_X(z) from the provider; the z-array form
        # is what this fork's provider exposes, and the [0] entries below
        # give today's values. omega_m_z enters the Map3 lensing kernels.
        # ------------------------------------------------------------------
        omega_b = self.provider.get_Omega_b(z=self.z_interp_2D)
        omega_c = self.provider.get_Omega_cdm(z=self.z_interp_2D)
        omega_nu = self.provider.get_Omega_nu_massive(z=self.z_interp_2D)
        self.omega_m_z = omega_b + omega_c + omega_nu

        # ------------------------------------------------------------------
        # The matter bispectrum B(k1, k2, k3; z) from the custom theory
        # block, as a table over four axes: redshift z, ln of the longest
        # side k, ln of the short/longest and middle/longest side ratios,
        # and the interpolation variable v of that parametrization; values
        # are in (Mpc/h)^6. The native constructor of the physics model
        # validates the axes and their sizes; here only the presence of the
        # native table is checked.
        # ------------------------------------------------------------------
        self.baseline = self.provider.get_result("matter_bispectrum")
        if "native" not in self.baseline:
            raise ValueError("matter_bispectrum must supply its native B table.")

        # ------------------------------------------------------------------
        # Optional baryonic feedback: the paired responses
        # R_P = P_feedback/P_gravity-only and R_B (the same ratio for the
        # bispectrum) from the baryon_response theory block. The table must
        # cover the full redshift range; its node boost follows the
        # three-point accuracy preset.
        # ------------------------------------------------------------------
        self.feedback = {}
        if self.external_baryon_response:
            response = self.provider.get_result("baryon_response")
            self.feedback = dict(response["native"])
            if self.feedback["z"][-1] < self.z_interp_2D[-1]:
                raise ValueError("The baryon-response zmax must cover the likelihood.")
            self.feedback["boost"] = ci.accuracy_count(
                accuracy=self.accuracy,
                baseline=self.feedback["boost"],
                stage="feedback_table_boost",
            )

        # ------------------------------------------------------------------
        # Optional power-only feedback (external_baryon_suppression): the
        # xi rows receive P_DMO * R_P, with R_P from the bfmt block. The
        # attribute p_nonlinear keeps the gravity-only array; without the
        # option the native power is that same array object, unchanged.
        # ------------------------------------------------------------------
        p_nonlinear_xi = self.p_nonlinear
        if self.external_baryon_suppression:
            p_nonlinear_xi = self.p_nonlinear * self.baryon_suppression_table()

        # ------------------------------------------------------------------
        # Native cosmology. The tables here are [z, k]; the C side stores
        # [k, z], and flatten(order="F") (Fortran order: the first index,
        # z, runs fastest) produces exactly that memory order. G is the
        # growth normalization cosmolike expects, growth times (1 + z).
        # ------------------------------------------------------------------
        ci.set_cosmology(
            omegam=float(self.omega_m_z[0]),
            H0=100.0 * self.h,
            log10k_2D=self.log10k_interp_2D,
            z_2D=self.z_interp_2D,
            lnP_linear=np.log(self.p_linear).flatten(order="F"),
            lnP_nonlinear=np.log(p_nonlinear_xi).flatten(order="F"),
            G=self.growth * (1.0 + self.z_interp_2D),
            z_1D=self.z_interp_2D,
            chi=self.chi,
            omegab=float(omega_b[0]),
            z_G=self.z_interp_2D,
            omegan2=float(omega_nu[0]) * self.h**2,
            lnP_linear_cb=np.log(p_linear_cb).flatten(order="F"),
        )

    def baryon_suppression_table(self):
        """Return the bfmt power response R_P on the table grid [z, k in h/Mpc].

        Hot path: runs at every sampled point when external_baryon_suppression
        is on. The bfmt block returns a dictionary {z: R_P array}, with
        R_P = P_feedback/P_DMO evaluated at the fixed wavenumbers requested
        in get_requirements: the numbers 10^log10k_interp_2D read as k in
        1/Mpc. The table nodes are k_h in h/Mpc, i.e. the physical
        wavenumbers h * k_h, so R_P must be read at log10(k_h) + log10(h).

        Both grids are the same uniform log10 k grid with spacing d, so the
        read is a constant shift of log10(h)/d grid cells: every table node
        sits between two neighbouring request nodes at the same fraction w,
        and linear interpolation in log10 k of ln R_P needs only index
        arithmetic, with one weight w for the whole table. Table
        nodes below the first request node (h * k_h < 10^log10k_min per
        Mpc, the largest scales, where every model has R_P = 1) hold the
        response of that first node.

        Accuracy: for the seven example models this read differs from
        bfmt evaluated directly at h * k by at most about 2e-3 between
        0.05 and 8 h/Mpc. A 4 to 16 times denser request grid leaves that
        difference unchanged, so it is not this read: bfmt evaluates each
        model on its own coarse nz x nk grid placed over the requested
        range and splines the result, and its options nz and nk set that
        accuracy (tests/test_bfmt_suppression.py).

        Returns:
          float64 array [power_nz, power_nk], R_P on z_interp_2D and the
          h/Mpc nodes 10^log10k_interp_2D.

        Raises:
          ValueError when the bfmt result lacks a requested redshift,
          returns an array of the wrong length or a non-positive or
          non-finite ratio, or when h > 1 (the shifted nodes would then
          leave the requested range at its high-k end).
        """
        # ------------------------------------------------------------------
        # Validation of the provider result and of h. The dictionary keys
        # are the requested z_interp_2D floats themselves, so an exact
        # lookup is the right test.
        # ------------------------------------------------------------------
        rows = self.provider.get_result("baryon_suppression")
        nz = self.z_interp_2D.size
        nk = self.log10k_interp_2D.size
        if self.h > 1.0:
            raise ValueError(
                f"external_baryon_suppression requires h <= 1; observed "
                f"h = {self.h}. The shifted table nodes h * k would exceed the "
                "largest wavenumber requested from bfmt."
            )
        log_response = np.empty((nz, nk))
        for index in range(nz):
            redshift = self.z_interp_2D[index]
            if redshift not in rows:
                raise ValueError(
                    f"The bfmt baryon_suppression result lacks z = {redshift}; "
                    "it must return every requested redshift."
                )
            values = np.asarray(rows[redshift], dtype=np.float64)
            if values.shape != (nk,):
                raise ValueError(
                    f"The bfmt R_P at z = {redshift} must have {nk} values; "
                    f"observed shape {values.shape}."
                )
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(
                    f"The bfmt R_P at z = {redshift} must be positive and "
                    f"finite; observed minimum {np.min(values)}."
                )
            log_response[index] = np.log(values)

        # ------------------------------------------------------------------
        # Shift by log10(h). cells = log10(h)/d is negative for h < 1;
        # table node j lies between request nodes lower[j] and
        # lower[j] + 1, at fraction weight. np.clip sends indices below 0
        # to 0, which turns the interpolation there into a hold of the
        # first request node (both neighbours become node 0).
        # ------------------------------------------------------------------
        spacing = self.log10k_interp_2D[1] - self.log10k_interp_2D[0]
        cells = np.log10(self.h) / spacing
        offset = int(np.floor(cells))
        weight = cells - offset
        nodes = np.arange(nk) + offset
        lower = np.clip(nodes, 0, nk - 1)
        upper = np.clip(nodes + 1, 0, nk - 1)
        shifted = ((1.0 - weight) * log_response[:, lower]
                   + weight * log_response[:, upper])
        return np.exp(shifted)

    def set_source_related(self, **params):
        """Send the shifted source n(z) and nuisance parameters to cosmolike.

        Hot path: Cobaya calls this at every sampled point, after
        set_cosmo_related. Both probes use the same reconstructed, shifted
        sources and the same CosmoLike intrinsic-alignment amplitudes; this
        method also builds the Python-side Map3 physics model from the
        arrays set_cosmo_related stored.

        Arguments:
          **params = the sampled nuisance values by name:
            DES_DZ_S1..4 (photo-z shifts), DES_M1..4 (shear calibrations),
            DES_A1_1, DES_A1_2, DES_A2_1, DES_A2_2, DES_BTA_1 (intrinsic
            alignment). A name the run does not sample reads as 0 (no
            shift, no calibration, no alignment).

        Side effects: sets cosmolike's source and lens samples and nuisance
        parameters; sets the attributes calibration, sources, ia and model
        (and, with external feedback, the native nonlinear power) that
        internal_get_datavector reads at the same point.
        """
        # ------------------------------------------------------------------
        # Per-bin photo-z shifts and multiplicative shear calibrations from
        # the sampled parameters (bin i reads DES_DZ_S<i> and DES_M<i>).
        # ------------------------------------------------------------------
        shifts = np.zeros(self.source_ntomo)
        self.calibration = np.zeros(self.source_ntomo)
        for index in range(self.source_ntomo):
            shifts[index] = params.get(survey + "_DZ_S" + str(index + 1), 0.0)
            self.calibration[index] = params.get(survey + "_M" + str(index + 1), 0.0)

        # ------------------------------------------------------------------
        # Reconstruct the source n(z) on the common source_z grid, with the
        # shifts applied, from the measured curves initialize read. The
        # second returned value (diagnostics) is not needed here; "_" is
        # the conventional name for a discarded value.
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Native samples. The table is one column of redshifts followed by
        # one n(z) column per bin. The one-bin lens sample only fills the
        # mandatory slot of cosmolike's pair bookkeeping (see initialize);
        # its first two columns are z and the first source curve.
        # ------------------------------------------------------------------
        source_table = np.column_stack((self.source_z, self.sources.T))
        ci.set_source_sample(nofz=source_table)
        ci.set_lens_sample(nofz=source_table[:, :2].copy())
        ci.init_ntomo_powerspectra()

        # These sources already contain the shift, and the multiplicative
        # calibration (1 + m_i) is applied in Python after projection, once
        # per shear leg of each statistic (two factors for xi, three for
        # Map3). The native per-bin photo-z and calibration are therefore
        # set to zero so neither correction is applied twice.
        ci.set_nuisance_shear_photoz(bias=np.zeros(self.source_ntomo))
        ci.set_nuisance_shear_calib(M=np.zeros(self.source_ntomo))

        # ------------------------------------------------------------------
        # Intrinsic-alignment amplitudes. init_IA selected the power-law
        # redshift evolution, under which cosmolike reads the per-bin
        # arrays as parameter slots, not as bins: slot 0 holds the
        # amplitude and slot 1 the redshift exponent of
        # A(z) = a * ((1 + z)/(1 + zpivot))^eta. The remaining slots must
        # be zero. DES_A1_1/DES_A1_2 (a1, eta1) are the tidal-alignment
        # term, DES_A2_1/DES_A2_2 (a2, eta2) the tidal-torquing term, and
        # DES_BTA_1 (bta, amplitude only) the density weighting of the
        # TATT expansion; a1 ... bta below are those values.
        # ------------------------------------------------------------------
        a1 = np.zeros(self.source_ntomo)
        a2 = np.zeros(self.source_ntomo)
        bta = np.zeros(self.source_ntomo)
        a1[:2] = [params.get(survey + "_A1_1", 0.0), params.get(survey + "_A1_2", 0.0)]
        a2[:2] = [params.get(survey + "_A2_1", 0.0), params.get(survey + "_A2_2", 0.0)]
        bta[0] = params.get(survey + "_BTA_1", 0.0)
        ci.set_nuisance_ia(
            A1=a1,
            A2=a2,
            B_TA=bta,
        )

        # ------------------------------------------------------------------
        # The IA record of the Map3 model: the same amplitudes plus the
        # fixed CosmoLike normalization c1rho = 0.01389 and pivot
        # zpivot = 0.62, the IA power prescription (power_mode), a zero
        # velocity-shear amplitude (ct), and the feedback switch. fb = 0
        # leaves the alignment power unsuppressed by feedback.
        # ------------------------------------------------------------------
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
            "feedback_amplitude": float(self.external_baryon_response),
        }

        # ------------------------------------------------------------------
        # With power-only feedback no Map3 row is computed (initialize
        # refused every kept Map3 row), so the Map3 physics model, which
        # would otherwise carry a gravity-only bispectrum, is not built.
        # ------------------------------------------------------------------
        if self.external_baryon_suppression:
            self.model = None
            return

        # ------------------------------------------------------------------
        # The Map3 physics model. normalization rescales the linear power
        # entering the model's IA terms when the bispectrum producer's
        # metadata declares a linear_power_normalization; the chained .get
        # calls read metadata -> linear_power_normalization and fall back
        # to the default 1.0 (no rescaling) at either missing key. It never
        # rescales the ordinary two-point CAMB power. The accuracy preset
        # resolves the kernel, table and fitted-n(z) node counts.
        # ------------------------------------------------------------------
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
            feedback=self.feedback,
        )

        # ------------------------------------------------------------------
        # Optional baryonic feedback of the two-point probe. The Map3 model
        # keeps the gravity-only (DMO, dark-matter-only: power without
        # baryonic feedback) power for its quadratic IA terms and applies
        # R_B to the matter/NLA bispectrum; the ordinary two-point core
        # instead receives P_DMO * R_P. Query the model's own response
        # reader here, so both probes use its uniform cubic preparation and
        # linear reads. Linear TATT power stays DMO. Each call starts from
        # the provider's fresh DMO array.
        # ------------------------------------------------------------------
        if self.external_baryon_response:
            wave, redshift = np.meshgrid(
                10.0**self.log10k_interp_2D,
                self.z_interp_2D,
            )
            power_response = self.model.power_response(
                k=wave.ravel(),
                z=redshift.ravel(),
                amplitude=1.0,
            ).reshape(self.p_nonlinear.shape)
            ci.set_non_linear_power_spectrum(
                log10k=self.log10k_interp_2D,
                z=self.z_interp_2D,
                lnP=np.log(self.p_nonlinear * power_response).flatten(order="F"),
            )

    def compute_logp(self, datavector):
        """Evaluate the native joint log-density of one theory vector.

        Hot path: runs at every sampled point. ci.compute_joint_loglike
        returns the normalized Gaussian log-density (or, when the dataset
        selects the finite-simulation likelihood, that density), which
        differs from -chi2/2 by a fixed constant. Retaining the constant
        permits evidence comparisons between the two likelihood choices.
        Cobaya adds the parameter priors itself.

        Arguments:
          datavector = theory vector [ndata] in the measured row order.

        Returns:
          float, the log-density of the data given this theory vector.
        """
        return ci.compute_joint_loglike(prediction=datavector)

    def logp(self, **params):
        """Return the log-likelihood of one sampled point (Cobaya's entry).

        Hot path: Cobaya calls this at every sampled point.

        Arguments:
          **params = the sampled nuisance values by name (the list is in
            the set_source_related docstring).

        Returns:
          float, the joint log-likelihood.

        Side effects: stores the scored theory vector in
        self.last_datavector, so notebooks can build Fisher derivatives
        after a model.loglike call without repeating the entire native
        projection or bypassing the Cobaya theory route.
        """
        self.last_datavector = self.get_datavector(**params)
        return self.compute_logp(datavector=self.last_datavector)

    def get_datavector(self, **params):
        """Return the theory data vector at one sampled point.

        Hot path: runs at every sampled point. The float64 conversion
        guards the native likelihood call, which requires a contiguous
        double array.

        Arguments:
          **params = the sampled nuisance values by name.

        Returns:
          float64 array [ndata], every xi_+/xi_-/Map3 row in the input
          measurement order.
        """
        return np.asarray(self.internal_get_datavector(**params))

    def internal_get_datavector(self, **params):
        """Set the physical inputs and project both probes into one vector.

        Hot path: runs at every sampled point. First the two setters send
        the current cosmology and nuisance parameters to cosmolike, then
        the xi rows come from the native shear spectra and the fixed
        quadrature, and the Map3 rows from the aperture engine; each block
        lands at its own row indices of the measured order.

        Arguments:
          **params = the sampled nuisance values by name.

        Returns:
          float array [ndata] in the measured row order; with
          external_baryon_suppression the Map3 entries are NaN (refused,
          never computed).
        """
        # ------------------------------------------------------------------
        # Current point: cosmology, sources and nuisance parameters into
        # cosmolike and the Map3 model.
        # ------------------------------------------------------------------
        self.set_cosmo_related()
        self.set_source_related(**params)
        ci.set_omp_threads(n=self.omp_threads)
        result = np.empty(self.rows.values.size)

        # ------------------------------------------------------------------
        # The xi block: native E- and B-mode shear spectra per bin pair,
        # then the fixed multipole quadrature and bin average of initialize
        # turn them into the binned xi_+ (E + B) and xi_- (E - B) rows,
        # with the (1 + m_i)(1 + m_j) calibration applied per row.
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # The Map3 block: the aperture engine projects the physics model
        # and the IA record into one value per bin triple and filter scale,
        # with three calibration factors per row. Under power-only feedback
        # the Map3 rows are refused: they hold NaN (not a number), which
        # marks them as never computed; initialize guaranteed that the
        # mask keeps none of them, and the native chi2 reads kept rows only.
        # ------------------------------------------------------------------
        if self.external_baryon_suppression:
            result[self.map_indices] = np.nan
            return result
        result[self.map_indices] = self.map_engine.aperture_vector(
            model=self.model,
            ia=self.ia,
            calibration=self.calibration,
        )
        return result
