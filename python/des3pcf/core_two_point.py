"""Use the existing CosmoLike CFASTPT two-point model beside the new port.

The ordinary core already calculates TATT E/B power spectra. Reusing that
calculation avoids a second implementation of its loop integrals while
keeping all existing C source files unchanged. This adapter installs the
same physical power, distance, growth and source arrays used for the
three-point calculation, then translates the IA amplitude convention.

The existing core has one mutable physical state per process. Every
spectrum request therefore resets and reinstalls this adapter's declared
calculation. A preceding notebook cell may have changed selectors, source
counts or even only part of another model. Construction stores settings;
the restoration occurs when spectra is called. The new 3pcf model remains
independent. Parallelism belongs inside C, not in simultaneous Python calls
that change the ordinary core between restoration and evaluation.
"""

import numpy as np

from .inputs import validate_tables
from .mean import core_power_rows
from .ownership import PreparedGeometry


def core_ia_parameters(ia, nbin):
    """Translate fastnc IA amplitudes to the ordinary core's normalization.

    Both models use C1=-A1*c1rho*Omega_m/D times a redshift power law.
    The ordinary core fixes c1rho=0.01389 and 1+zp=1.62. Matching the
    physical C1 therefore multiplies the incoming amplitude by
    (c1rho/0.01389)*(1.62/(1+zp))^eta. C2 has the same normalization
    ratio and its own exponent. Its factor 5 is present in both models.

    Arguments:
      ia = fastnc parameter dictionary with a1/eta1/a2/eta2/bta/zpivot/c1rho.
      nbin = source-bin count, at least two for the amplitude/slope slots.
    Returns:
      owned A1, A2 and B_TA arrays accepted by set_nuisance_ia.
    Raises:
      ValueError for unsupported velocity-shear, baryon or normalization choices.
    """
    if nbin < 2:
        raise ValueError(
            "The ordinary core's redshift-law adapter requires at least two bins."
        )
    if ia.get("ct", 0.0) != 0.0:
        raise ValueError(
            "Ordinary two-point TATT has no velocity-shear term; supply its spectra."
        )
    if ia.get("fb", 0.0) != 0.0:
        raise ValueError(
            "Bispectrum TNG feedback does not specify two-point feedback power."
        )
    pivot = float(ia.get("zpivot", 0.62))
    normalization = float(ia.get("c1rho", 0.0134))
    if not np.isfinite(pivot) or pivot <= -1.0 or normalization <= 0.0:
        raise ValueError("IA requires finite zpivot>-1 and positive c1rho.")
    first = np.zeros(nbin, dtype=np.float64)
    second = np.zeros_like(first)
    density = np.zeros_like(first)
    first[1] = float(ia.get("eta1", 0.0))
    second[1] = float(ia.get("eta2", 0.0))
    ratio = normalization / 0.01389
    first[0] = ia.get("a1", 0.0) * ratio * (1.62 / (1.0 + pivot))**first[1]
    second[0] = ia.get("a2", 0.0) * ratio * (1.62 / (1.0 + pivot))**second[1]
    density[0] = float(ia.get("bta", 0.0))
    for values in (first, second, density):
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Translated IA parameters are nonfinite; check amplitudes/slopes."
            )
    return first, second, density


class CoreTwoPoint(PreparedGeometry):
    """Prepare and query the existing compiled core for a joint TATT forecast.

    Arguments:
      core = imported ordinary project module, for example DES Y3's extension.
      nbin = source-bin count, at least two.
      threads = positive worker count for the existing core.
      interpolation_type = ordinary-core n(z) interpolation: 0 cubic, 1 linear.
      accuracy_boost = ordinary-core common numerical refinement, positive.
      integration_accuracy = fixed-quadrature setting, integer zero through seven.
      fpt_internal_boost = CFASTPT internal/output grid ratio, positive.
      ell_internal = coarse C_ell table count, or zero for exact table nodes.
    Side effects:
      Construction changes no core state. Each spectra call resets and
      installs the declared model, leaving it as that module's current state.
      Settings are immutable: construct a new adapter to change them in the
      same kernel. No files or C source are changed.
    """

    def __init__(
        self, core, nbin, threads=1, interpolation_type=0,
        accuracy_boost=1.0, integration_accuracy=0,
        fpt_internal_boost=0.5, ell_internal=192,
    ):
        """Store a complete ordinary-core shear calculation without running it.

        Arguments:
          All arguments have the meanings given in the class contract.
        Returns:
          nothing; no core state or physical table is changed here.
        Raises:
          ValueError for an invalid count or interpolation choice.
        """
        if (int(nbin) != nbin or nbin < 2 or int(threads) != threads or threads < 1
                or interpolation_type not in (0, 1)):
            raise ValueError(
                "Core two-point adapter needs nbin>=2, threads>=1 and interpolation 0/1."
            )
        if (not np.isfinite(accuracy_boost) or accuracy_boost <= 0.0
                or not np.isfinite(fpt_internal_boost)
                or fpt_internal_boost <= 0.0):
            raise ValueError("Core accuracy and CFASTPT grid ratios must be positive.")
        if (isinstance(integration_accuracy, bool)
                or int(integration_accuracy) != integration_accuracy
                or not 0 <= integration_accuracy <= 7):
            # The shear quadrature has eight distinct rules, from 96 to
            # 8192 nodes. Higher raw-core settings keep the final rule while
            # still enlarging some redshift arrays. This adapter supports
            # only the named 0..7 ladder, with no silently saturated setting.
            raise ValueError("Core integration_accuracy must be an integer in 0..7.")
        # The existing core resets its dense ell table to 512 nodes before
        # applying the common boost. Check the coarse-table setter's range
        # here rather than allowing its fatal C diagnostic to stop Python.
        dense_ell = int(np.ceil(512 * accuracy_boost))
        if (isinstance(ell_internal, bool) or int(ell_internal) != ell_internal
                or (ell_internal != 0 and not 4 <= ell_internal <= dense_ell)):
            raise ValueError("ell_internal must be zero or 4..ceil(512*accuracy_boost).")
        self.core = core
        self.nbin = int(nbin)
        self.threads = int(threads)
        self.interpolation_type = int(interpolation_type)
        self.accuracy_boost = float(accuracy_boost)
        self.integration_accuracy = int(integration_accuracy)
        self.fpt_internal_boost = float(fpt_internal_boost)
        self.ell_internal = int(ell_internal)
        self._seal_geometry()

    def restore_core(self):
        """Reset unknown notebook state and install the owned shear selectors.

        The existing binding cannot read back all numerical settings or
        the hidden baselines captured by init_accuracy_boost. Photo-z
        selectors can also reload old source arrays before new counts are
        installed. A full reset removes that uncertainty even after a
        partially configured notebook model; no external ownership token
        or manual reclaim call is required.

        init_ntomo_powerspectra already invalidates Ntable on every spectra
        request. This reset does not add another numerical table evaluation
        before the requested spectra, but it does reconstruct input
        splines. No performance claim is made for that restoration cost.

        Returns:
          nothing; the supplied ordinary module now has these selectors.
        """
        core = self.core
        core.initial_setup()
        core.set_omp_threads(n=self.threads)
        core.init_accuracy_boost(
            accuracy_boost=self.accuracy_boost,
            integration_accuracy=self.integration_accuracy,
        )
        # Apply dedicated overrides after the common boost, whose core
        # implementation also scales the coarse ell and CFASTPT controls.
        core.init_fpt_internal_boost(internal_boost=self.fpt_internal_boost)
        core.init_ntable_ell_internal(nell_internal=self.ell_internal)
        core.init_probes(possible_probes="xi")
        core.init_cosmo_runmode(is_linear=False)
        core.init_include_halo_IA(include_halo_IA=0)
        core.init_photoz_histogram_akima(enabled=0, preparation_factor=3)
        core.init_photoz_conventions(
            interpolation_type=self.interpolation_type, zmid_convention=1,
        )
        core.init_source_sample_size(Ntomo=self.nbin)
        # One unused lens bin permits the shared pair bookkeeping to be
        # initialized. Only source-source spectra are ever requested here.
        core.init_lens_sample_size(Ntomo=1)
        core.init_IA(ia_model=1, ia_redshift_evolution=3, ia_code=0)
        core.reset_bary_struct()

    def spectra(self, parameters, tables, metadata, ia, ell, bins):
        """Install shared physical arrays and return row-ordered TATT E/B powers.

        Full Plinear(z,k), not Plinear(0,k)*D(z)^2, is required for the
        ordinary core's loops. Sources already include photo-z shifts;
        the ordinary core's shift/calibration parameters are set to zero
        to avoid applying those nuisance effects twice. XiEngine applies
        the final multiplicative calibration after its bin integration.

        Arguments:
          parameters = cosmological h and omega_b, matching the supplied tables.
          tables = physics schema plus p_linear[nz,nk] and optional p_linear_cb.
              Optional source_z[nsource] is independent of the power z[nz].
              Its uniform density samples already contain any photo-z shift.
          metadata = physical preparation record, including omega_nu_h2.
          ia = fastnc IA parameters, translated by core_ia_parameters.
          ell = requested 1D multipoles.
          bins = zero-based [nrow,2] source pairs.
        Returns:
          E/B signal spectra [nrow,nell], before shear calibration.
        Raises:
          ValueError for missing physical tables, neutrino information or IA support.
        Side effects:
          Resets and reinstalls the supplied ordinary module's in-memory
          selectors and tables; any preceding notebook model is replaced.
        """
        checked = validate_tables(tables=tables)
        if "p_linear" not in checked:
            raise ValueError(
                "The TATT two-point route requires full p_linear[z,k] tables."
            )
        if checked["sources"].shape[0] != self.nbin:
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
        first, second, density = core_ia_parameters(ia=ia, nbin=self.nbin)
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
        self.restore_core()
        core = self.core
        core.set_source_sample(nofz=source_table)
        core.set_lens_sample(nofz=lens_table)
        core.init_ntomo_powerspectra()
        cold = []
        if "p_linear_cb" in checked:
            cold = np.log(checked["p_linear_cb"]).flatten(order="F")
        # The old core stores [k,z], whereas the new tables store [z,k].
        # Fortran flattening of the latter gives exactly the former order.
        core.set_cosmology(
            omegam=checked["omega_m"], H0=100.0 * h,
            log10k_2D=np.log10(checked["k"]), z_2D=checked["z"],
            lnP_linear=np.log(checked["p_linear"]).flatten(order="F"),
            lnP_nonlinear=np.log(checked["p_nonlinear"]).flatten(order="F"),
            G=checked["growth"] * (1.0 + checked["z"]),
            z_1D=checked["z"], chi=checked["chi"], omegab=omega_b,
            z_G=checked["z"], omegan2=omega_nu, lnP_linear_cb=cold,
        )
        core.set_nuisance_shear_photoz(bias=np.zeros(self.nbin))
        core.set_nuisance_shear_calib(M=np.zeros(self.nbin))
        core.set_nuisance_ia(A1=first, A2=second, B_TA=density)
        core.set_omp_threads(n=self.threads)
        return core_power_rows(core=core, ell=ell, bins=bins)
