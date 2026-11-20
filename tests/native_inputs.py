"""Direct native setup for supplied-array regression tests only.

The production likelihood reads Cobaya products; these fixtures install
small explicit arrays to isolate cache invalidation and interpolation.
"""
import numpy as np
from des3pcf.physical_provider import validate_tables


def init_two_point(
    core, nbin, threads=1, interpolation_type=0, accuracy_boost=1.0,
    integration_accuracy=0, fpt_internal_boost=0.5, ell_internal=192,
):
    """Set the two-point selectors using the ordinary project interface."""
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
    core.init_photoz_histogram_akima(enabled=0, preparation_factor=3)
    core.init_photoz_conventions(
        interpolation_type=interpolation_type, zmid_convention=1,
    )
    core.init_source_sample_size(Ntomo=nbin)
    # One unused lens bin permits the shared pair bookkeeping to be
    # initialized. Only source-source spectra are ever requested here.
    core.init_lens_sample_size(Ntomo=1)
    core.init_IA(ia_model=1, ia_redshift_evolution=3, ia_code=0)
    core.reset_bary_struct()

def two_point_spectra(
    parameters, tables, metadata, ia, ell, bins,
    core, nbin, threads=1, interpolation_type=0, accuracy_boost=1.0,
    integration_accuracy=0, fpt_internal_boost=0.5, ell_internal=192,
):
    """Supply cosmology and nuisance arrays to the ordinary two-point core.

    The power arrays use (Mpc/h)^3 and k uses h/Mpc. Sources already
    include photo-z shifts, so the core's shift is zero here. The existing
    CFASTPT calculation supplies the TATT E/B spectra; no duplicate loops
    are calculated in Python.
    """

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
    init_two_point(
        core=core, nbin=nbin, threads=threads, interpolation_type=interpolation_type,
        accuracy_boost=accuracy_boost, integration_accuracy=integration_accuracy,
        fpt_internal_boost=fpt_internal_boost, ell_internal=ell_internal,
    )
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
    core.set_nuisance_ia(A1=first, A2=second, B_TA=density)
    core.set_omp_threads(n=threads)
    return core_power_rows(core=core, ell=ell, bins=bins)

def check_bispectrum(baseline, tables=None):
    """Check the theory arrays before sending them to the C projection.

    The native dictionary holds z, ln(middle k), ln(short/middle),
    (long-middle)/short and flattened ln B, with the last coordinate
    varying fastest. B is in (Mpc/h)^6. No fitting formula is assumed.
    When background arrays are provided, they must match the prediction.
    """
    native = baseline["native"]
    names = ("z", "logk", "logu", "v", "log_bispectrum")
    if set(native) != set(names) | {"boost", "clamp"}:
        raise ValueError("Bispectrum requires five arrays, boost and clamp.")
    size = 1
    for name in names:
        values = native[name]
        if (not isinstance(values, np.ndarray) or values.dtype != np.float64
                or values.ndim != 1 or not values.flags.c_contiguous
                or not np.all(np.isfinite(values))):
            raise ValueError(f"Bispectrum {name} must be a finite float64 vector.")
        if name != "log_bispectrum":
            if values.size < 4 or np.any(np.diff(values) <= 0.0):
                raise ValueError(f"Bispectrum {name} needs four increasing nodes.")
            regular = np.linspace(start=values[0], stop=values[-1], num=values.size)
            tolerance = 64.0 * np.finfo(float).eps * max(1.0, np.max(np.abs(values)))
            if not np.allclose(values, regular, rtol=0.0, atol=tolerance):
                raise ValueError(f"Bispectrum {name} must be uniformly sampled.")
            size *= values.size
    if native["log_bispectrum"].size != size:
        raise ValueError("Bispectrum values disagree with the four axis sizes.")
    if (native["z"][0] < 0.0 or native["logu"][-1] > 0.0
            or native["v"][0] < 0.0 or native["v"][-1] > 1.0):
        raise ValueError("Bispectrum requires z>=0, logu<=0 and v in [0,1].")
    boost = native["boost"]
    if (not np.isfinite(boost) or boost < 1 or boost != int(boost)
            or native["clamp"] not in (0, 1)):
        raise ValueError("Bispectrum boost must be a positive integer; clamp is 0 or 1.")
    if tables is not None and baseline.get("background_z") is not None:
        if (not np.array_equal(baseline["background_z"], tables["z"])
                or not np.array_equal(baseline["background_omega_m_z"],
                                      tables["omega_m_z"])):
            raise ValueError("Bispectrum and projection have different background arrays.")


def core_power_rows(core, ell, bins):
    """Read full E/B power from an already prepared ordinary CosmoLike module.

    The existing CFASTPT calculation supplies the two-point TATT model.
    Its batch wrapper fills only source pairs with i<=j in cubes ordered
    [ell,i,j]. This adapter selects the correct triangle of those cubes
    and never edits the data-vector C source files. The caller must first
    prepare that core with the same physical parameters, source curves,
    IA normalization and redshift pivot as the three-point model.

    Arguments:
      core = configured project extension exposing C_ss_tomo_limber(l=...).
      ell = positive 1D multipoles, identical to the XiEngine integration grid.
      bins = zero-based [nrow,2] source pairs in the requested row order.
    Returns:
      owned C-contiguous E and B signal powers [nrow,nell].
    Raises:
      ValueError for a shape/bin mismatch or nonfinite core output.
    Side effects:
      The existing core may fill its own cosmology-dependent caches.
    """
    ell = np.asarray(ell, dtype=np.float64)
    pairs = np.asarray(bins)
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu"
            or np.any(pairs < 0)):
        raise ValueError(
            "Core shear projection requires nonnegative integer source pairs."
        )
    even_cube, odd_cube = core.C_ss_tomo_limber(l=ell)
    even_cube = np.asarray(even_cube, dtype=np.float64)
    odd_cube = np.asarray(odd_cube, dtype=np.float64)
    if (even_cube.ndim != 3 or even_cube.shape != odd_cube.shape
            or even_cube.shape[0] != ell.size or even_cube.shape[1] != even_cube.shape[2]
            or np.any(pairs >= even_cube.shape[1])):
        raise ValueError(
            "Existing core's [ell,source,source] spectra do not match these rows."
        )
    even = np.empty((pairs.shape[0], ell.size), dtype=np.float64)
    odd = np.empty_like(even)
    for row in range(pairs.shape[0]):
        first = int(min(pairs[row]))
        second = int(max(pairs[row]))
        even[row] = even_cube[:, first, second]
        odd[row] = odd_cube[:, first, second]
    if not np.all(np.isfinite(even)) or not np.all(np.isfinite(odd)):
        raise ValueError("Existing core returned nonfinite E/B power.")
    return even, odd
