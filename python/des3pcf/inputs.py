"""Physical input tables for the C bispectrum and covariance calculations.

The port accepts power and distance tables rather than running a Boltzmann
solver inside an OpenMP worker. Preparation therefore happens once before
parallel triangle or halo evaluation. Wavenumbers are in h/Mpc, distances
in Mpc/h, and power in (Mpc/h)^3. Growth is D(z), normalized to D(0)=1;
the older CosmoLike quantity D/a must not be passed without conversion.

Saved NPZ tables contain numerical arrays and a JSON metadata string;
loading them never permits Python pickle objects. CAMB is an optional
table producer using the installation already supplied by Cocoa. This
module neither installs software nor starts a calculation on import.
"""

import json
from pathlib import Path

import numpy as np


# These are the exact constructor names understood by the C++ physics model.
ARRAY_NAMES = (
    "k", "p_linear0", "z", "p_nonlinear", "chi", "dz_dchi",
    "growth", "sources", "los_z",
)
SCALAR_NAMES = ("omega_m", "omega_de", "w0", "wa", "ns", "sigma8")
# Full redshift-dependent linear spectra feed the unchanged two-point core.
OPTIONAL_ARRAY_NAMES = ("p_linear", "p_linear_cb", "p_linear_nu")
# Native samples define a reconstruction; source_z defines its numerical
# read/integration grid. Neither axis is a CAMB transfer-redshift request.
SOURCE_ARRAY_NAMES = (
    "source_z", "source_native_z", "source_native",
    "source_native_low", "source_native_high",
)


def _source_edge_arrays(native_z, native_low, native_high):
    """Check supplied cell edges without inferring widths from midpoint spacing.

    The DES input includes a zero-valued observer row whose cell overlaps
    the next cell. Preserve these supplied edges; they set outer support,
    not a new histogram integration or cell-probability reconstruction.
    """
    low = np.asarray(native_low, dtype=np.float64)
    high = np.asarray(native_high, dtype=np.float64)
    if (low.shape != native_z.shape or high.shape != native_z.shape
            or not np.all(np.isfinite(low)) or not np.all(np.isfinite(high))
            or np.any(np.diff(low) <= 0.0) or np.any(np.diff(high) <= 0.0)
            or np.any(low >= high) or np.any(native_z < low)
            or np.any(native_z > high)):
        raise ValueError("Source edges must bracket their increasing native midpoints.")
    return low, high


def validate_tables(tables):
    """Copy and check the unit-specific arrays needed by PhysicsModel.

    Rejecting a mismatched axis here prevents a Python exception from
    becoming a fatal C validation message. The C++ layer repeats its own
    checks because callers can also use the extension directly.

    Arguments:
      tables = dictionary containing ARRAY_NAMES and SCALAR_NAMES.
    Returns:
      independent C-contiguous float64 arrays and scalar values.
    Raises:
      ValueError for missing fields, shapes, domains or normalization.
    """
    result = {}
    for name in ARRAY_NAMES:
        if name not in tables:
            raise ValueError(f"Physics tables lack {name}; regenerate the input file.")
        array = np.array(tables[name], dtype=np.float64, order="C", copy=True)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"Physics table {name} must be nonempty and finite.")
        result[name] = array
    for name in SCALAR_NAMES:
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
    if result["los_z"][0] <= 0.0 or result["los_z"][-1] > z[-1]:
        raise ValueError("LOS nodes must lie within 0<z<=the physical table maximum.")
    for name in ("chi", "dz_dchi", "growth"):
        if result[name].shape != z.shape:
            raise ValueError(f"{name} must have one value per redshift node.")
    if result["p_linear0"].shape != k.shape:
        raise ValueError("p_linear0 must have one value per k node.")
    if result["p_nonlinear"].shape != (z.size, k.size):
        raise ValueError("p_nonlinear must have shape [nz,nk], with k contiguous.")
    for name in SOURCE_ARRAY_NAMES:
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
    for name in OPTIONAL_ARRAY_NAMES:
        if name in tables:
            values = np.array(tables[name], dtype=np.float64, order="C", copy=True)
            if (values.shape != (z.size, k.size) or np.any(values <= 0.0)
                    or not np.all(np.isfinite(values))):
                raise ValueError(f"Optional {name} must be positive finite [nz,nk].")
            result[name] = values
    return result


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
        for name in ARRAY_NAMES:
            tables[name] = archive[name]
        for name in SCALAR_NAMES:
            tables[name] = metadata[name]
        for name in OPTIONAL_ARRAY_NAMES + SOURCE_ARRAY_NAMES:
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
    for name in ARRAY_NAMES:
        arrays[name] = checked[name]
    for name in SCALAR_NAMES:
        record[name] = checked[name]
    for name in OPTIONAL_ARRAY_NAMES + SOURCE_ARRAY_NAMES:
        if name in checked:
            arrays[name] = checked[name]
    arrays["metadata_json"] = np.array(json.dumps(record, sort_keys=True))
    # An open file prevents NumPy from silently appending another suffix.
    mode = "wb" if overwrite else "xb"
    with destination.open(mode=mode) as stream:
        np.savez_compressed(stream, **arrays)
    return destination


def shifted_sources(z, sources, shifts):
    """Apply n_i(z-delta_z_i) and normalize each shifted distribution.

    Positive delta_z moves galaxies toward larger redshift. Values beyond
    the supplied support are zero. Normalizing after the shift makes the
    treatment of support loss explicit; increase the grid if that loss
    matters for a derivative. The original arrays are never changed.

    Arguments:
      z = increasing 1D redshift grid.
      sources = nonnegative [nbin,nz] densities per unit redshift.
      shifts = finite [nbin] redshift biases.
    Returns:
      C-contiguous float64 [nbin,nz] densities with unit trapezoidal integrals.
    Raises:
      ValueError for malformed inputs or a shift removing all source support.
    """
    z = np.asarray(z, dtype=np.float64)
    sources = np.asarray(sources, dtype=np.float64)
    shifts = np.asarray(shifts, dtype=np.float64)
    if (z.ndim != 1 or z.size < 2 or np.any(np.diff(z) <= 0.0)
            or sources.ndim != 2 or sources.shape[1] != z.size
            or shifts.shape != (sources.shape[0],)):
        raise ValueError("Photo-z inputs require z[nz], sources[nbin,nz], shifts[nbin].")
    if (not np.all(np.isfinite(z)) or not np.all(np.isfinite(sources))
            or not np.all(np.isfinite(shifts)) or np.any(sources < 0.0)):
        raise ValueError(
            "Photo-z inputs must be finite and source densities nonnegative."
        )
    output = np.empty_like(sources, order="C")
    for index in range(sources.shape[0]):
        output[index] = np.interp(
            z - shifts[index], z, sources[index], left=0.0, right=0.0,
        )
        # A shifted curve is defined only at positive source distance. Pin
        # its observer endpoint before normalization, also for negative
        # shifts. A finite grid interpolates over this first cell; refine
        # the source grid if appreciable probability reaches that cell.
        if z[0] == 0.0:
            output[index, 0] = 0.0
        normalization = np.trapz(output[index], x=z)
        if normalization <= 0.0:
            raise ValueError(f"Photo-z shift removes all sources in bin {index + 1}.")
        output[index] /= normalization
    return output


def reconstructed_sources(
    native_z, native_sources, source_z, shifts,
    reconstruction="linear_midpoint", padding="zero_outside_native",
    native_low=None, native_high=None,
):
    """Shift the original source curve onto a fixed independent uniform grid.

    PCHIP is a declared positive, shape-preserving cubic reconstruction of
    the midpoint samples, not an accuracy boost or a new calibration. Its
    padding choice adds zero-valued endpoint samples and is recorded
    explicitly. zero_at_measured_edges uses the supplied outer FITS edges,
    intersected with z>=0, and a zero plateau to the background upper bound.
    This preserves that support, not each histogram cell's probability.
    The linear choice preserves zero outside the original midpoint range.

    Arguments:
      native_z = original increasing midpoint coordinates [nnative].
      native_sources = original nonnegative densities [nbin,nnative].
      source_z = fixed uniform grid [nsource], starting at the observer zero.
      shifts = absolute photo-z shifts [nbin]; positive moves sources upward.
      reconstruction = linear_midpoint or explicitly selected pchip_midpoint.
      padding = zero_outside_native for linear, or the explicit PCHIP policy
                zero_at_observer_and_background_edge or zero_at_measured_edges.
      native_low, native_high = original Z_LOW/Z_HIGH [nnative], required
                for measured-edge padding; never inferred from midpoints.
    Returns:
      normalized sources [nbin,nsource] and a source preparation record.
      Norms are fixed-grid trapezoidal integrals before renormalization;
      they expose support loss and quadrature changes, not exact probabilities.
    The observer sample is zero even when a negative shift carries density
    there. Surviving support is normalized after each shift. Input arrays
    are never changed, and no previously shifted curve is used as input.
    Nonnegative source samples do not guarantee that the C model's later
    cubic interpolation of a divided IA kernel is nonnegative. That
    separate finite-grid interpolation still needs a convergence check.
    """
    native_z = np.asarray(native_z, dtype=np.float64)
    native = np.asarray(native_sources, dtype=np.float64)
    nodes = np.asarray(source_z, dtype=np.float64)
    offsets = np.asarray(shifts, dtype=np.float64)
    if (native_z.ndim != 1 or native_z.size < 2 or native_z[0] < 0.0
            or not np.all(np.isfinite(native_z)) or np.any(np.diff(native_z) <= 0.0)
            or native.ndim != 2 or native.shape[0] == 0
            or native.shape[1] != native_z.size
            or not np.all(np.isfinite(native)) or np.any(native < 0.0)
            or offsets.shape != (native.shape[0],)
            or not np.all(np.isfinite(offsets))):
        raise ValueError("Native source axes, densities and shifts are inconsistent.")
    if (nodes.ndim != 1 or nodes.size < 4 or nodes[0] != 0.0
            or not np.all(np.isfinite(nodes)) or np.any(np.diff(nodes) <= 0.0)):
        raise ValueError("Independent source grid needs >=4 increasing nodes from zero.")
    expected = np.linspace(start=0.0, stop=nodes[-1], num=nodes.size)
    if not np.allclose(nodes, expected, rtol=0.0,
                       atol=32.0 * np.finfo(float).eps * nodes[-1]):
        raise ValueError(
            "Prepared source_z must be uniform for the ordinary-core bridge."
        )
    if np.any(native[:, native_z > nodes[-1]] != 0.0):
        raise ValueError("Nonzero native source support exceeds the background edge.")
    support = (0.0, float(nodes[-1]))
    if reconstruction == "linear_midpoint":
        if padding != "zero_outside_native":
            raise ValueError(
                "Linear midpoint sources require zero_outside_native padding."
            )
        curve = None
    elif reconstruction == "pchip_midpoint":
        if padding not in ("zero_at_observer_and_background_edge",
                           "zero_at_measured_edges"):
            raise ValueError("PCHIP requires explicit zero endpoint padding.")
        from scipy.interpolate import PchipInterpolator

        if padding == "zero_at_measured_edges":
            low, high = _source_edge_arrays(
                native_z=native_z, native_low=native_low, native_high=native_high,
            )
            support = (max(0.0, float(low[0])), float(high[-1]))
            if not 0.0 <= support[0] < support[1] < nodes[-1]:
                raise ValueError(
                    "Measured-edge PCHIP needs background beyond the source upper edge "
                    "for its explicit zero plateau."
                )
        inside = (native_z >= support[0]) & (native_z <= support[1])
        if np.any(native[:, ~inside] != 0.0):
            raise ValueError(
                "Positive native samples lie outside the selected source support."
            )
        knots = native_z[inside].copy()
        values = native[:, inside].copy()
        if knots.size < 2:
            raise ValueError(
                "PCHIP needs at least two native samples inside the background."
            )
        if knots[0] > support[0]:
            knots = np.concatenate((np.asarray([support[0]]), knots))
            values = np.column_stack((np.zeros(native.shape[0]), values))
        elif np.any(values[:, 0] != 0.0):
            raise ValueError(
                "PCHIP lower padding cannot replace a nonzero native sample."
            )
        if knots[-1] < support[1]:
            knots = np.concatenate((knots, np.asarray([support[1]])))
            values = np.column_stack((values, np.zeros(native.shape[0])))
        elif np.any(values[:, -1] != 0.0):
            raise ValueError(
                "A nonzero native sample occupies the selected upper support edge; "
                "choose a different explicit support policy rather than overwrite it."
            )
        # The extra zero segment makes the derivative at the measured upper
        # edge zero in PCHIP. It adds no sources beyond the declared support.
        # A physical observer edge at zero need not have a zero derivative:
        # low-z truncation and renormalization still need their own h tests.
        if support[0] > 0.0:
            knots = np.concatenate((np.zeros(1), knots))
            values = np.column_stack((np.zeros(native.shape[0]), values))
        if support[1] < nodes[-1]:
            knots = np.concatenate((knots, nodes[-1:]))
            values = np.column_stack((values, np.zeros(native.shape[0])))
        curve = PchipInterpolator(x=knots, y=values, axis=1, extrapolate=False)
    else:
        raise ValueError("Choose linear_midpoint or pchip_midpoint reconstruction.")
    output = np.zeros(shape=(native.shape[0], nodes.size), dtype=np.float64)
    normalizations = np.empty(shape=native.shape[0])
    for index in range(native.shape[0]):
        coordinates = nodes - offsets[index]
        if curve is None:
            output[index] = np.interp(
                x=coordinates, xp=native_z, fp=native[index], left=0.0, right=0.0,
            )
        else:
            inside = (coordinates > support[0]) & (coordinates < support[1])
            output[index, inside] = curve(coordinates[inside])[index]
        output[index, 0] = 0.0
        if not np.all(np.isfinite(output[index])) or np.any(output[index] < 0.0):
            raise ValueError(
                "Source reconstruction produced invalid density; no clipping."
            )
        normalization = np.trapz(output[index], x=nodes)
        if not np.isfinite(normalization) or normalization <= 0.0:
            raise ValueError(f"Photo-z shift removes sources in bin {index + 1}.")
        normalizations[index] = normalization
        output[index] /= normalization
    record = {
        "source_reconstruction": reconstruction,
        "source_padding": padding,
        "source_nz": int(nodes.size),
        "source_shift_normalizations": normalizations.tolist(),
        "source_shifts": offsets.tolist(),
        "source_normalization": "unit fixed-grid trapezoidal area after each shift",
        "source_observer_value": "zero at actual z=0",
    }
    if padding == "zero_at_measured_edges":
        record["source_unshifted_support"] = list(support)
        record["source_support_convention"] = (
            "unshifted outer Z_LOW/Z_HIGH intersected with nonnegative redshift; "
            "PCHIP preserves support, not individual histogram cell areas"
        )
    return output, record


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


def camb_tables(
    parameters, k, z, sources, los_z, halofit_version="takahashi",
    kmax_boltzmann=5.0, extrap_kmax=None, source_z=None,
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
    Returns:
      validated physical tables and metadata recording CAMB and model choices.
    Raises:
      ValueError for unavailable cosmology choices or invalid grids.
      ImportError when CAMB is absent from the selected environment.
    Side effects:
      Runs CAMB serially; does not write files or alter an existing core state.
    """
    import camb

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
    settings.set_matter_power(redshifts=z[::-1], kmax=kmax_boltzmann, silent=True)
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
    return validate_tables(tables=tables), metadata
