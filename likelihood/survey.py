"""Describe the survey rows and read the ASCII inputs of the likelihood.

The joint vector is ordered as xi_plus, xi_minus, then third-order
aperture moments. A row of the latter carries three source labels and
three filter radii: permuting labels without the corresponding radii
would define a different observable.

The inputs follow lsst_y1. The .dataset file names ASCII files in data/:
data_file and mask_file ("index value" per line), cov_file ("i j value"
per line, upper triangle) and nz_source_file ("z n1 ... n4" per line).
The row geometry is not in those files: survey_rows rebuilds it from the
.dataset keys (source_ntomo, n_theta, theta_min_arcmin, theta_max_arcmin,
map3_radii_arcmin), as lsst_y1 does. scripts/convert_desy3_fits.py writes
the ASCII files from the released DES Y3 FITS files in data/upstream_y3/
and data/upstream_y3_observed/ and checks that their rows equal those
rebuilt rows; no likelihood code opens a FITS file.
"""

from itertools import combinations_with_replacement

import numpy as np

from cosmolike_notebook_utils.fisher import symmetric_matrix


# One arcminute in radians; every compiled angular calculation uses radians.
ARCMIN_TO_RADIAN = np.pi / (180.0 * 60.0)


class SurveyRows:
    """Hold the explicit observable definition and optional measured values.

    Arguments:
      kind = [nrow] strings 'xip', 'xim', or 'map3'.
      bins = [nrow,3] zero-based source bins; third entry is -1 for xi.
      angles = [nrow,3] radians: xi lower/upper/center, or three aperture radii.
      values = [nrow] observed dimensionless shear correlations/moments.
      labels = unique strings identifying rows independently of their values.
    Returns:
      a record that owns copies of its arrays.
    """

    def __init__(self, kind, bins, angles, values, labels):
        """Copy the five measurement columns into ordinary public arrays.

        Changing rows after native initialization requires reinitializing
        its weights and covariance, as with other Cocoa input arrays.
        """
        self.kind = np.array(object=kind, copy=True)
        self.bins = np.array(object=bins, copy=True)
        self.angles = np.array(object=angles, copy=True)
        self.values = np.array(object=values, copy=True)
        self.labels = np.array(object=labels, copy=True)

    def select(self, indices):
        """Keep exactly the requested rows in their requested order.

        Arguments:
          indices = unique integer row indices within this record.
        Returns:
          a new SurveyRows containing owned selected arrays.
        Raises:
          ValueError for noninteger, repeated or out-of-range indices.
        """
        selected = np.asarray(indices)
        # One predicate per failure mode: the selection must be
        # one-dimensional; integer-typed (dtype.kind "i"/"u" are numpy's
        # signed and unsigned integer classes), because a float index
        # would be truncated silently; nonnegative, because a negative
        # index would wrap around to the end of the arrays; within the
        # row count; and free of repeats, because a repeated row would
        # silently double its weight in anything matched against these
        # rows.
        if (selected.ndim != 1 or selected.dtype.kind not in "iu"
                or np.any(selected < 0) or np.any(selected >= self.values.size)
                or np.unique(selected).size != selected.size):
            raise ValueError(
                "Selection requires unique integer indices within the row count."
            )
        return SurveyRows(
            kind=self.kind[selected].copy(),
            bins=self.bins[selected].copy(),
            angles=self.angles[selected].copy(),
            values=self.values[selected].copy(),
            labels=self.labels[selected].copy(),
        )


def read_vector(path, size):
    """Read an "index value" file: a data vector or a 0/1 mask.

    Arguments:
      path = ASCII file with one "index value" pair per line.
      size = expected number of lines (the number of survey rows).
    Returns:
      values[size], in index order.
    Raises:
      ValueError when the indices are not 0, 1, ..., size-1 in order or a
      value is not finite.
    """
    table = np.loadtxt(path, ndmin=2)
    if (table.shape != (size, 2) or not np.array_equal(table[:, 0], np.arange(size))
            or not np.all(np.isfinite(table[:, 1]))):
        raise ValueError(f"{path} must hold {size} lines 'index value', indices 0..{size - 1}.")
    return table[:, 1].copy()


def read_covariance(path, size):
    """Read an "i j value" covariance file (upper triangle, i <= j).

    Arguments:
      path = ASCII file; each line gives one entry and its mirror (j, i).
      size = number of survey rows.
    Returns:
      the symmetric, positive definite [size, size] matrix.
    Raises:
      ValueError for indices outside 0..size-1, a missing or repeated entry
      of the upper triangle, or a matrix that is not positive definite.
    """
    table = np.loadtxt(path, ndmin=2)
    if table.shape[1] != 3:
        raise ValueError(f"{path} must hold lines 'i j value'.")
    i = table[:, 0].astype(np.int64)
    j = table[:, 1].astype(np.int64)
    upper = np.minimum(i, j) * size + np.maximum(i, j)
    if (np.any(i != table[:, 0]) or np.any(j != table[:, 1]) or np.any(i < 0)
            or np.any(j < 0) or np.any(i >= size) or np.any(j >= size)
            or np.unique(upper).size != upper.size or upper.size != size * (size + 1) // 2):
        raise ValueError(f"{path} must give every entry i <= j < {size} exactly once.")
    matrix = np.zeros((size, size))
    matrix[i, j] = table[:, 2]
    matrix[j, i] = table[:, 2]
    return symmetric_matrix(matrix, name="Joint covariance", positive=True)


def read_sources(path, nbin, bin_width):
    """Read the source n(z) file "z n1 ... n_nbin" of histogram cells.

    Each line is a redshift cell of width bin_width centred on z, so its
    edges are z -+ bin_width / 2. The cell values are kept as given; no
    hidden renormalization.

    Arguments:
      path = ASCII n(z) file.
      nbin = number of source bins (columns after z).
      bin_width = redshift width of every cell (dataset key nz_source_bin_width).
    Returns:
      z[nz], sources[nbin, nz], low[nz] and high[nz] (the cell edges).
    Raises:
      ValueError for a wrong column count, z not increasing or negative,
      or a negative or nonfinite n(z) value.
    """
    table = np.loadtxt(path, ndmin=2)
    z = table[:, 0].copy()
    sources = np.ascontiguousarray(table[:, 1:].T)
    if (table.shape[1] != nbin + 1 or z.size < 2 or np.any(np.diff(z) <= 0.0)
            or np.any(z < 0.0) or not np.all(np.isfinite(table))
            or np.any(sources < 0.0) or not bin_width > 0.0):
        raise ValueError(
            f"{path} must hold 'z n1 ... n{nbin}' with increasing z >= 0 and n(z) >= 0.")
    return z, sources, z - 0.5 * bin_width, z + 0.5 * bin_width


def survey_rows(nbin, aperture_radii, xi_edges=None):
    """Define the survey rows from their geometry.

    The likelihood builds its rows here from the .dataset keys, as lsst_y1
    does: every source pair and angular bin for xi+ and xi-, then every
    source triplet and aperture radius for Map3 (equal radii, ordered by
    triplet, then radius). scripts/convert_desy3_fits.py checks that the
    released FITS rows are exactly these. Tests call it at chosen angles.
    The values are zeros; the likelihood fills them from data_file.

    Arguments:
      nbin = positive source-bin count.
      aperture_radii = positive aperture radii in radians, possibly empty.
      xi_edges = optional increasing angular bin edges in radians.
    Returns:
      SurveyRows with explicit labels and zero values.
    Raises:
      ValueError for invalid counts/angles or an empty overall vector.
    """
    if int(nbin) != nbin or nbin < 1:
        raise ValueError("Forecast tomography requires an integer nbin>=1.")
    radii = np.asarray(aperture_radii, dtype=np.float64)
    if (radii.ndim != 1 or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0)
            or np.unique(radii).size != radii.size):
        raise ValueError("Aperture radii must be distinct positive finite angles.")
    kinds = []
    bins = []
    angles = []
    labels = []
    if xi_edges is not None:
        edges = np.asarray(xi_edges, dtype=np.float64)
        if (edges.ndim != 1 or edges.size < 2 or np.any(edges <= 0.0)
                or np.any(np.diff(edges) <= 0.0) or not np.all(np.isfinite(edges))):
            raise ValueError("Xi edges must be positive, finite and strictly increasing.")
        for kind in ("xip", "xim"):
            for pair in combinations_with_replacement(range(nbin), 2):
                for index in range(edges.size - 1):
                    low = float(edges[index])
                    high = float(edges[index + 1])
                    kinds.append(kind)
                    bins.append((pair[0], pair[1], -1))
                    angles.append((low, high, np.sqrt(low * high)))
                    labels.append(f"{kind}:{pair[0]}:{pair[1]}:{low.hex()}:{high.hex()}")
    for triplet in combinations_with_replacement(range(nbin), 3):
        for radius in radii:
            kinds.append("map3")
            bins.append(triplet)
            angles.append((radius, radius, radius))
            labels.append(
                f"map3:{triplet[0]}:{triplet[1]}:{triplet[2]}:{float(radius).hex()}"
            )
    if not kinds:
        raise ValueError("Forecast rows need at least one aperture or xi bin.")
    return SurveyRows(
        kind=np.asarray(kinds, dtype="U4"),
        bins=np.asarray(bins, dtype=np.int32),
        angles=np.asarray(angles, dtype=np.float64),
        values=np.zeros(len(kinds), dtype=np.float64),
        labels=np.asarray(labels),
    )


def _source_edge_arrays(native_z, native_low, native_high):
    """Check supplied cell edges without inferring widths from midpoint spacing.

    The DES input includes a zero-valued observer row whose cell overlaps
    the next cell. Preserve these supplied edges; they set outer support,
    not a new histogram integration or cell-probability reconstruction.
    """
    low = np.asarray(native_low, dtype=np.float64)
    high = np.asarray(native_high, dtype=np.float64)
    # One predicate per failure mode: the edge arrays must match the
    # midpoint grid in shape and be finite; both must strictly increase
    # (the cells come in redshift order); every cell must be nonempty
    # (low < high); and every midpoint must sit inside its own cell.
    # Neighboring cells may overlap: the DES zero-valued observer row
    # legitimately overlaps the next cell, so overlap is deliberately
    # not refused here.
    if (low.shape != native_z.shape or high.shape != native_z.shape
            or not np.all(np.isfinite(low)) or not np.all(np.isfinite(high))
            or np.any(np.diff(low) <= 0.0) or np.any(np.diff(high) <= 0.0)
            or np.any(low >= high) or np.any(native_z < low)
            or np.any(native_z > high)):
        raise ValueError("Source edges must bracket their increasing native midpoints.")
    return low, high


def shifted_sources(z, sources, shifts):
    """Apply n_i(z-delta_z_i) and normalize each shifted distribution.

    The shifts delta_z_i are the photo-z calibration parameters of the
    DES source bins (dz1..dz4 in the example yamls): each sampled point
    moves bin i's redshift distribution rigidly by delta_z_i before the
    lensing kernels are built. This function applies the shift on the
    caller's own grid and serves the forecast tests
    (tests/test_forecast.py); the likelihood shifts through
    reconstructed_sources below, which also lands the curves on the
    fixed uniform grid the compiled model indexes.

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

    The likelihood's set_source_related calls this at every sampled
    point: the measured midpoint curves that initialize read are
    shifted by the sampled photo-z parameters dz_i and interpolated
    onto the fixed uniform source_z grid of the compiled model. The
    second return value records the choices made, so a caller can
    store them with its outputs.

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
    # The compiled model locates source samples by index arithmetic on
    # a uniform grid instead of searching, so the prepared grid must be
    # exactly uniform ("ordinary-core bridge" in the message names that
    # hand-over to the ordinary cosmolike core). The tolerance 32 eps
    # of the endpoint accepts the rounding of a linspace computed
    # elsewhere while rejecting any real non-uniformity.
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

        # Pad a zero sample at the lower support edge unless a native
        # sample already sits there; an existing nonzero sample at the
        # edge must never be overwritten by padding.
        if knots[0] > support[0]:
            knots = np.concatenate((np.asarray([support[0]]), knots))
            values = np.column_stack((np.zeros(native.shape[0]), values))
        elif np.any(values[:, 0] != 0.0):
            raise ValueError(
                "PCHIP lower padding cannot replace a nonzero native sample."
            )

        # The same at the upper support edge: pad a zero sample there,
        # and refuse to overwrite a nonzero native one.
        if knots[-1] < support[1]:
            knots = np.concatenate((knots, np.asarray([support[1]])))
            values = np.column_stack((values, np.zeros(native.shape[0])))
        elif np.any(values[:, -1] != 0.0):
            raise ValueError(
                "A nonzero native sample occupies the selected upper support edge; "
                "choose a different explicit support policy rather than overwrite it."
            )
        # Extend the knots down to z = 0 with a zero segment, so the
        # curve is defined (and zero) between the observer and the
        # measured lower edge. A physical observer edge at zero need
        # not have a zero derivative: low-z truncation and
        # renormalization still need their own finite-difference step
        # (h) convergence tests.
        if support[0] > 0.0:
            knots = np.concatenate((np.zeros(1), knots))
            values = np.column_stack((np.zeros(native.shape[0]), values))

        # Extend a zero plateau from the measured upper edge to the
        # background edge. Together with the zero sample at that edge,
        # the plateau makes the PCHIP derivative zero there, so the
        # cubic adds no sources beyond the declared support.
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
            # Strict inequalities: the support endpoints themselves keep
            # the padded zero value already present in output.
            inside = (coordinates > support[0]) & (coordinates < support[1])
            # The PCHIP object interpolates all bins at once (axis=1 at
            # construction), so curve(...) returns every bin's values
            # at these coordinates and [index] selects this bin's row.
            # Cold path: clarity over building one interpolator per bin.
            output[index, inside] = curve(coordinates[inside])[index]
        # The observer sample is pinned to zero even when a negative
        # shift carries density there: a source density is defined only
        # at positive distance (the convention the docstring states).
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
