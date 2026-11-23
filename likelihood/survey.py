"""Read survey rows without changing their source-bin or radius association.

The joint vector is ordered as xi_plus, xi_minus, then third-order
aperture moments. A row of the latter carries three source labels and
three filter radii: permuting labels without the corresponding radii
would define a different observable. This reader preserves every file
row and requires covariance block labels to agree with that order.

DES Y3 and Y6 use separate files and selections. File names are not used
to infer a release, likelihood prescription or nuisance prior. Measured
values can be used for comparisons, while Fisher calculations use the
same row definitions to evaluate model derivatives.

The files themselves are the released DES Y3 products bundled under
data/upstream_y3/ (simulated vectors) and data/upstream_y3_observed/
(observed measurements): the xi and Map3 measurement tables, the joint
covariance and the nz_source table; data/README.md records their origin
and hashes. The likelihood's initialize reads them once, through
read_survey, read_joint_covariance and read_source_distributions, and a
notebook that needs the raw rows calls the same functions on the same
dataset entries.
"""

from pathlib import Path

import numpy as np
from astropy.io import fits
from itertools import combinations_with_replacement

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
      a record; construction through read_survey validates and owns its arrays.
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


def _columns(table, names, extension):
    """Check that an observable table contains its documented columns.

    Arguments:
      table = FITS binary-table data.
      names = required case-sensitive column names.
      extension = table name included in an error.
    Returns:
      nothing when all required columns are present.
    Raises:
      ValueError listing a missing column.
    """
    for name in names:
        if name not in table.names:
            raise ValueError(
                f"{extension} lacks {name}; supply a labeled measurement table."
            )


def _angle_column(hdu, name, assumed_unit):
    """Convert an angular column after checking its declared unit.

    Arguments:
      hdu = FITS binary table containing name.
      name = angular column name.
      assumed_unit = explicit convention used only when TUNIT is absent.
    Returns:
      owned float64 column in radians.
    Raises:
      ValueError for an unsupported unit or nonpositive/nonfinite angle.
    """
    unit = hdu.columns[name].unit
    if unit is None or unit == "":
        unit = assumed_unit
    factors = {"arcmin": ARCMIN_TO_RADIAN, "rad": 1.0, "deg": np.pi / 180.0}
    if unit not in factors:
        raise ValueError(f"{hdu.name}.{name} unit {unit!r} must be arcmin, rad or deg.")
    values = np.asarray(hdu.data[name], dtype=np.float64) * factors[unit]
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{hdu.name}.{name} requires positive finite angles.")
    return values


def _bin_column(table, name, nbin):
    """Convert FITS one-based source labels to checked zero-based integers.

    Arguments:
      table = FITS data containing integer-like bin labels.
      name = column name, such as BIN1.
      nbin = number of available source distributions.
    Returns:
      owned int32 column in the interval [0,nbin).
    Raises:
      ValueError for fractional, missing or out-of-range source labels.
    """
    values = np.asarray(table[name])
    if (not np.all(np.isfinite(values)) or np.any(values != np.floor(values))
            or np.any(values < 1) or np.any(values > nbin)):
        raise ValueError(f"{name} must contain integer source labels from 1 to {nbin}.")
    return np.asarray(values - 1, dtype=np.int32)


def read_survey(two_point_path, aperture_path, nbin, assumed_angle_unit="arcmin"):
    """Read the full xi± and aperture measurement rows in joint order.

    The likelihood's initialize calls this once with the dataset's
    two_point_file and aperture_file: the released DES Y3 FITS
    measurement tables bundled under data/ (module docstring). A
    notebook that needs the raw measured rows calls it the same way.
    Either input can be None for a single-probe calculation. Angular bin
    edges are required for xi: evaluating a Bessel function at the center
    is not the same as averaging the estimator over its angular bin.

    Arguments:
      two_point_path = optional FITS file with xip and xim extensions.
      aperture_path = optional FITS file with map3 extension.
      nbin = positive number of source bins.
      assumed_angle_unit = explicit unit for columns lacking FITS TUNIT.
    Returns:
      SurveyRows ordered xip, xim, map3, retaining order within each block.
    Raises:
      ValueError for absent inputs, malformed columns or duplicate row labels.
      OSError when a file cannot be opened.
    """
    if (two_point_path is None and aperture_path is None) or nbin < 1:
        raise ValueError(
            "Supply at least one survey file and a positive source-bin count."
        )
    kinds = []
    bin_blocks = []
    angle_blocks = []
    value_blocks = []
    labels = []
    inputs = []
    if two_point_path is not None:
        inputs.append((two_point_path, ("xip", "xim")))
    if aperture_path is not None:
        inputs.append((aperture_path, ("map3",)))
    for path, extensions in inputs:
        with fits.open(Path(path), memmap=False) as hdus:
            for name in extensions:
                hdu = hdus[name]
                table = hdu.data

                # Required columns per row kind: every table carries two
                # source bins and a value; a Map3 table adds the third
                # source bin and its three filter radii, a xi table its
                # angular bin edges and quoted centre.
                required = ["BIN1", "BIN2", "VALUE"]
                if name == "map3":
                    required.extend(["BIN3", "THETA1", "THETA2", "THETA3"])
                    angle_names = ("THETA1", "THETA2", "THETA3")
                else:
                    required.extend(["ANGLEMIN", "ANGLEMAX", "ANG"])
                    angle_names = ("ANGLEMIN", "ANGLEMAX", "ANG")
                _columns(table=table, names=required, extension=name)

                # Source bins: the FITS labels are one-based, and
                # _bin_column checks and converts them to the zero-based
                # convention of the compiled code. The third column
                # stays -1 for xi rows, which have only a source pair.
                bins = np.full((len(table), 3), fill_value=-1, dtype=np.int32)
                bins[:, 0] = _bin_column(table=table, name="BIN1", nbin=nbin)
                bins[:, 1] = _bin_column(table=table, name="BIN2", nbin=nbin)
                if name == "map3":
                    bins[:, 2] = _bin_column(table=table, name="BIN3", nbin=nbin)

                # Angles: every column is converted to radians, the unit
                # of every compiled angular calculation; _angle_column
                # reads the declared FITS unit first.
                angles = np.empty((len(table), 3), dtype=np.float64)
                for column, angle_name in enumerate(angle_names):
                    angles[:, column] = _angle_column(
                        hdu=hdu, name=angle_name, assumed_unit=assumed_angle_unit,
                    )

                # A valid xi angular bin has lower < upper and its
                # quoted centre inside the bin; a centre outside its own
                # bin means the edge and centre columns were paired
                # wrongly in the file.
                if name != "map3" and (
                    np.any(angles[:, 0] >= angles[:, 1])
                    or np.any(angles[:, 2] < angles[:, 0])
                    or np.any(angles[:, 2] > angles[:, 1])
                ):
                    raise ValueError(
                        f"{name} requires lower<upper and center inside each bin."
                    )
                values = np.asarray(table["VALUE"], dtype=np.float64)
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"{name}.VALUE must be finite.")

                # Row labels: the kind, the source bins and the exact
                # bits of each angle. float.hex() writes the full binary
                # value, so a regenerated or reordered file that changes
                # an angle even in its last digit produces a different
                # label and is caught; a rounded decimal string would
                # let such a change pass.
                for row in range(len(table)):
                    parts = [name]
                    for bin_value in bins[row]:
                        parts.append(str(int(bin_value)))
                    for angle in angles[row]:
                        parts.append(float(angle).hex())
                    labels.append(":".join(parts))
                kinds.append(np.full(len(table), fill_value=name, dtype="U4"))
                bin_blocks.append(bins)
                angle_blocks.append(angles)
                value_blocks.append(values.copy())

    # Two equal labels would mean the same observable row appears twice,
    # making any match against a covariance ambiguous.
    if len(set(labels)) != len(labels):
        raise ValueError("Measurement file contains duplicate observable rows.")
    return SurveyRows(
        kind=np.concatenate(kinds),
        bins=np.concatenate(bin_blocks),
        angles=np.concatenate(angle_blocks),
        values=np.concatenate(value_blocks),
        labels=np.asarray(labels),
    )


def read_joint_covariance(path, rows):
    """Read COVMAT only when its full block labels match the measurement order.

    The FITS convention records NAME_i and STRT_i for each block. Matrix
    shape alone cannot distinguish a covariance ordered (map3,xi) from one
    ordered (xi,map3). The labels therefore remain mandatory even for a
    square matrix with the expected number of rows.

    Arguments:
      path = FITS file containing the joint COVMAT image.
      rows = unselected SurveyRows in that file's block order.
    Returns:
      owned, symmetric, positive definite full covariance.
    Raises:
      ValueError for missing/mismatched block metadata or matrix dimensions.
    """
    # Compress rows.kind into contiguous runs: one (block name, starting
    # row) pair per run of equal kinds, which is exactly what the FITS
    # convention stores in the COVMAT header as NAME_i / STRT_i.
    expected = []
    previous = None
    for index, kind in enumerate(rows.kind):
        if kind != previous:
            expected.append((str(kind), index))
            previous = kind
    with fits.open(Path(path), memmap=False) as hdus:
        image = hdus["COVMAT"]
        for index, (name, start) in enumerate(expected):
            observed_name = str(image.header.get(f"NAME_{index}", "")).lower()
            observed_start = image.header.get(f"STRT_{index}")
            if observed_name != name or observed_start != start:
                raise ValueError(
                    f"Covariance block {index} must be {name} starting at {start}."
                )
        # A header block beyond the expected ones would mean the matrix
        # covers observables this measurement does not contain.
        if f"NAME_{len(expected)}" in image.header:
            raise ValueError(
                "Covariance contains extra blocks absent from the supplied rows."
            )
        matrix = symmetric_matrix(image.data, name="Joint covariance", positive=True)
    if matrix.shape != (rows.values.size, rows.values.size):
        raise ValueError(
            "Joint covariance size differs from the full measurement row count."
        )
    return matrix


def read_source_distributions(path, nbin, extension="nz_source", with_edges=False):
    """Read source n(z) midpoint values from a two-point FITS file.

    The likelihood's initialize reads the dataset's nz_source_file
    through this once; the returned curves are the native inputs that
    reconstructed_sources shifts at every sampled point. This function
    preserves the supplied midpoint-curve convention.
    Histogram reconstruction is a different numerical choice and must
    be made explicitly before passing densities to the C model.

    Arguments:
      path = FITS file containing the named source extension.
      nbin = number of BIN columns to read.
      extension = FITS table with Z_MID and BIN1,...,BINnbin.
      with_edges = also return the supplied Z_LOW/Z_HIGH, without inferring
          edges or interpreting cell heights as integrated probabilities.
    Returns:
      z[nz] and sources[nbin,nz], with no hidden renormalization.
      If with_edges is True, append native_low[nz] and native_high[nz].
    Raises:
      ValueError for missing columns, nonfinite/negative densities or bad z order.
    """
    with fits.open(Path(path), memmap=False) as hdus:
        table = hdus[extension].data
        names = ["Z_MID"]
        for index in range(nbin):
            names.append(f"BIN{index + 1}")
        if with_edges:
            names.extend(("Z_LOW", "Z_HIGH"))
        _columns(table=table, names=names, extension=extension)
        z = np.array(table["Z_MID"], dtype=np.float64, copy=True)
        sources = np.empty((nbin, z.size), dtype=np.float64)
        for index in range(nbin):
            sources[index] = table[f"BIN{index + 1}"]
        if with_edges:
            low = np.array(table["Z_LOW"], dtype=np.float64, copy=True)
            high = np.array(table["Z_HIGH"], dtype=np.float64, copy=True)
    if (z.size < 2 or np.any(np.diff(z) <= 0.0) or np.any(z < 0.0)
            or not np.all(np.isfinite(z)) or not np.all(np.isfinite(sources))
            or np.any(sources < 0.0)):
        raise ValueError(
            "Source curves require increasing finite z and nonnegative n(z)."
        )
    if with_edges:
        return z, sources, low, high
    return z, sources


def forecast_rows(nbin, aperture_radii, xi_edges=None):
    """Define a synthetic forecast vector without requiring measured values.

    The synthetic rows drive the accuracy, quadrature and geometry tests
    (tests/test_accuracy.py, tests/test_quadrature.py,
    tests/test_estimator_geometry.py), which need row definitions at
    chosen angles without any measurement file. No production
    calculation reads them, so there is no forecast pipeline to look
    for behind this function.

    Equal-radius aperture moments are ordered by source triplet, then
    radius. Optional xi± rows include every source pair and angular bin.
    The stored zeros mark the absence of observations; they are never used
    as model predictions or fitted data in the Fisher calculation.

    Arguments:
      nbin = positive source-bin count.
      aperture_radii = positive aperture radii in radians, possibly empty.
      xi_edges = optional increasing angular bin edges in radians.
    Returns:
      SurveyRows with explicit labels and zero placeholder observations.
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
