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
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from itertools import combinations_with_replacement

from .forecast import symmetric_matrix
from .ownership import immutable_copy


# One arcminute in radians; every compiled angular calculation uses radians.
ARCMIN_TO_RADIAN = np.pi / (180.0 * 60.0)


@dataclass(frozen=True)
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

    kind: np.ndarray
    bins: np.ndarray
    angles: np.ndarray
    values: np.ndarray
    labels: np.ndarray

    def __post_init__(self):
        """Own read-only rows so retained weights cannot outlive their definition.

        Construct a new SurveyRows to change bins, angles or measurements.
        The dataclass also refuses rebinding these arrays after construction.
        """
        for name in ("kind", "bins", "angles", "values", "labels"):
            values = np.asarray(getattr(self, name))
            object.__setattr__(self, name, immutable_copy(value=values))

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
                required = ["BIN1", "BIN2", "VALUE"]
                if name == "map3":
                    required.extend(["BIN3", "THETA1", "THETA2", "THETA3"])
                    angle_names = ("THETA1", "THETA2", "THETA3")
                else:
                    required.extend(["ANGLEMIN", "ANGLEMAX", "ANG"])
                    angle_names = ("ANGLEMIN", "ANGLEMAX", "ANG")
                _columns(table=table, names=required, extension=name)
                bins = np.full((len(table), 3), fill_value=-1, dtype=np.int32)
                bins[:, 0] = _bin_column(table=table, name="BIN1", nbin=nbin)
                bins[:, 1] = _bin_column(table=table, name="BIN2", nbin=nbin)
                if name == "map3":
                    bins[:, 2] = _bin_column(table=table, name="BIN3", nbin=nbin)
                angles = np.empty((len(table), 3), dtype=np.float64)
                for column, angle_name in enumerate(angle_names):
                    angles[:, column] = _angle_column(
                        hdu=hdu, name=angle_name, assumed_unit=assumed_angle_unit,
                    )
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


def read_source_distributions(path, nbin, extension="nz_source"):
    """Read source n(z) midpoint values from a two-point FITS file.

    This function preserves the supplied midpoint-curve convention.
    Histogram reconstruction is a different numerical choice and must
    be made explicitly before passing densities to the C model.

    Arguments:
      path = FITS file containing the named source extension.
      nbin = number of BIN columns to read.
      extension = FITS table with Z_MID and BIN1,...,BINnbin.
    Returns:
      z[nz] and sources[nbin,nz], with no hidden renormalization.
    Raises:
      ValueError for missing columns, nonfinite/negative densities or bad z order.
    """
    with fits.open(Path(path), memmap=False) as hdus:
        table = hdus[extension].data
        names = ["Z_MID"]
        for index in range(nbin):
            names.append(f"BIN{index + 1}")
        _columns(table=table, names=names, extension=extension)
        z = np.array(table["Z_MID"], dtype=np.float64, copy=True)
        sources = np.empty((nbin, z.size), dtype=np.float64)
        for index in range(nbin):
            sources[index] = table[f"BIN{index + 1}"]
    if (z.size < 2 or np.any(np.diff(z) <= 0.0) or np.any(z < 0.0)
            or not np.all(np.isfinite(z)) or not np.all(np.isfinite(sources))
            or np.any(sources < 0.0)):
        raise ValueError(
            "Source curves require increasing finite z and nonnegative n(z)."
        )
    return z, sources


def forecast_rows(nbin, aperture_radii, xi_edges=None):
    """Define a synthetic forecast vector without requiring measured values.

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
