"""Convert the released DES Y3 FITS files into the ASCII inputs Cocoa reads.

Cocoa likelihoods read plain text, as lsst_y1 does: a data vector
("index value" per line), a covariance ("i j value", upper triangle), a
0/1 mask ("index value") and a source n(z) ("z n1 n2 n3 n4"). The released
DES Y3 products are FITS files, kept unchanged in data/upstream_y3/
(simulated vectors at the CosmoGrid cosmology) and data/upstream_y3_observed/
(the observed, unblinded vectors), both from github.com/rchgomes/
3pcf_integrator at commit d6098d3. This script turns them into the
likelihood's inputs: it writes the ASCII files into data/, and the likelihood
never opens a FITS file (some covariance notebooks and measurement scripts
still read the upstream FITS for their own comparisons).

The row geometry is not stored in the ASCII files. The likelihood rebuilds
it from the .dataset keys (survey.survey_rows), as lsst_y1 does from n_theta,
theta_min_arcmin and theta_max_arcmin. Before writing anything this script
checks that the FITS rows equal those rebuilt rows: the same row kinds and
source bins in the same order, the same xi bin edges and the same Map3
filter radii. A FITS file with a different layout is refused.

The published analysis also compresses the 227 cut xi rows to 16 MOPED
modes. Its matrix is a plain text file of the same upstream commit
(data/upstream_y3_observed/data/moped/moped-moped-compress-2pt.txt, one
row per kept xi point and one column per mode). The last step writes the
case 1 compression from it: the MOPED modes for the xi rows and the
identity for the Map3 rows, with the row labels the likelihood checks.

Run once from projects/des_3pcf (a few seconds):

    python scripts/convert_desy3_fits.py

Rerun it only when an upstream FITS file changes.
"""

import configparser
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "likelihood"))
sys.path.insert(0, str(PROJECT.parents[1] / "external_modules/code/cosmolike_core"))
from survey import ARCMIN_TO_RADIAN, survey_rows  # noqa: E402
from compression import LinearCompression, load_compression  # noqa: E402
from cosmolike_notebook_utils.fisher import symmetric_matrix  # noqa: E402

DATA = PROJECT / "data"
SIM = DATA / "upstream_y3/data"
OBSERVED = DATA / "upstream_y3_observed/data"

# The DES Y3 geometry, written in every .dataset file (lsst_y1 key names).
SOURCE_NTOMO = 4
N_THETA = 20
THETA_MIN_ARCMIN = 2.5
THETA_MAX_ARCMIN = 250.0
MAP3_RADII_ARCMIN = (7.0, 14.0, 25.0, 40.0)
NZ_SOURCE_BIN_WIDTH = 0.01


# ---------------------------------------------------------------------------
# FITS readers (formerly in likelihood/survey.py)
# ---------------------------------------------------------------------------
def _columns(table, names, extension):
    """Raise ValueError if the table lacks one of the named columns."""
    for name in names:
        if name not in table.names:
            raise ValueError(f"{extension} lacks {name}; supply a labeled measurement table.")


def _angle_column(hdu, name, assumed_unit="arcmin"):
    """Return an angular column in radians, after checking its FITS unit."""
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
    """Return one-based FITS source labels as checked zero-based integers."""
    values = np.asarray(table[name])
    if (not np.all(np.isfinite(values)) or np.any(values != np.floor(values))
            or np.any(values < 1) or np.any(values > nbin)):
        raise ValueError(f"{name} must contain integer source labels from 1 to {nbin}.")
    return np.asarray(values - 1, dtype=np.int32)


def read_measurement(two_point_path, aperture_path, nbin):
    """Read the xi+, xi- and Map3 rows of two FITS files, in joint order.

    Arguments:
      two_point_path = FITS file with xip and xim extensions.
      aperture_path = FITS file with a map3 extension.
      nbin = number of source bins.
    Returns:
      kind[n], bins[n,3] (zero-based; third is -1 for xi), angles[n,3]
      (radians: xi lower/upper/centre, Map3 three radii) and values[n].
    """
    kinds, bins, angles, values = [], [], [], []
    for path, extensions in ((two_point_path, ("xip", "xim")), (aperture_path, ("map3",))):
        with fits.open(path, memmap=False) as hdus:
            for name in extensions:
                hdu = hdus[name]
                table = hdu.data
                if name == "map3":
                    columns = ("THETA1", "THETA2", "THETA3")
                    _columns(table, ["BIN1", "BIN2", "BIN3", "VALUE", *columns], name)
                else:
                    columns = ("ANGLEMIN", "ANGLEMAX", "ANG")
                    _columns(table, ["BIN1", "BIN2", "VALUE", *columns], name)
                block = np.full((len(table), 3), fill_value=-1, dtype=np.int32)
                block[:, 0] = _bin_column(table, "BIN1", nbin)
                block[:, 1] = _bin_column(table, "BIN2", nbin)
                if name == "map3":
                    block[:, 2] = _bin_column(table, "BIN3", nbin)
                angle = np.column_stack([_angle_column(hdu, column) for column in columns])
                value = np.asarray(table["VALUE"], dtype=np.float64)
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"{name}.VALUE must be finite.")
                kinds.append(np.full(len(table), name, dtype="U4"))
                bins.append(block)
                angles.append(angle)
                values.append(value.copy())
    return (np.concatenate(kinds), np.concatenate(bins), np.concatenate(angles),
            np.concatenate(values))


def read_covariance(path, kind):
    """Read COVMAT after checking its block labels against the row kinds.

    The FITS header names each block (NAME_i) and its first row (STRT_i).
    The matrix shape alone cannot tell (xi, map3) from (map3, xi) order, so
    the labels are required.
    """
    expected = []
    for index, value in enumerate(kind):
        if not expected or expected[-1][0] != value:
            expected.append((str(value), index))
    with fits.open(path, memmap=False) as hdus:
        image = hdus["COVMAT"]
        for index, (name, start) in enumerate(expected):
            if (str(image.header.get(f"NAME_{index}", "")).lower() != name
                    or image.header.get(f"STRT_{index}") != start):
                raise ValueError(f"Covariance block {index} must be {name} starting at {start}.")
        if f"NAME_{len(expected)}" in image.header:
            raise ValueError("Covariance contains extra blocks absent from the rows.")
        matrix = symmetric_matrix(image.data, name="Joint covariance", positive=True)
    if matrix.shape != (kind.size, kind.size):
        raise ValueError("Joint covariance size differs from the row count.")
    return matrix


def read_sources(path, nbin, extension="nz_source"):
    """Read the source n(z): Z_MID, Z_LOW, Z_HIGH and BIN1..BINnbin."""
    with fits.open(path, memmap=False) as hdus:
        table = hdus[extension].data
        names = ["Z_MID", "Z_LOW", "Z_HIGH"] + [f"BIN{i + 1}" for i in range(nbin)]
        _columns(table, names, extension)
        z = np.array(table["Z_MID"], dtype=np.float64)
        low = np.array(table["Z_LOW"], dtype=np.float64)
        high = np.array(table["Z_HIGH"], dtype=np.float64)
        sources = np.array([table[f"BIN{i + 1}"] for i in range(nbin)], dtype=np.float64)
    return z, low, high, sources


# ---------------------------------------------------------------------------
# Checks and writers
# ---------------------------------------------------------------------------
def check_geometry(kind, bins, angles):
    """Refuse FITS rows that differ from the rows rebuilt from the keys."""
    edges = np.geomspace(THETA_MIN_ARCMIN, THETA_MAX_ARCMIN, N_THETA + 1) * ARCMIN_TO_RADIAN
    radii = np.asarray(MAP3_RADII_ARCMIN) * ARCMIN_TO_RADIAN
    rows = survey_rows(nbin=SOURCE_NTOMO, aperture_radii=radii, xi_edges=edges)
    xi = kind != "map3"
    if not (np.array_equal(rows.kind, kind) and np.array_equal(rows.bins, bins)):
        raise ValueError("FITS row kinds or source bins differ from the dataset geometry.")
    if not np.allclose(rows.angles[xi, :2], angles[xi, :2], rtol=1.0e-14, atol=0.0):
        raise ValueError("FITS xi bin edges differ from n_theta/theta_min/theta_max.")
    if not np.allclose(rows.angles[~xi], angles[~xi], rtol=1.0e-14, atol=0.0):
        raise ValueError("FITS Map3 radii differ from map3_radii_arcmin.")


def write_vector(path, values):
    """Write "index value" lines; repr keeps every bit of each float."""
    path.write_text("".join(f"{i} {float(v)!r}\n" for i, v in enumerate(values)))
    print("wrote", path)


def write_covariance(path, matrix):
    """Write "i j value" for the upper triangle (i <= j), in row order."""
    lines = []
    for i in range(matrix.shape[0]):
        for j in range(i, matrix.shape[0]):
            lines.append(f"{i} {j} {float(matrix[i, j])!r}\n")
    path.write_text("".join(lines))
    print("wrote", path)


def scale_cut_mask(kind, bins, angles):
    """Flag the xi rows kept by the DES Y3 cosmic-shear cuts; keep all Map3.

    data/upstream_cosmosis/des-y3-scale-cuts.ini gives one angle range per
    (correlation, bin pair); CosmoSIS keeps a row whose ANG lies inside it.
    """
    cuts = configparser.ConfigParser()
    cuts.read(DATA / "upstream_cosmosis/des-y3-scale-cuts.ini")
    ranges = cuts["2pt_like"]
    mask = np.ones(kind.size, dtype=int)
    for row in np.flatnonzero(kind != "map3"):
        key = f"angle_range_{kind[row]}_{bins[row, 0] + 1}_{bins[row, 1] + 1}"
        minimum, maximum = map(float, ranges[key].split())
        ang = angles[row, 2] / ARCMIN_TO_RADIAN
        mask[row] = int(minimum <= ang <= maximum)
    return mask


def write_moped_compression(mask):
    """Write data/des_y3_moped_compression.npz from the published MOPED file.

    The published joint likelihood (python/joint_likelihood.py of the
    upstream repository) applies block_diag(T, I) to the selected rows:
    T = the 16 MOPED modes of the 227 kept xi rows, and the identity for
    the 80 Map3 rows. The MOPED file holds T transposed, [227, 16]. It was
    trained on the analytic DES Y3 cosmic-shear covariance, not on the 796
    simulations of the joint covariance, so the finite-simulation
    likelihood may treat T as fixed (covariance_training_independent).

    Arguments:
      mask = the 0/1 scale-cut mask of the 480 rows (case 1's mask_file).
    Raises:
      ValueError when the file shape differs from the cut rows, or when an
      existing output differs from the one rebuilt here.
    """
    source = OBSERVED / "moped/moped-moped-compress-2pt.txt"
    modes = np.loadtxt(source).T
    edges = np.geomspace(THETA_MIN_ARCMIN, THETA_MAX_ARCMIN, N_THETA + 1) * ARCMIN_TO_RADIAN
    radii = np.asarray(MAP3_RADII_ARCMIN) * ARCMIN_TO_RADIAN
    rows = survey_rows(nbin=SOURCE_NTOMO, aperture_radii=radii, xi_edges=edges)
    selected = np.flatnonzero(mask)
    xi = selected[rows.kind[selected] != "map3"]
    aperture = selected[rows.kind[selected] == "map3"]
    if modes.shape != (16, xi.size) or not np.array_equal(selected, np.concatenate((xi, aperture))):
        raise ValueError("The MOPED file does not match the 227 cut xi rows.")

    # block_diag(T, I): MOPED modes first, then every Map3 row unchanged.
    matrix = np.zeros((modes.shape[0] + aperture.size, selected.size))
    matrix[:modes.shape[0], :xi.size] = modes
    matrix[modes.shape[0]:, xi.size:] = np.eye(aperture.size)
    names = ["S_8", "omega_m", "h0", "omega_b", "n_s", "mnu", "m1", "m2", "m3", "m4",
             "bias_1", "bias_2", "bias_3", "bias_4", "a1", "alpha1"]
    compression = LinearCompression(
        matrix=matrix,
        input_labels=rows.labels[selected],
        output_labels=np.concatenate(([f"moped_{name}" for name in names],
                                      rows.labels[aperture])),
        metadata={
            "source": "github.com/rchgomes/3pcf_integrator",
            "commit": "d6098d326dcb03bc750adcc9627119f52d928944",
            "file": "data/moped/moped-moped-compress-2pt.txt",
            "git_blob_sha1": "ab60986ba020ff3a2e0d2e9202613a4cdfbab734",
            "layout": "block_diag(MOPED modes of the cut xi rows, identity on Map3)",
            "training_covariance": "analytic DES Y3 cosmic-shear covariance "
                                   "(2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate.v2.fits)",
            "covariance_training_independent": True,
            "mode_parameters": names,
        },
    )
    destination = DATA / "des_y3_moped_compression.npz"
    if destination.exists():
        saved = load_compression(destination)
        if not (np.array_equal(saved.matrix, compression.matrix)
                and np.array_equal(saved.input_labels, compression.input_labels)
                and np.array_equal(saved.output_labels, compression.output_labels)
                and saved.metadata == compression.metadata):
            raise ValueError(f"{destination} differs from the rebuilt compression; "
                             "remove it only if the upstream MOPED file changed.")
        print("unchanged", destination)
        return
    compression.save(destination)
    print("wrote", destination)


def main():
    """Convert every upstream file the datasets use; refuse any mismatch."""
    # Step 1: the observed and the simulated measurements.
    sets = {
        "real": (OBSERVED / "dv/real_2pt_desy3_unblind_g2flip_matchElisa_2025Mar10.fits",
                 OBSERVED / "dv/real_map3_desy3_unblind_scalecut3pcf_796simcov_2025Mar11.fits",
                 OBSERVED / "covariance/xip_xim_map3_covariance_8ARCMINCUT_25Feb25.fits"),
        "sim": (SIM / "dv/sim_2pt-NLA-cosmoCosmogrid-04Nov24.fits",
                SIM / "dv/sim_map3_emu-NLA-cosmoCosmogrid-24FEBRUARY2025_WITH_COV.fits",
                SIM / "covariance/xip_xim_map3_covariance_8ARCMINCUT_25Feb25.fits"),
    }
    covariances = {}
    rows = {}
    for label, (two_point, aperture, covariance) in sets.items():
        kind, bins, angles, values = read_measurement(two_point, aperture, SOURCE_NTOMO)
        check_geometry(kind, bins, angles)
        covariances[label] = read_covariance(covariance, kind)
        rows[label] = (kind, bins, angles)
        write_vector(DATA / f"des_y3_{label}.datavector", values)

    # Step 2: one covariance file; both releases carry the same matrix.
    if not np.array_equal(covariances["real"], covariances["sim"]):
        raise ValueError("The observed and simulated covariances differ.")
    write_covariance(DATA / "des_y3_joint.cov", covariances["sim"])

    # Step 3: the masks (DES Y3 xi cuts; xi rows only, for power-only feedback).
    # Both releases share the row geometry (checked above); the cuts read
    # the observed rows' quoted xi centres (ANG), identical in both files.
    kind, bins, angles = rows["real"]
    write_vector(DATA / "des_y3_xi_scale_cut.mask", scale_cut_mask(kind, bins, angles))
    write_vector(DATA / "des_y3_xi_only.mask", (kind != "map3").astype(int))

    # Step 4: the processed source n(z) (demonstration/y3_demo_sources.fits,
    # made from the upstream nz_source table), "z n1 ... n4". Every row is a
    # histogram cell of width NZ_SOURCE_BIN_WIDTH centred on z.
    z, low, high, sources = read_sources(DATA / "demonstration/y3_demo_sources.fits", SOURCE_NTOMO)
    half = 0.5 * NZ_SOURCE_BIN_WIDTH
    if not (np.allclose(low, z - half, rtol=0.0, atol=1.0e-15)
            and np.allclose(high, z + half, rtol=0.0, atol=1.0e-15)):
        raise ValueError("Source cells are not z -+ nz_source_bin_width / 2.")
    lines = []
    for index in range(z.size):
        lines.append(" ".join(repr(float(v)) for v in (z[index], *sources[:, index])) + "\n")
    (DATA / "des_y3_source.nz").write_text("".join(lines))
    print("wrote data/des_y3_source.nz")

    # Step 5: the case 1 compression (published MOPED on the cut xi rows).
    write_moped_compression(scale_cut_mask(kind, bins, angles))


if __name__ == "__main__":
    main()
