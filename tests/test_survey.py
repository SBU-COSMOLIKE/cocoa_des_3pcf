"""Survey labels, angular units, ASCII inputs and quadrature on small arrays.

The FITS readers live in scripts/convert_desy3_fits.py, the only code that
opens the released FITS files. Its synthetic FITS file deliberately uses
unequal radii and unsorted source labels. Their association must survive
reading: sorting just the labels would change the physical statistic while
leaving its length unchanged. The ASCII files it writes must reach the
likelihood's readers (likelihood/survey.py) bit for bit.
"""

import numpy as np
import pytest
from astropy.io import fits

import sys
from pathlib import Path

from cobaya.likelihoods.des_3pcf.survey import ARCMIN_TO_RADIAN, read_covariance, read_vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import convert_desy3_fits as convert  # noqa: E402
from cosmolike_des_3pcf_interface import quadrature, shape_noise, xi_bin_weights
from cosmolike_des_3pcf_interface import aligned_fft_grid


def test_unequal_aperture_labels_and_covariance_order(tmp_path):
    """Keep each source attached to its own aperture radius through FITS I/O.

    Arguments:
      tmp_path = pytest-owned temporary output directory.
    Returns:
      nothing; checks data labels, units and the covariance metadata refusal.
    """
    columns = []
    for name, values in (
        ("BIN1", [3, 1]), ("BIN2", [1, 1]), ("BIN3", [2, 2]),
    ):
        columns.append(fits.Column(name=name, format="K", array=np.array(values)))
    for name, values in (
        ("THETA1", [3.0, 5.0]), ("THETA2", [5.0, 5.0]),
        ("THETA3", [10.0, 5.0]),
    ):
        columns.append(fits.Column(
            name=name, format="D", unit="arcmin", array=np.array(values),
        ))
    columns.append(fits.Column(
        name="VALUE", format="D", array=np.array([1.0e-9, 2.0e-9]),
    ))
    table = fits.BinTableHDU.from_columns(columns=columns, name="map3")
    covariance = fits.ImageHDU(data=np.array([[4.0, 1.0], [1.0, 9.0]]), name="COVMAT")
    covariance.header["NAME_0"] = "map3"
    covariance.header["STRT_0"] = 0
    path = tmp_path / "map3.fits"
    fits.HDUList([fits.PrimaryHDU(), table, covariance]).writeto(fileobj=path)
    with fits.open(path) as hdus:
        hdu = hdus["map3"]
        bins = np.column_stack([convert._bin_column(hdu.data, name, 3)
                                for name in ("BIN1", "BIN2", "BIN3")])
        angles = np.column_stack([convert._angle_column(hdu, name)
                                  for name in ("THETA1", "THETA2", "THETA3")])
    kind = np.array(["map3", "map3"])
    np.testing.assert_array_equal(bins[0], [2, 0, 1])
    np.testing.assert_allclose(angles[0], np.array([3.0, 5.0, 10.0]) * ARCMIN_TO_RADIAN)
    matrix = convert.read_covariance(path, kind)
    np.testing.assert_array_equal(matrix, [[4.0, 1.0], [1.0, 9.0]])
    covariance.header["NAME_0"] = "xip"
    fits.HDUList([fits.PrimaryHDU(), table, covariance]).writeto(
        fileobj=path, overwrite=True,
    )
    with pytest.raises(ValueError, match="must be map3"):
        convert.read_covariance(path, kind)


def test_ascii_inputs_round_trip_bitwise(tmp_path):
    """The converter's ASCII files reach the likelihood's readers bit for bit."""
    # Step 1: an awkward vector and a positive definite matrix.
    rng = np.random.default_rng(20261124)
    values = rng.normal(size=5) * 1.0e-7
    root = rng.normal(size=(5, 5))
    matrix = root @ root.T + 5.0 * np.eye(5)

    # Step 2: write with the converter, read with the likelihood's readers.
    convert.write_vector(tmp_path / "v.datavector", values)
    convert.write_covariance(tmp_path / "c.cov", matrix)
    np.testing.assert_array_equal(read_vector(tmp_path / "v.datavector", 5).view(np.uint64),
                                  values.view(np.uint64))
    read = read_covariance(tmp_path / "c.cov", 5)
    np.testing.assert_array_equal(read.view(np.uint64), matrix.view(np.uint64))

    # Step 3: a missing upper-triangle entry or a wrong length is refused.
    lines = (tmp_path / "c.cov").read_text().splitlines(keepends=True)
    (tmp_path / "short.cov").write_text("".join(lines[:-1]))
    with pytest.raises(ValueError, match="exactly once"):
        read_covariance(tmp_path / "short.cov", 5)
    with pytest.raises(ValueError, match="index value"):
        read_vector(tmp_path / "v.datavector", 6)


def test_log_quadrature_measure_and_shape_noise():
    """Recover known integrals in dx and the per-component white-noise formula.

    Returns:
      nothing; an extra log-Jacobian or factor of two fails these checks.
    """
    nodes, weights = quadrature(lower=0.1, upper=10.0, count=32, logarithmic=True)
    np.testing.assert_allclose(np.sum(weights / nodes), np.log(100.0), rtol=1.0e-14)
    np.testing.assert_allclose(
        np.sum(weights * nodes), 0.5 * (100.0 - 0.01), rtol=1.0e-14,
    )
    noise = shape_noise(number_density=[2.0, 4.0], sigma_component=[0.3, 0.2])
    np.testing.assert_allclose(np.diag(noise), np.array([0.09 / 2.0, 0.04 / 4.0])
                               * ARCMIN_TO_RADIAN**2, rtol=1.0e-14)
    assert noise[0, 1] == 0.0


def test_fft_grid_preserves_requested_bins_and_minimum_size():
    """Align a small radial measurement exactly even for an odd FFT request.

    Returns:
      nothing; validates both Fourier-range bounds and all selected angles.
    """
    width = np.log(150.0) / 25.0
    lower = ARCMIN_TO_RADIAN * np.exp(width * np.arange(25))
    ell, indices = aligned_fft_grid(
        theta_lower=lower, dlnt=width, ell_min=0.1, ell_max=1.0e6, minimum_size=151,
    )
    assert ell.size >= 151
    assert ell.size % 2 == 0
    assert ell[0] <= 0.1
    assert ell[-1] >= 1.0e6
    np.testing.assert_allclose(1.0 / ell[::-1][indices], lower, rtol=2.0e-12, atol=0.0)
