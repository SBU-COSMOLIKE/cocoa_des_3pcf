"""Check two shared cosmolike_notebook_utils fixes found in review.

stable_inverse must keep the constrained directions when a parameter has
no information at all (a zero row and column), and cache_record must keep
the checksum of every input file even when two inputs share a file name.
"""

import numpy as np
import pytest

from cosmolike_notebook_utils import fisher as cnu
from cosmolike_notebook_utils.caches import cache_record


def test_stable_inverse_keeps_the_constrained_direction():
    """diag(4, 0): the first parameter keeps variance 1/4, the second is null."""
    inverse, null = cnu.stable_inverse(np.diag([4.0, 0.0]), 1.0e-12)
    np.testing.assert_allclose(inverse, [[0.25, 0.0], [0.0, 0.0]], atol=1.0e-15)
    assert null.shape == (2, 1)
    np.testing.assert_allclose(np.abs(null[:, 0]), [0.0, 1.0], atol=1.0e-15)

    # A positive definite matrix is inverted as usual.
    fisher = np.array([[2.0, 0.5], [0.5, 1.0]])
    inverse, null = cnu.stable_inverse(fisher, 1.0e-12)
    np.testing.assert_allclose(inverse, np.linalg.inv(fisher), rtol=1.0e-13)
    assert null.shape == (2, 0)

    # A negative diagonal is not a Fisher matrix.
    with pytest.raises(ValueError, match="nonnegative"):
        cnu.stable_inverse(np.diag([1.0, -1.0]), 1.0e-12)

    # Neither is a positive diagonal with a negative eigenvalue (-1 and 3).
    with pytest.raises(ValueError, match="negative eigenvalue"):
        cnu.stable_inverse(np.array([[1.0, 2.0], [2.0, 1.0]]), 1.0e-12)


def test_cache_record_keeps_every_checksum():
    """Two inputs named mask.txt keep both checksums; a unique name stays short."""
    # Step 1: a unique file name keeps the short key, as in older records.
    record = cache_record(name="t", settings={"a": {"data_files": {"x.txt": "h1"}}})
    assert record["data_files"] == {"x.txt": "h1"}

    # Step 2: two datasets with the same file name keep both checksums.
    settings = {"lsst": {"data_files": {"mask.txt": "first"}},
                "desi": {"data_files": {"mask.txt": "second"}}}
    record = cache_record(name="t", settings=settings)
    assert sorted(record["data_files"].values()) == ["first", "second"]

    # Step 3: changing the first checksum changes the record, so a read
    # against the old cache refuses instead of passing.
    settings["lsst"]["data_files"]["mask.txt"] = "changed"
    assert cache_record(name="t", settings=settings)["data_files"] != record["data_files"]


def test_get_fom_refuses_a_fisher_that_is_not_positive_definite():
    """A negative eigenvalue outside the pair must stop the FoM.

    The pair block of the inverse can look positive definite while the
    full matrix has a negative eigenvalue; such a matrix is no Gaussian
    likelihood, so no FoM exists.
    """
    good = np.diag([4.0, 9.0, 1.0])
    np.testing.assert_allclose(cnu.get_FoM(i=0, j=1, F=good), 6.0, rtol=1.0e-14)
    bad = np.diag([4.0, 9.0, -1.0])
    with pytest.raises(ValueError, match="not positive definite"):
        cnu.get_FoM(i=0, j=1, F=bad)
