"""Check the general observables of the notebook wrappers.

interface/cosmolike_des_3pcf_notebook_wrappers.py computes xi, Map3, the
joint data vector and chi2 at any parameter point through the Cobaya model
of a numbered example. These checks run EXAMPLE_EVALUATE1 (NLA) three
times: the arrays of xi and map3 must hold exactly the values of the
joint vector, a change of sigma8 must reach every Map3 entry, chi2 must
be the compiled joint chi2 of that vector, and a value for a parameter
the yaml fixes, or a parameter name the model does not know, must be
refused rather than ignored.
"""

import numpy as np
import pytest

import cosmolike_des_3pcf_interface as ci
import cosmolike_des_3pcf_notebook_wrappers as nw


def test_observables_hold_the_joint_vector_and_follow_parameters():
    """xi and map3 rearrange the data vector; sigma8 changes Map3; chi2 matches."""
    nw.configure(yaml="EXAMPLE_EVALUATE1.yaml", threads=2)

    labels, values = nw.datavector()
    theta, xi_plus, xi_minus = nw.xi()
    radii, triplets, map3 = nw.map3()

    # The xi tables hold exactly the values of the xi rows (each off-diagonal
    # pair appears in both bin orders), and map3 those of the Map3 rows.
    is_xi = np.char.startswith(labels.astype(str), "xi")
    tables = np.concatenate((xi_plus[np.isfinite(xi_plus)],
                             xi_minus[np.isfinite(xi_minus)]))
    np.testing.assert_array_equal(np.unique(tables), np.unique(values[is_xi]))
    np.testing.assert_array_equal(np.unique(map3), np.unique(values[~is_xi]))
    assert map3.size == np.count_nonzero(~is_xi)
    assert theta.size > 0 and np.all(np.diff(theta) > 0.0)
    assert radii.size * triplets.shape[0] == map3.size

    # A different sigma8 reaches every Map3 entry.
    _, _, changed = nw.map3(sigma8=0.82)
    assert np.all(changed != map3)

    # chi2 is the compiled joint chi2 of the same vector.
    chi2 = nw.get_chi2()
    assert chi2 == float(ci.compute_joint_chi2(prediction=values))

    # EXAMPLE_EVALUATE1 fixes a2 = 0 (NLA); another value cannot be honoured.
    with pytest.raises(ValueError, match="fixes it"):
        nw.datavector(A2=[0.5, 0.0])

    # A mistyped parameter is refused before any evaluation, rather than
    # returning the fiducial value as if the change had been applied.
    with pytest.raises(ValueError, match="has no parameter"):
        nw.get_chi2(sigm8=0.8)
