"""Check the general observables of the notebook wrappers.

interface/cosmolike_des_3pcf_notebook_wrappers.py computes xi, Map3, the
joint data vector and chi2 at any parameter point through the Cobaya model
of a numbered example. These checks run EXAMPLE_EVALUATE3 (NLA) three
times: the arrays of xi and map3 must hold exactly the values of the
joint vector, a change of A_s must reach every Map3 entry, chi2 must
be the compiled joint chi2 of that vector, and a value for a parameter
the yaml fixes, or a parameter name the model does not know, must be
refused rather than ignored.
"""

import numpy as np
import pytest

import cosmolike_des_3pcf_interface as ci
import cosmolike_des_3pcf_notebook_wrappers as nw


def test_observables_hold_the_joint_vector_and_follow_parameters():
    """xi and map3 rearrange the data vector; A_s changes Map3; chi2 matches."""
    nw.configure(yaml="EXAMPLE_EVALUATE3.yaml", threads=2)

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

    # A different amplitude 10^9 A_s reaches every Map3 entry.
    _, _, changed = nw.map3(As_1e9=2.0)
    assert np.all(changed != map3)

    # chi2 is the compiled joint chi2 of the same vector.
    chi2 = nw.get_chi2()
    assert chi2 == float(ci.compute_joint_chi2(prediction=values))

    # EXAMPLE_EVALUATE3 fixes DES_A2_1 = 0 (NLA); another value cannot be
    # honoured.
    with pytest.raises(ValueError, match="fixes it"):
        nw.datavector(A2=[0.5, 0.0])

    # A mistyped parameter is refused before any evaluation, rather than
    # returning the fiducial value as if the change had been applied.
    with pytest.raises(ValueError, match="has no parameter"):
        nw.get_chi2(As_1e8=2.0)


def test_defaults_follow_the_configured_yaml():
    """A parameter left out takes the configured yaml's fiducial value.

    EXAMPLE_EVALUATE4 (TATT) samples DES_A2_1, DES_A2_2 and DES_BTA_1 at
    nonzero fiducial values; case 3 fixes them to zero. A default call
    after configuring case 3 must equal a call at case 3's own fiducial,
    and differ from a call at the case-3 values.
    """
    from cobaya.yaml import yaml_load_file

    nw.configure(yaml="EXAMPLE_EVALUATE4.yaml", threads=2)
    override = yaml_load_file(str(nw.PROJECT / "EXAMPLE_EVALUATE4.yaml"))["sampler"]["evaluate"]["override"]
    _, values = nw.datavector()
    _, explicit = nw.datavector(A2=[override["DES_A2_1"], override["DES_A2_2"]],
                                BTA=override["DES_BTA_1"])
    np.testing.assert_array_equal(values, explicit)
    _, nla = nw.datavector(A2=[0.0, 0.0], BTA=0.0)
    assert np.any(values != nla)
