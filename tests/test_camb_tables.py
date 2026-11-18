"""Check bounded CAMB preparation against a fresh final-amplitude solve.

The production helper solves transfers to 5/Mpc and extrapolates their
power to the requested table edge. These checks use four redshifts and
eight wavenumbers, including 1000 h/Mpc, without solving any Boltzmann
mode to that high wavenumber. A separate CAMB result starts directly at
the final primordial amplitude; it does not reuse the helper's results.
Its interpolators return h units directly, independently checking the
production helper's physical-unit queries and explicit powers of h.
"""

import numpy as np
import pytest

from des3pcf.inputs import _camb_power_limits, camb_tables


@pytest.fixture(scope="module")
def prepared_and_fresh():
    """Return the helper output and an independent matter-only CAMB solve.

    Both calculations use the same background, transfer settings and
    final As. Nonzero neutrino mass makes total-matter and cold-matter
    power distinct, so exchanging their output tables cannot pass.
    """
    import camb

    parameters = {
        "omega_m": 0.31,
        "omega_b": 0.049,
        "h": 0.67,
        "ns": 0.963,
        "sigma8": 0.76,
        "w0": -0.9,
        "wa": 0.15,
        "mnu": 0.12,
        "num_massive_neutrinos": 1,
        "nnu": 3.046,
    }
    k = np.array([
        0.001,
        0.03,
        0.2,
        1.0,
        5.0,
        50.0,
        100.0,
        1000.0,
    ])
    z = np.array([0.0, 0.4, 0.9, 1.7])
    sources = np.array([[0.0, 0.7, 1.0, 0.2]])
    los_z = np.array([0.1, 0.35, 0.8, 1.5])
    tables, metadata = camb_tables(
        parameters=parameters,
        k=k,
        z=z,
        sources=sources,
        los_z=los_z,
        kmax_boltzmann=5.0,
        extrap_kmax=800.0,
    )

    # Build a new parameter object at the final As. Supplying this As
    # from the returned metadata isolates transfer reuse from amplitude
    # calibration; a separate assertion checks the requested sigma8.
    h = parameters["h"]
    settings = camb.CAMBparams()
    settings.set_cosmology(
        H0=100.0*h,
        ombh2=parameters["omega_b"]*h**2,
        omch2=(parameters["omega_m"] - parameters["omega_b"])*h**2,
        mnu=parameters["mnu"],
        omk=0.0,
        num_massive_neutrinos=1,
        nnu=3.046,
    )
    settings.omch2 -= settings.omnuh2
    settings.set_dark_energy(
        w=parameters["w0"],
        wa=parameters["wa"],
        dark_energy_model="ppf",
    )
    settings.InitPower.set_params(As=metadata["As"], ns=parameters["ns"])
    settings.WantCls = False
    settings.set_matter_power(redshifts=z[::-1], kmax=5.0, silent=True)
    settings.NonLinearModel.set_params(halofit_version="takahashi")
    settings.NonLinear = camb.model.NonLinear_both
    fresh = camb.get_results(params=settings)
    return parameters, tables, metadata, fresh


def test_transfer_reuse_reaches_sigma8_and_preserves_background(
    prepared_and_fresh,
):
    """Rescaling As must reach the requested sigma8 without a second solve."""
    parameters, tables, metadata, fresh = prepared_and_fresh
    sigma = np.asarray(fresh.get_sigma8())[::-1]
    np.testing.assert_allclose(
        tables["sigma8"], parameters["sigma8"], rtol=2.0e-12, atol=0.0,
    )
    np.testing.assert_allclose(
        tables["sigma8"], sigma[0], rtol=2.0e-12, atol=0.0,
    )
    np.testing.assert_allclose(
        tables["growth"], sigma/sigma[0], rtol=2.0e-12, atol=0.0,
    )
    h = parameters["h"]
    z = tables["z"]
    np.testing.assert_allclose(
        tables["chi"], fresh.comoving_radial_distance(z=z)*h,
        rtol=2.0e-12, atol=1.0e-12,
    )
    np.testing.assert_allclose(
        tables["dz_dchi"], fresh.hubble_parameter(z=z)/(299792.458*h),
        rtol=2.0e-12, atol=0.0,
    )
    assert metadata["kmax_boltzmann"] == 5.0
    assert metadata["extrap_kmax"] == 800.0
    assert metadata["camb_wavenumber_limit_units"] == "1/Mpc"
    assert metadata["amplitude_method"] == "reuse_matter_transfers"


def test_reused_power_matches_fresh_solve_in_native_h_units(
    prepared_and_fresh,
):
    """Compare linear, nonlinear and cb spectra through the opposite unit API.

    With k_hunit=True, CAMB's extrapolation boundary is in h/Mpc, so
    800/Mpc becomes 800/h here. No conversion of the returned power is
    necessary. Agreement therefore checks both k*h and P*h^3 in the
    production reader, including the remote extrapolated tail.
    """
    parameters, tables, metadata, fresh = prepared_and_fresh
    h = parameters["h"]
    cases = (
        ("p_linear", False, "delta_tot"),
        ("p_nonlinear", True, "delta_tot"),
        ("p_linear_cb", False, "delta_nonu"),
    )
    for name, nonlinear, variable in cases:
        reader = fresh.get_matter_power_interpolator(
            nonlinear=nonlinear,
            var1=variable,
            var2=variable,
            hubble_units=True,
            k_hunit=True,
            extrap_kmax=metadata["extrap_kmax"]/h,
        )
        expected = reader.P(z=tables["z"], kh=tables["k"])
        np.testing.assert_allclose(
            tables[name], expected, rtol=2.0e-10, atol=0.0,
            err_msg=f"Fresh final-As comparison for {name}",
        )
    np.testing.assert_array_equal(
        tables["p_linear0"], tables["p_linear"][0],
    )
    # These reach checks ensure the fixture distinguishes nonlinear from
    # linear power and cold matter from matter including neutrinos.
    assert np.max(tables["p_nonlinear"]/tables["p_linear"]) > 2.0
    cold_difference = np.abs(tables["p_linear_cb"]/tables["p_linear"] - 1.0)
    assert np.max(cold_difference) > 0.001
    assert tables["k"][-1]*h > 100.0*metadata["kmax_boltzmann"]


def test_tail_domain_is_separate_from_transfer_cutoff():
    """Extending output k leaves the solved transfer cutoff unchanged."""
    small = _camb_power_limits(
        k=np.array([0.01, 1.0]), h=0.67,
        kmax_boltzmann=5.0, extrap_kmax=None,
    )
    remote = _camb_power_limits(
        k=np.array([0.01, 1000.0]), h=0.67,
        kmax_boltzmann=5.0, extrap_kmax=None,
    )
    assert small == (5.0, 250.0)
    assert remote[0] == 5.0
    assert remote[1] >= 670.0
    explicit = _camb_power_limits(
        k=np.array([0.01, 1000.0]), h=0.67,
        kmax_boltzmann=7.5, extrap_kmax=800.0,
    )
    assert explicit == (7.5, 800.0)


@pytest.mark.parametrize("cutoff", [0.0, -1.0, np.inf, np.nan, True])
def test_invalid_transfer_cutoff_is_refused(cutoff):
    """Invalid solved-domain requests fail before any CAMB work."""
    with pytest.raises(ValueError, match="kmax_boltzmann"):
        _camb_power_limits(
            k=np.array([0.01, 1000.0]), h=0.67,
            kmax_boltzmann=cutoff, extrap_kmax=800.0,
        )


@pytest.mark.parametrize("endpoint", [250.0, 669.0, np.inf, np.nan, True])
def test_uncovered_or_invalid_tail_is_refused(endpoint):
    """A nominal LSST 250/Mpc tail cannot cover a 670/Mpc query."""
    with pytest.raises(ValueError, match="extrap_kmax must cover"):
        _camb_power_limits(
            k=np.array([0.01, 1000.0]), h=0.67,
            kmax_boltzmann=5.0, extrap_kmax=endpoint,
        )
