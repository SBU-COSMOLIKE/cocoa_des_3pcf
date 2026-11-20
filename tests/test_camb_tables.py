"""Check bounded CAMB preparation against a fresh final-amplitude solve.

The production helper solves transfers to 5/Mpc and extrapolates their
power to the requested table edge. These checks use four redshifts and
eight wavenumbers, including 1000 h/Mpc, without solving any Boltzmann
mode to that high wavenumber. A separate CAMB result starts directly at
the final primordial amplitude; it does not reuse the helper's results.
Its interpolators return h units directly, independently checking the
production helper's physical-unit queries and explicit powers of h.
"""

from test_pipeline import compact_bispectrum_grid

import numpy as np
import pytest

from des3pcf.physical_provider import _camb_power_limits, camb_tables, camb_accuracy_settings


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
    log10k = np.linspace(start=-3.0, stop=3.0, num=8)
    k = 10.0**log10k
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
    assert metadata["camb_accuracy_boost"] == 1.0
    assert metadata["camb_k_per_logint"] == 0


@pytest.mark.parametrize("name,value", [
    ("accuracy_boost", True), ("accuracy_boost", 0.5),
    ("accuracy_boost", np.nan), ("accuracy_boost", np.inf),
    ("accuracy_boost", "2"), ("accuracy_boost", 2.0j),
    ("k_per_logint", True), ("k_per_logint", -1),
    ("k_per_logint", 1.5), ("k_per_logint", np.inf),
    ("k_per_logint", 2**31),
    ("k_per_logint", "50"),
])
def test_invalid_camb_precision_is_refused_before_calculation(name, value):
    """Precision controls must have the stated scalar domains."""
    with pytest.raises(ValueError, match="camb_"):
        camb_accuracy_settings(**{name: value})


@pytest.mark.parametrize("name,value", [
    ("camb_accuracy_boost", 2.0), ("camb_k_per_logint", 50),
])
def test_real_camb_accuracy_rebuild_and_restore(
    compiled, prepared_and_fresh, name, value,
):
    """A notebook precision change rebuilds power; A/B/A restores it exactly."""
    from des3pcf.physical_provider import PhysicalProvider

    parameters, tables, metadata, unused = prepared_and_fresh
    provider = PhysicalProvider(
        bispectrum_table=compact_bispectrum_grid(zmax=tables["z"][-1]),
        compiled=compiled, reference_tables=tables,
        reference_metadata=metadata, reference_parameters=parameters,
        cosmology_mode="camb", kernel_nz=21, extrap_kmax=800.0,
    )
    initial, original, unused = provider.get(parameters=parameters)
    saved = original["p_linear0"].copy()
    baseline = getattr(provider, name)
    setattr(provider, name, value)
    changed, displaced, record = provider.get(parameters=parameters)
    assert changed is not initial
    assert not np.array_equal(displaced["p_linear0"], saved)
    assert provider.cosmology_metadata[name] == value
    repeated, unused, unused_record = provider.get(parameters=parameters)
    assert repeated is changed
    np.testing.assert_array_equal(original["p_linear0"], saved)
    setattr(provider, name, baseline)
    restored, restored_tables, unused = provider.get(parameters=parameters)
    np.testing.assert_array_equal(restored_tables["p_linear0"], saved)
    assert provider.cosmology_metadata[name] == baseline


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
        ("p_linear_nu", False, "delta_nu"),
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


def test_cold_sigma8_and_neutrino_archive_use_the_same_cosmology(
    prepared_and_fresh, tmp_path,
):
    """Compare sigma8_cb using physical radii and preserve the neutrino spectrum.

    The production call uses 8 Mpc/h. This reference supplies 8/h Mpc
    through the opposite radius-unit convention of a fresh CAMB solution.
    The saved neutrino field must survive NPZ serialization unchanged.
    """
    from des3pcf.physical_provider import load_tables, save_tables

    parameters, tables, metadata, fresh = prepared_and_fresh
    expected = fresh.get_sigmaR(
        R=8. / parameters["h"],
        z_indices=-1,
        var1="delta_nonu",
        var2="delta_nonu",
        hubble_units=False,
    )
    np.testing.assert_allclose(
        actual=metadata["sigma8_cb"], desired=np.asarray(expected).item(),
        rtol=2.e-12, atol=0.,
    )
    assert metadata["sigma8_cb"] > tables["sigma8"]
    path = tmp_path / "physical.npz"
    save_tables(path=path, tables=tables, metadata=metadata)
    restored, record = load_tables(path=path)
    np.testing.assert_array_equal(x=restored["p_linear_nu"], y=tables["p_linear_nu"])
    assert record["sigma8_cb"] == metadata["sigma8_cb"]


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


def test_matter_fraction_uses_exact_massive_neutrino_background(prepared_and_fresh):
    """The saved fraction comes from the same CAMB background as the powers."""
    parameters, tables, metadata, fresh = prepared_and_fresh
    redshift = tables["z"]
    expected = fresh.get_Omega(var="cdm", z=redshift)
    expected += fresh.get_Omega(var="baryon", z=redshift)
    expected += fresh.get_Omega(var="nu", z=redshift)
    np.testing.assert_allclose(
        actual=tables["omega_m_z"], desired=expected, rtol=2.0e-12, atol=0.0,
    )
