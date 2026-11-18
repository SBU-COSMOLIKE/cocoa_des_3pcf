"""Check CAMB settings and saved identities without running CAMB or native C.

The controlled arrays only supply a schema-valid reference. Provider creation,
configuration validation and rejected setting changes perform no prediction.
"""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from des3pcf.pipeline import PhysicalProvider
from des3pcf.scaling_threepcf import _configuration, preflight_threepcf
from test_pipeline import toy_tables


PROJECT = Path(__file__).resolve().parents[1]


def make_provider(kmax_boltzmann=5.0, extrap_kmax=1000.0, mode="camb"):
    """Construct a provider with declared reference provenance and no C interface.

    A native call would fail because compiled=None. The tests exercise only
    identity comparisons and refusals that must precede any native calculation.
    """
    tables, parameters = toy_tables()
    metadata = {
        "parameters": dict(parameters),
        "halofit_version": "takahashi",
        "kmax_boltzmann": 5.0,
        "extrap_kmax": 1000.0,
    }
    return PhysicalProvider(
        compiled=None, reference_tables=tables, reference_metadata=metadata,
        reference_parameters=parameters, cosmology_mode=mode,
        kmax_boltzmann=kmax_boltzmann, extrap_kmax=extrap_kmax,
    )


@pytest.mark.parametrize("name", ["kmax_boltzmann", "extrap_kmax"])
def test_provider_refuses_mismatched_reference_limits(name):
    """A changed limit cannot silently reuse the saved reference power table."""
    settings = {name: 6.0 if name == "kmax_boltzmann" else 1100.0}
    with pytest.raises(ValueError, match="limits differ from table provenance"):
        make_provider(**settings)


@pytest.mark.parametrize("name", ["kmax_boltzmann", "extrap_kmax"])
def test_mutable_limits_change_exact_keys_and_fixed_tables_refuse_them(name):
    """Changing and restoring a limit changes and restores its exact cache key."""
    provider = make_provider(mode="fixed_tables")
    original = provider.numerical_key()
    value = getattr(provider, name)
    setattr(provider, name, 1.1 * value)
    assert provider.numerical_key() != original
    with pytest.raises(ValueError, match="select camb or supply new tables"):
        provider.get(parameters={})
    assert provider.model is None
    setattr(provider, name, value)
    assert provider.numerical_key() == original
    setattr(provider, name, True)
    with pytest.raises(ValueError, match=name):
        provider.numerical_key()


def test_camb_reference_requires_limit_provenance_but_fixed_arrays_do_not():
    """An older table can stay fixed, but cannot claim the bounded-CAMB settings."""
    tables, parameters = toy_tables()
    for mode in ("fixed_tables", "camb"):
        arguments = {
            "compiled": None,
            "reference_tables": tables,
            "reference_metadata": {},
            "reference_parameters": parameters,
            "cosmology_mode": mode,
        }
        if mode == "camb":
            with pytest.raises(ValueError, match="lack .* provenance"):
                PhysicalProvider(**arguments)
        else:
            provider = PhysicalProvider(**arguments)
            assert provider.model is None
            # k[-1]=1000 h/Mpc and h=0.7 require a 700/Mpc query;
            # the default extension is 1.01 times that endpoint.
            assert provider.reference_producer_settings[1:] == (5.0, 707.0)


def test_cli_preserves_explicit_camb_limits_and_rejects_inadequate_tail(tmp_path):
    """Both limits pass through YAML; coverage is checked over the full draw bank."""
    path = PROJECT / "configs" / "des_y3_nla_scaling.yaml"
    _, config, _, _ = _configuration(path=path)
    assert config["physics"]["kmax_boltzmann"] == 5.0
    assert config["physics"]["extrap_kmax"] == 1000.0
    changed = deepcopy(yaml.safe_load(path.read_text()))
    # 750/Mpc covers central h=0.722, but not a permitted h=0.8 draw.
    changed["physics"]["extrap_kmax"] = 750.0
    filename = tmp_path / "short_tail.yaml"
    filename.write_text(yaml.safe_dump(changed, sort_keys=False))
    report = preflight_threepcf(config_path=filename)
    assert any(
        message.startswith("CAMB interpolation range:")
        for message in report["blockers"]
    )
    changed["physics"]["extrap_kmax"] = 700.0
    filename.write_text(yaml.safe_dump(changed, sort_keys=False))
    with pytest.raises(ValueError, match="cover h"):
        _configuration(path=filename)
