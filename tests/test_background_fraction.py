"""Check which expansion history the matter fit and halo covariance consume.

Controlled backgrounds separate the pressureless H(z) identity from an exact
provider table. They are component tests, not survey cosmologies. Old model
owners must retain their fractions after inputs or core state change.
"""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from des3pcf.variance import ReferenceVariance
from data_vector.test_physics_model import physical_inputs
from test_supplied_baseline import install_bridge


@pytest.mark.parametrize("source_readers", [False, True])
@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_expansion_history_sets_pressureless_fraction(compiled, source_readers, adapter):
    """E(z)=1+z gives the exactly linear fraction Omega_m0*(1+z)."""
    inputs = physical_inputs()
    inputs["source_readers"] = source_readers
    redshift = inputs["z"]
    inputs["chi"] = 2997.92458 * np.log1p(redshift)
    inputs["dz_dchi"] = (1.0 + redshift) / 2997.92458
    fit, unused, record = ReferenceVariance().prepare(tables=inputs)
    binding = getattr(compiled, adapter)
    model = binding.PhysicsModel(**inputs, variance=dict(fit))
    query = np.linspace(start=0.0, stop=2.0, num=57)
    expected = inputs["omega_m"] * (1.0 + query)
    np.testing.assert_allclose(
        actual=model.background_matter_fraction(redshifts=query),
        desired=expected, rtol=2.0e-14, atol=0.0,
    )
    # The copied expansion history already specifies this background. Changing
    # unused CPL scalars must not introduce a second expansion model.
    inputs["w0"] = -0.7
    inputs["wa"] = 0.4
    other = binding.PhysicsModel(**inputs, variance=dict(fit))
    np.testing.assert_array_equal(
        x=model.background_matter_fraction(redshifts=query),
        y=other.background_matter_fraction(redshifts=query),
    )


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_exact_fraction_is_owned_and_changes_only_its_matter_coefficient(compiled, adapter):
    """An explicit provider fraction overrides H(z), including on its own axis."""
    inputs = physical_inputs()
    fit, unused, record = ReferenceVariance().prepare(tables=inputs)
    axis = np.linspace(start=0.0, stop=2.0, num=17)
    fraction = 0.3 + 0.07 * axis
    binding = getattr(compiled, adapter)
    first = binding.PhysicsModel(
        **inputs, variance=dict(fit), omega_m_z=fraction,
        omega_m_redshift=axis,
    )
    saved = fraction.copy()
    fraction += 0.2
    second = binding.PhysicsModel(
        **inputs, variance=dict(fit), omega_m_z=fraction,
        omega_m_redshift=axis,
    )
    np.testing.assert_allclose(
        actual=first.background_matter_fraction(redshifts=axis),
        desired=saved, rtol=2.0e-15, atol=0.0,
    )
    sides = np.asarray([[0.03, 0.04, 0.05], [0.1, 0.2, 0.25]])
    query = np.asarray([0.2, 1.0])
    for component in (1, 2):
        np.testing.assert_array_equal(
            x=first.matter(sides=sides, redshifts=query, component=component),
            y=second.matter(sides=sides, redshifts=query, component=component),
        )
    assert np.any(first.matter(sides=sides, redshifts=query, component=3)
                  != second.matter(sides=sides, redshifts=query, component=3))
    parameters = {"a1": 0.4, "a2": 0.7, "bta": 0.3}
    np.testing.assert_array_equal(
        x=first.ia(sides=sides, redshifts=query, parameters=parameters),
        y=second.ia(sides=sides, redshifts=query, parameters=parameters),
    )


def test_bridge_fraction_replacement_clear_and_old_owner(compiled):
    """A/B/A, identical resends and clearing preserve the declared ownership."""
    inputs = physical_inputs()
    install_bridge(compiled=compiled, inputs=inputs, install_scalars=True)
    core = compiled.core
    first_values = 0.3 + 0.07 * inputs["z"]
    core.set_matter_fraction(omega_m_z=first_values)
    first = core.model_from_core()
    first_result = first.background_matter_fraction(redshifts=inputs["z"])
    core.set_matter_fraction(omega_m_z=first_values.copy())
    assert core.model_from_core() is first
    core.set_matter_fraction(omega_m_z=first_values + 0.2)
    second = core.model_from_core()
    assert second is not first
    core.set_matter_fraction(omega_m_z=first_values)
    restored = core.model_from_core()
    np.testing.assert_array_equal(
        x=restored.background_matter_fraction(redshifts=inputs["z"]),
        y=first_result,
    )
    core.set_matter_fraction()
    derived = core.model_from_core()
    assert derived is not restored
    np.testing.assert_array_equal(
        x=first.background_matter_fraction(redshifts=inputs["z"]),
        y=first_result,
    )
    core.initial_setup()


@pytest.mark.parametrize("case", ["stale", "massive", "shape", "axis"])
def test_inconsistent_fraction_is_refused(compiled, case):
    """Reach the intended guard in a fresh process, including the neutrino case."""
    project = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([
        str(project / "interface"),
        str(project / "python"),
        str(project / "tests"),
        str(project / "tests/data_vector"),
    ])
    code = '''
import sys
import numpy as np
import cosmolike_des_3pcf_interface as compiled
from test_physics_model import physical_inputs
from test_supplied_baseline import install_bridge
from des3pcf.variance import ReferenceVariance
inputs = physical_inputs()
case = sys.argv[1]
if case in ("stale", "massive"):
    install_bridge(compiled=compiled, inputs=inputs, install_scalars=True)
    core = compiled.core
    if case == "stale":
        core.set_matter_fraction(omega_m_z=np.full(inputs["z"].size, 0.3))
    core.set_cosmological_parameters(
        omega_matter=0.31, omega_baryon=0.05, hubble=70.0,
        omega_nu_h2=0.001 if case == "massive" else 0.0,
    )
    core.model_from_core()
else:
    fit, unused, record = ReferenceVariance().prepare(tables=inputs)
    values = np.full(inputs["z"].size, 0.3)
    if case == "shape":
        values = values[:-1]
        compiled.production.PhysicsModel(**inputs, variance=dict(fit), omega_m_z=values)
    else:
        compiled.production.PhysicsModel(
            **inputs, variance=dict(fit), omega_m_z=values,
            omega_m_redshift=inputs["z"] + 0.01,
        )
'''
    result = subprocess.run(
        args=[sys.executable, "-B", "-c", code, case], env=environment,
        capture_output=True, text=True, check=False, timeout=30,
    )
    expected = {
        "stale": "supplied Omega_m(z) is stale",
        "massive": "massive neutrinos require provider Omega_m(z)",
        "shape": "Omega_m(z) must match its redshift axis",
        "axis": "Omega_m(z) must cover [0,zmax]",
    }
    assert result.returncode != 0
    assert expected[case] in result.stdout + result.stderr
    assert "runtime error:" not in result.stderr
