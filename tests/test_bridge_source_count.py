"""Check that the bridge forgets the bin count of its source table on reset.

The core bridge records how many source bins its stored n(z) table was
installed for. A notebook can install a table, rebuild the core with
initial_setup, and declare sources again without restarting the kernel.
The reset must then forget the old count, so the bridge refuses until a
matching table is installed instead of trusting a count that belongs to a
table no longer there. Bridge refusals end the process, so each case runs
in a child interpreter and the test reads its diagnostic.
"""

from pathlib import Path
import subprocess
import sys

import pytest


TESTS = Path(__file__).resolve().parent

# The child installs the small controlled inputs of test_supplied_baseline
# at two source bins and builds one model, then runs one case.
PROGRAM = """
import sys
sys.path[:0] = [{tests!r}]
import numpy as np
import cosmolike_des_3pcf_interface as compiled
from test_supplied_baseline import install_bridge, baseline_input
from data_vector.test_physics_model import physical_inputs

compiled.set_omp_threads(nthreads=1)
inputs = physical_inputs()
core = compiled.core
fraction = np.full(shape=inputs["z"].shape, fill_value=0.3)
install_bridge(compiled=compiled, inputs=inputs, install_scalars=True)
core.set_baseline_bispectrum(baseline=baseline_input(varying=False),
                             omega_m_z=fraction, sigma8=0.8)
core.model_from_core()

if "after_reset" == {case!r}:
    # Rebuild the core and reinstall everything except the source table.
    core.initial_setup()
    core.set_cosmological_parameters(omega_matter=0.3, omega_baryon=0.05,
                                     hubble=70.0, omega_nu_h2=0.0)
    wave = np.log10(inputs["k"])
    linear = inputs["growth"][:, None]**2 * inputs["p_linear0"][None, :]
    core.set_linear_power_spectrum(log10k=wave, z=inputs["z"],
                                   lnP=np.log(linear).ravel(order="F"))
    core.set_non_linear_power_spectrum(
        log10k=wave, z=inputs["z"],
        lnP=np.log(inputs["p_nonlinear"]).ravel(order="F"))
    core.set_distances(z=inputs["z"], chi=inputs["chi"])
    core.set_growth(z=inputs["z"], G=(1.0 + inputs["z"]) * inputs["growth"])
    core.set_source_sample_size(ntomo=2)
    core.set_nuisance_shear_photoz(bias=np.zeros(shape=2))
    core.set_los(z=inputs["los_z"])
    core.set_resolution(kernel_nz=101, source_readers=1, table_boost=8,
                        fit_nz=33)
    core.set_scalars(ns=0.965, sigma8=0.8, w0=-1.0, wa=0.0)
    core.set_baseline_bispectrum(baseline=baseline_input(varying=False),
                                 omega_m_z=fraction, sigma8=0.8)
else:
    # Declare a third bin without installing a matching table.
    core.set_source_sample_size(ntomo=3)

core.model_from_core()
print("model built")
"""


@pytest.mark.parametrize("case,message", [
    # After the reset the table is gone and its count must be 0, not the
    # 2 of the table installed before initial_setup.
    ("after_reset", "installed for 0 bins but 2 are declared"),
    # Without a reset the stored table still has 2 bins.
    ("without_reset", "installed for 2 bins but 3 are declared"),
])
def test_source_count_follows_the_stored_table(case, message):
    """The refusal names the count of the table actually stored."""
    result = subprocess.run(
        args=[sys.executable, "-B", "-c", PROGRAM.format(tests=str(TESTS), case=case)],
        capture_output=True, text=True, check=False, timeout=120,
    )
    output = result.stdout + result.stderr
    assert 0 != result.returncode, output
    assert "model built" not in output
    assert message in output, output
