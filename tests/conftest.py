"""Load this project's compiled interface for small component checks.

The project has its own extension module, as the LSST Y1 project does.
The extension contains the three-point C calculations and their two C++
adapters. It does not install a cosmology or load survey data. Adding its
directory to the import search path selects this local build; it does not
replace any imported function or module.
"""

from pathlib import Path
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]  # Owns interface/ and tests/.
CORE = PROJECT.parents[1] / "external_modules" / "code" / "cosmolike_core"
sys.path.insert(0, str(PROJECT / "interface"))
sys.path.insert(0, str(PROJECT / "python"))
# Fisher diagnostics are shared source, not a separately installed package.
sys.path.insert(0, str(CORE))


@pytest.fixture(scope="session")
def compiled():
    """Return the project extension with a one-worker default.

    Returns:
        module = production and notebook adapters around the shared C code.

    The explicit thread setting keeps ordinary checks small. The thread
    comparison test changes it and restores it before returning.
    """
    import cosmolike_des_3pcf_interface

    cosmolike_des_3pcf_interface.set_omp_threads(nthreads=1)
    return cosmolike_des_3pcf_interface
