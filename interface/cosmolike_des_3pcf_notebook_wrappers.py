"""Notebook-facing comparison calls for the DES three-point project.

The numerical comparison machinery lives in scripts/des3pcf/comparison.py;
this module only re-exports its five entry points, so notebooks import
them from the same cosmolike_<project>_notebook_wrappers location the
other Cocoa projects use. Importing it prepares no cosmology and starts
no calculation. The original code is fastnc, the external three-point
natural-components package this project is validated against; its runs
happen in isolated worker processes, so its imports and settings can
never leak into the Cocoa process doing the comparison.
"""

from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
CORE = PROJECT.parents[1] / "external_modules/code/cosmolike_core"
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(CORE))

# comparison_status reports whether the original-code checkouts and the
# Python dependencies are available and unmodified, without computing.
from des3pcf.comparison import comparison_status
# comparison_case selects one named case from a finished report, refusing
# absent or failed calculations.
from des3pcf.comparison import comparison_case
# comparison_summary returns a short table with every case's status and
# worst scaled difference.
from des3pcf.comparison import comparison_summary
# load_comparison_case reads a completed case's plotting arrays back from
# its saved npz file without recomputing anything.
from des3pcf.comparison import load_comparison_case
# run_code_comparison runs every available comparison serially and saves
# the case arrays and provenance the other calls read.
from des3pcf.comparison import run_code_comparison


__all__ = [
    "comparison_status",
    "comparison_case",
    "comparison_summary",
    "load_comparison_case",
    "run_code_comparison",
]
