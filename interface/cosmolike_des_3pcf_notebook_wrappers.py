"""Notebook-facing comparison calls for the DES three-point project.

Numerical comparisons live in des3pcf.comparison. This module provides the
same named import location used by the other Cocoa notebook projects. It
does not prepare a cosmology, import an original source package, or start
a calculation at import time. Original-code runs use isolated processes.
"""

from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
CORE = PROJECT.parents[1] / "external_modules/code/cosmolike_core"
sys.path.insert(0, str(PROJECT / "python"))
sys.path.insert(0, str(CORE))

from des3pcf.comparison import comparison_status
from des3pcf.comparison import comparison_case
from des3pcf.comparison import comparison_summary
from des3pcf.comparison import load_comparison_case
from des3pcf.comparison import run_code_comparison


__all__ = [
    "comparison_status",
    "comparison_case",
    "comparison_summary",
    "load_comparison_case",
    "run_code_comparison",
]
