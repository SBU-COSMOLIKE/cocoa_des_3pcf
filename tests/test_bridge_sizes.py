"""The native bridge rejects oversized table counts before reading arrays.

A product of positive C int counts can overflow even though each axis count
is individually valid. The source and feedback setters scan and copy flat
arrays using those products, so their size checks must precede both the
multiplications and the first input read. These tests never allocate a
large table. They pass one NaN value as an unread sentinel and require the
specific size error, which also proves the NaN scan was not reached.

Each refusal runs in a child process because the native error contract
calls exit(1). The parent pytest process must survive to inspect the error.
No matter model, integration or survey prediction is constructed.
"""

import ctypes
import os
from pathlib import Path
import subprocess
import sys

import pytest


# Derive the bound from the same C int type ctypes passes to the exported
# functions. This does not assume that a pointer and an integer have equal
# widths, and it needs no guessed public-struct layout.
INT_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_int) - 1)) - 1


# The native scalar/pointer prototypes below match threepcf_core_bridge.h.
# The shared object is the same explicit file selected by pytest's compiled
# fixture. Importing it initializes the module before ctypes calls its C API.
REFUSAL_CALL = r"""
import ctypes
from pathlib import Path
import sys

binary = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(binary.parent))
import cosmolike_des_3pcf_interface

if Path(cosmolike_des_3pcf_interface.__file__).resolve() != binary:
    raise RuntimeError("The child imported a different extension.")
library = ctypes.CDLL(name=str(binary))
pointer_type = ctypes.POINTER(ctypes.c_double)
sentinel = ctypes.c_double(float("nan"))
pointer = ctypes.pointer(sentinel)
counts = []
for value in sys.argv[3:]:
    counts.append(int(value))
if sys.argv[2] == "sources":
    function = library.threepcf_bridge_set_sources
    function.argtypes = [
        pointer_type,
        pointer_type,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = None
    function(pointer, pointer, counts[0], counts[1])
else:
    function = library.threepcf_bridge_set_feedback
    function.argtypes = [
        pointer_type,
        ctypes.c_int,
        pointer_type,
        ctypes.c_int,
        pointer_type,
        ctypes.c_int,
        pointer_type,
        ctypes.c_int,
        pointer_type,
        pointer_type,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
    ]
    function.restype = None
    # The final controls request ordinary upsampling, boundary holding and
    # zero response amplitude. Only the table dimensions are invalid.
    function(pointer, counts[0], pointer, counts[1], pointer, counts[2],
             pointer, counts[3], pointer, pointer, 2, 1, 0.0)
raise RuntimeError("The oversized request unexpectedly returned.")
"""


@pytest.mark.parametrize("kind,counts", [
    ("sources", (INT_MAX, 2)),
    ("sources", (2, INT_MAX)),
    ("feedback", (INT_MAX, 4, 4, 4)),
    ("feedback", (4, INT_MAX, 4, 4)),
    ("feedback", (4, 4, INT_MAX, 4)),
    ("feedback", (4, 4, 4, INT_MAX)),
])
def test_bridge_refuses_overflow_before_array_reads(compiled, kind, counts):
    """This test refuses a flat size that cannot fit the core's int count.

    A one-value NaN sentinel makes the order of validation observable without
    allocating gigabytes. The intended size error must occur before scanning
    even the first axis value; a later NaN error cannot satisfy this test.

    Arguments:
        compiled = extension selected by pytest's project fixture.
        kind = sources or feedback native setter.
        counts = positive axis counts whose product exceeds INT_MAX.
    Returns:
        None; assertions require exit(1) and the precise size error.
    Side effects:
        Starts one short child interpreter, which exits during validation.
    """
    environment = dict(os.environ)
    # No numerical model is evaluated, but child imports must still obey the
    # project's rule that BLAS uses one worker and OpenMP receives the cores.
    for name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["OMP_THREAD_LIMIT"] = "6"
    command = [
        sys.executable,
        "-c",
        REFUSAL_CALL,
        str(Path(compiled.__file__).resolve()),
        kind,
    ]
    for count in counts:
        command.append(str(count))
    result = subprocess.run(
        args=command,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    if kind == "sources":
        assert "source table count" in output, output
    else:
        assert "feedback table count" in output, output
    assert "exceeds INT_MAX" in output, output
    assert "contains NaN" not in output, output
    assert "runtime error:" not in output, output
