"""Check that the n(z) readers refuse a table of another bin count.

The stored source and lens n(z) tables hold one row per bin plus the
redshift row. Declaring a new bin count (init_source_sample_size,
init_lens_sample_size) without installing a matching table must stop the
readers zdistr_histo_n and pf_histo_n: with more bins they would read past
the allocation, with fewer they would read the wrong rows. The refusal
ends the process, so each case runs in a child interpreter. The readers
are plain exported C functions, called through ctypes.
"""

import subprocess
import sys

import pytest


PROGRAM = """
import ctypes
import numpy as np
import cosmolike_des_3pcf_interface as compiled

library = ctypes.CDLL(compiled.__file__)
reader = getattr(library, {reader!r})
reader.restype = ctypes.c_double
reader.argtypes = [ctypes.c_double, ctypes.c_int]
z = np.linspace(0.01, 2.0, 60)
table = np.ascontiguousarray(np.column_stack(
    [z] + [np.exp(-((z - 0.3 * (index + 1)) / 0.2)**2) for index in range(3)]))
size = getattr(compiled, {size!r})
install = getattr(compiled, {install!r})

compiled.initial_setup()
size(Ntomo=3)
install(nofz=table)
# flush: the native refusal ends the process without flushing Python.
print("matched", reader(0.5, 0), flush=True)
size(Ntomo={later})
print("read", reader(0.5, 0), flush=True)
"""

SAMPLES = {
    "source": ("zdistr_histo_n", "init_source_sample_size", "set_source_sample"),
    "lens": ("pf_histo_n", "init_lens_sample_size", "set_lens_sample"),
}


@pytest.mark.parametrize("sample", ["source", "lens"])
@pytest.mark.parametrize("later", [2, 4])
def test_reader_refuses_a_declared_count_without_its_table(sample, later):
    """A 3-bin table read after declaring another count is refused."""
    reader, size, install = SAMPLES[sample]
    program = PROGRAM.format(reader=reader, size=size, install=install, later=later)
    result = subprocess.run(args=[sys.executable, "-B", "-c", program],
                            capture_output=True, text=True, check=False, timeout=120)
    output = result.stdout + result.stderr
    assert "matched" in output, output
    assert 0 != result.returncode, output
    assert "read " not in output
    assert f"{sample} n(z) table was installed for 3 bins but {later} are declared" in output
