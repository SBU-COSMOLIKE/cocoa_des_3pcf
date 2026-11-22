"""Check the companion notebook's covariance algebra without measuring maps.

Read the actual function cells, then compare the distinct-factor recurrence
with explicit products in a three-dimensional known-covariance example.
This executes no MPI, GPU or TreeCorr program and writes no simulation data.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from pathlib import Path
import json
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
notebook = json.loads((PROJECT / 'notes/gptg_simulations/EXAMPLE_GPTG_SIMULATIONS.ipynb').read_text())
namespace = {'np': np, 'PROJECT': PROJECT}
for cell in notebook['cells']:
    source = ''.join(cell['source'])
    if cell['cell_type'] == 'code' and (
        'def estimates(' in source or source.startswith('def check_fixed_block(')
    ):
        exec(compile(source, 'companion notebook', 'exec'), namespace)
namespace['check_fixed_block']()
