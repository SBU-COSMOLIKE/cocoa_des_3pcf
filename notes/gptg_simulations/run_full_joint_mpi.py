"""Run the complete 1000+500 test within the twelve-core allocation.

Keep the generator at four native workers. This driver uses eight single-CPU
MPI measurement ranks plus an idle coordinator. All ranks exit before the
single-threaded matrix analysis, so matrix work does not add a thirteenth
busy core. The second measurement resumes saved vectors from the first.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mpiexec', default='mpiexec')
args = parser.parse_args()
folder = Path(__file__).resolve().parent
for count, validate in ((1000, False), (1500, True)):
    subprocess.run([args.mpiexec, '-n', '9', sys.executable, '-u',
                    str(folder / 'measure_full_joint_mpi.py'),
                    '--count', str(count)], check=True)
    command = [sys.executable, '-u', str(folder / 'analyze_full_joint.py')]
    if validate:
        command.append('--validate')
    subprocess.run(command, check=True)
