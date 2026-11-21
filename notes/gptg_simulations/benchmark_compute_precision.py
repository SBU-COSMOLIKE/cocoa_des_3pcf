"""Time optional CPU/Metal arithmetic with the same seeds and MPI allocation.

Run when no other generation job is using the assigned CPUs. This script
never suspends other jobs. It creates three separate, previously absent
ensembles below --output, so every timed evaluation generates a fresh map.
A coordinator plus three workers, each with four ducc0 threads, uses twelve
compute CPUs. Metal evaluates the point transformation; ducc0 stays on CPU.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from generate_maps import PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mpiexec', default='mpiexec')
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--ranks', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--point-batch', type=int, choices=(1, 4), default=1)
    parser.add_argument('--output', type=Path,
                        default=PROJECT / 'chains/compute_precision_mpi')
    args = parser.parse_args()
    if args.count < 1 or args.ranks < 2 or args.threads < 1:
        parser.error('Use positive count/threads and at least two MPI ranks.')
    source = PROJECT / 'chains/gptg_ensemble_15000'
    args.output.mkdir(parents=True, exist_ok=True)
    modes = [('cpu64', 'float64', 'cpu'), ('cpu32', 'float32', 'cpu'),
             ('metal32', 'float32', 'metal')]
    if args.reverse:
        modes.reverse()
    for name, precision, device in modes:
        if (args.output / name).exists():
            raise FileExistsError('Choose a fresh --output; timing must generate new maps.')
    with (args.output / 'wall_seconds.txt').open('w') as report:
        report.write('# mode count total_seconds; includes worker initialization\n')
        for name, precision, device in modes:
            folder = args.output / name
            folder.mkdir()
            for filename in ('calibration.yaml', 'calibration.npz'):
                shutil.copy2(source / filename, folder / filename)
            command = [
                args.mpiexec, '-n', str(args.ranks),
                '--mca', 'mpi_yield_when_idle', '1',
                sys.executable, '-u',
                str(Path(__file__).with_name('generate_ensemble_mpi.py')),
                '--threads', str(args.threads), '--limit', str(args.count),
                '--precision', precision, '--point-device', device,
                '--point-batch', str(args.point_batch if device == 'metal' else 1),
                '--output', str(folder.resolve()),
            ]
            start = time.monotonic()
            with (args.output / f'{name}.log').open('w') as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            report.write(f'{name} {args.count} {time.monotonic()-start:.9f}\n')
            report.flush()


if __name__ == '__main__':
    main()
