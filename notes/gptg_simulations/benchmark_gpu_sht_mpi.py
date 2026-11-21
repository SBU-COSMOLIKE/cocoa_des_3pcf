"""Compare CPU-only and mixed scalar-GPU generation at twelve CPU threads.

Use an otherwise quiet allocation. This script does not stop another job.
Fresh directories are required: no timed call may read a cached map.
The mixed case has three CPU workers and one GPU-scalar/CPU-shear worker;
each uses three CPU transform threads. All configurations use BLAS=1.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['VECLIB_MAXIMUM_THREADS']='1'
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from generate_maps import PROJECT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=32)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mpiexec',default='mpiexec')
    parser.add_argument('--reverse',action='store_true')
    parser.add_argument('--compare-gpu-workers', action='store_true',
                        help='Compare one versus two GPU workers at twelve CPU threads.')
    parser.add_argument('--without-cpu32',action='store_true')
    parser.add_argument('--compare-pipeline', action='store_true',
                        help='Compare sequential and overlapped CPU/GPU stages.')
    args=parser.parse_args()
    cases=[('cpu64',4,4,'float64',0),('cpu32',4,4,'float32',0),
           ('mixed',5,3,'float32',1)]
    if args.compare_gpu_workers:
        # Replace a CPU worker rather than adding a process. Both cases
        # retain four workers with three native CPU threads apiece.
        cases=[('gpu1',5,3,'float32',1),('gpu2',5,3,'float32',2)]
    if args.compare_pipeline:
        cases=[('sequential',5,3,'float32',1),('pipeline',5,3,'float32',1)]
    if args.without_cpu32:
        cases=[case for case in cases if case[0]!='cpu32']
    if args.reverse:
        cases.reverse()
    args.output.mkdir(parents=True,exist_ok=True)
    for name,ranks,threads,precision,gpu in cases:
        if (args.output/name).exists():
            raise FileExistsError('Choose a fresh --output for a real computation timing.')
    with (args.output/'wall_seconds.txt').open('w') as report:
        report.write('# mode maps wall_seconds ranks threads gpu_workers\n')
        for name,ranks,threads,precision,gpu in cases:
            folder=args.output/name
            folder.mkdir()
            for filename in ('calibration.yaml','calibration.npz'):
                shutil.copy2(PROJECT/'chains/gptg_ensemble_15000'/filename,folder/filename)
            command=[args.mpiexec,'-n',str(ranks),'--mca','mpi_yield_when_idle','1',
                     sys.executable,'-u',str(Path(__file__).with_name('generate_ensemble_mpi.py')),
                     '--threads',str(threads),'--precision',precision,
                     '--gpu-workers',str(gpu),'--limit',str(args.count),
                     '--output',str(folder.resolve())]
            if name == 'pipeline':
                command.append('--gpu-pipeline')
            start=time.monotonic()
            with (args.output/f'{name}.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            report.write(f'{name} {args.count} {time.monotonic()-start:.9f} {ranks} {threads} {gpu}\n')
            report.flush()


if __name__=='__main__':
    main()
