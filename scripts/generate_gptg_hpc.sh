#!/bin/bash
# Execute from activated Cocoa on a Linux HPC allocation with shared storage.
# Argument 1 counts ALL MPI ranks, including the coordinator; argument 2
# is the desired global map prefix. Repeating a command resumes its journal.
set -e
: "${ROOTDIR:?Source start_cocoa.sh first}"
: "${CONDA_PREFIX:?Activate the cluster MPI environment first}"
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
cd "${ROOTDIR}/projects/des_3pcf"
# Leave the transport to the cluster's MPI configuration. In particular,
# do not exclude its inter-node network or force a legacy Open MPI BTL.
"${CONDA_PREFIX}/bin/mpirun" -n "${1:?Supply total MPI ranks}" \
  -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR \
  -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC \
  -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x VECLIB_MAXIMUM_THREADS \
  --mca mpi_yield_when_idle 1 --bind-to core --report-bindings \
  --map-by slot:PE=${OMP_NUM_THREADS} \
  "${ROOTDIR}/.local/bin/python" -u \
  scripts/generate_ensemble_mpi.py --limit "${2:-15000}" --threads "${OMP_NUM_THREADS}"
