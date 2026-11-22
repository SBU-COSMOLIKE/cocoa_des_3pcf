#!/bin/bash
#SBATCH --job-name=D3PCF_TC
#SBATCH --output=D3PCF_TC-%j.out
#SBATCH --error=D3PCF_TC-%j.err
#SBATCH --nodes=10
#SBATCH --ntasks=960
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler
#SBATCH --export=ALL

# Submit from the activated Cocoa directory containing start_cocoa.sh.
# One MPI rank coordinates; the remaining ranks measure whole maps.
# No per-socket constraint: use Slurm's configured topology for these nodes.
echo "Running on host $(hostname)"
echo "Time is $(date)"
echo "Submission directory is ${SLURM_SUBMIT_DIR}"
echo "Slurm job name is ${SLURM_JOB_NAME}"
echo "Slurm job ID is ${SLURM_JOB_ID}"
echo "MPI ranks: ${SLURM_NTASKS}; CPUs per rank: ${SLURM_CPUS_PER_TASK}"
cd "${SLURM_SUBMIT_DIR:?Submit from the Cocoa directory}" || exit 1

# Prefer activating Cocoa before sbatch so this job inherits its environment.
# For a fresh batch environment, initialize conda's shell function explicitly.
# Do not launch multiple jobs which source start_cocoa.sh on this same checkout:
# it rebuilds shared links. This is one MPI job, not a Slurm array.
if [ -z "${ROOTDIR:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || exit 1
    conda activate cocoa || exit 1
    source start_cocoa.sh || exit 1
fi
: "${ROOTDIR:?Cocoa activation must define ROOTDIR}"
: "${CONDA_PREFIX:?Activate the HPC MPI environment}"

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# UofA recommendation for large MPI jobs. The site's hard limit must permit it.
ulimit -u 2000000

# Keep the launch and its Slurm allocation in this one file, as in LSST Y1.
# Forward Cocoa paths and thread settings to every node. Let the cluster's
# MPI configuration choose the network transport; do not add a second srun.
export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
cd "${ROOTDIR}/projects/des_3pcf" || exit 1

# Set GPTG_WAIT_FOR_READY=1 before sbatch while maps are still transferring.
# A .npy.ready marker must be written only after that map's rsync succeeds.
# The unquoted option below expands to either one flag or no argument.
GPTG_READY_OPTION=""
if [ "${GPTG_WAIT_FOR_READY:-0}" = 1 ]; then
  GPTG_READY_OPTION="--wait-for-ready"
fi

# Measure the map count requested below. The coordinator saves vectors in
# the maps folder's measurement output directory. Finite entries survive
# resubmission; only unfinished tasks are dispatched to workers.
# Keep the same maps folder when resubmitting after the wall-time limit,
# and never run two jobs writing to that folder at the same time.
"${CONDA_PREFIX}/bin/mpirun" -n "${SLURM_NTASKS}" \
  -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR \
  -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC \
  -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x VECLIB_MAXIMUM_THREADS \
  -x BLIS_NUM_THREADS -x NUMEXPR_NUM_THREADS \
  --mca mpi_yield_when_idle 1 --bind-to core --report-bindings \
  --map-by slot:PE=1 \
  "${ROOTDIR}/.local/bin/python" -u \
  scripts/measure_full_joint_mpi.py --count 40000 \
  ${GPTG_READY_OPTION} \
  --maps-dir "${GPTG_MAP_DIR:-${ROOTDIR}/projects/des_3pcf/chains/gptg_ensemble_15000}"
