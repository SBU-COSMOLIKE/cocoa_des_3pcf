#!/bin/bash
#SBATCH --job-name=D3PCF_TC
#SBATCH --output=D3PCF_TC-%j.out
#SBATCH --error=D3PCF_TC-%j.err
#SBATCH --nodes=10
#SBATCH --ntasks=940
#SBATCH --ntasks-per-node=94
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
# Initialize Cocoa once before submitting array jobs. Simultaneous calls to
# start_cocoa.sh rebuild shared links in the same checkout.
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
# Forward Cocoa paths and thread settings to every node. Use Cocoa's ob1
# transport: shared memory within a node and TCP between nodes, bypassing UCX.
# Keep one MPI launcher; do not add a second srun. Adapt the TCP exclusion
# list below to the cluster; remove ib0 from it if using routable IP over IB.
# This exclusion list was tested across UofA nodes; it leaves enp1s0f1np1
# available. Other clusters may use different interface names.
# Each worker is single-threaded, so use Cocoa's slot mapping.
# Open MPI cannot use TCP include and exclude lists together. Clear an
# inherited inclusion list so the explicit Cocoa exclusion line below wins.
unset OMPI_MCA_btl_tcp_if_include
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

# Array task zero starts at map zero; each following task advances by one
# range size. Each range has its own checkpoint directory.
# The original unbounded job uses the historical output directory.
GPTG_RANGE_OPTIONS=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  GPTG_RANGE_SIZE="${GPTG_RANGE_SIZE:-5000}"
  GPTG_MAP_COUNT="${GPTG_MAP_COUNT:-40000}"
  if ! [[ "$GPTG_RANGE_SIZE" =~ ^[0-9]+$ &&
          "$GPTG_MAP_COUNT" =~ ^[0-9]+$ && "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]]; then
    echo "Map range settings must be nonnegative integers." >&2
    exit 1
  fi
  GPTG_START=$((SLURM_ARRAY_TASK_ID * GPTG_RANGE_SIZE))
  GPTG_STOP=$((GPTG_START + GPTG_RANGE_SIZE))
  if [ "$GPTG_RANGE_SIZE" -eq 0 ] || [ "$GPTG_START" -ge "$GPTG_MAP_COUNT" ]; then
    echo "This array task has no map IDs in the requested ensemble." >&2
    exit 1
  fi
  if [ "$GPTG_STOP" -gt "$GPTG_MAP_COUNT" ]; then
    GPTG_STOP="$GPTG_MAP_COUNT"
  fi
  GPTG_RANGE_OPTIONS=(--count "$GPTG_MAP_COUNT" --start "$GPTG_START"
    --stop "$GPTG_STOP" --output-dir
    "${GPTG_MAP_DIR:-${ROOTDIR}/projects/des_3pcf/chains/gptg_ensemble_15000}/full_joint_480_shards/${GPTG_START}_${GPTG_STOP}")
  echo "Measuring map IDs [${GPTG_START}, ${GPTG_STOP}) in a separate checkpoint directory"
else
  GPTG_RANGE_OPTIONS=(--count 40000)
fi
"${CONDA_PREFIX}/bin/mpirun" -n "${SLURM_NTASKS}" --oversubscribe \
  --mca pml ob1 --mca btl vader,tcp,self \
  -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR \
  -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC \
  -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x VECLIB_MAXIMUM_THREADS \
  -x BLIS_NUM_THREADS -x NUMEXPR_NUM_THREADS \
  --mca mpi_yield_when_idle 1 \
  --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0,eno1,enp1s0f0np0 \
  --bind-to core:overload-allowed --report-bindings \
  --rank-by slot --map-by slot \
  "${ROOTDIR}/.local/bin/python" -u \
  scripts/measure_full_joint_mpi.py "${GPTG_RANGE_OPTIONS[@]}" \
  ${GPTG_READY_OPTION} \
  --maps-dir "${GPTG_MAP_DIR:-${ROOTDIR}/projects/des_3pcf/chains/gptg_ensemble_15000}"
