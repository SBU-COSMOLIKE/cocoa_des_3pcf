#!/bin/bash
#SBATCH --job-name=D3PCF_GEN
#SBATCH --output=D3PCF_GEN-%j.out
#SBATCH --error=D3PCF_GEN-%j.err
#SBATCH --nodes=5
#SBATCH --ntasks=300
#SBATCH --ntasks-per-node=60
#SBATCH --ntasks-per-socket=30
#SBATCH --cpus-per-task=1
#SBATCH --mem=450G
#SBATCH --time=24:00:00
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler
#SBATCH --export=ALL

# Submit from the activated Cocoa directory. One rank owns the journals;
# the other ranks generate independent maps, with CPU transforms only.
# Keep the launch in this file, following the LSST Y1 Slurm examples.
echo "Running on host $(hostname)"
echo "Time is $(date)"
echo "Directory is ${SLURM_SUBMIT_DIR}"
echo "Slurm job name is ${SLURM_JOB_NAME}"
echo "Slurm job ID is ${SLURM_JOB_ID}"
echo "MPI ranks: ${SLURM_NTASKS}; CPUs per rank: ${SLURM_CPUS_PER_TASK}"
cd "${SLURM_SUBMIT_DIR:?Submit from the Cocoa directory}" || exit 1
if [ -z "${ROOTDIR:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || exit 1
    conda activate cocoa || exit 1
    source start_cocoa.sh || exit 1
fi

# The calibration folder contains calibration.yaml, calibration.npz and
# the input indices_<calibration NSIDE>.npy. Its fitted harmonic cutoff is
# retained: changing the output pixels does not invent small-scale power.
# Use a new output folder for a new resolution or calibration.
# Counts may be reduced through the environment for a generation pilot.
: "${GPTG_CALIBRATION_DIR:?Set the folder containing the frozen calibration}"
: "${GPTG_OUTPUT_DIR:?Set a separate output folder for this ensemble}"
GPTG_COMPARISON_COUNT="${GPTG_COMPARISON_COUNT:-10000}"
GPTG_VALIDATION_COUNT="${GPTG_VALIDATION_COUNT:-5000}"
GPTG_NSIDE="${GPTG_NSIDE:-2048}"
GPTG_LMAX="${GPTG_LMAX:-4096}"
echo "Output ensemble: ${GPTG_OUTPUT_DIR}"
echo "Target NSIDE: ${GPTG_NSIDE}; required harmonic cutoff: ${GPTG_LMAX}"
echo "Maps: ${GPTG_COMPARISON_COUNT} comparison; ${GPTG_VALIDATION_COUNT} validation"

# Check before launching MPI. LMAX selects a required calibration; it does
# not extrapolate its spectrum. Check the frozen output copy on resume too.
"${ROOTDIR}/.local/bin/python" - "${GPTG_CALIBRATION_DIR}" "${GPTG_OUTPUT_DIR}" "${GPTG_LMAX}" <<'PYTHON'
from pathlib import Path
import sys
import yaml

expected = int(sys.argv[3])
for index, directory in enumerate(sys.argv[1:3]):
    path = Path(directory) / 'calibration.yaml'
    if index == 1 and not path.exists():
        continue
    record = yaml.safe_load(path.read_text())
    if record['lmax'] != expected:
        raise SystemExit(
            f'{path}: lmax={record["lmax"]}, expected {expected}. '
            'Supply a calibration fitted at the requested cutoff; '
            'do not change only its YAML value.'
        )
PYTHON
if [ "$?" -ne 0 ]; then
    exit 1
fi

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
ulimit -u 2000000

# Keep Cocoa's TCP exclusion list; an inherited inclusion list conflicts
# with it. The interface names below are the tested UofA configuration.
unset OMPI_MCA_btl_tcp_if_include
# Each worker owns one CPU core. Limit native transforms and BLAS to one
# thread so independent maps use the allocation without oversubscription.
# Worker memory was measured locally at this cutoff. Check HPC peak RSS
# in a pilot before filling the allocation with production workers.
"${CONDA_PREFIX:?Activate Cocoa}/bin/mpirun" -n "${SLURM_NTASKS}" \
  --mca pml ob1 --mca btl vader,tcp,self \
  -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR \
  -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC \
  -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x VECLIB_MAXIMUM_THREADS \
  -x BLIS_NUM_THREADS -x NUMEXPR_NUM_THREADS \
  --mca mpi_yield_when_idle 1 \
  --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0,eno1,enp1s0f0np0 \
  --bind-to core --report-bindings \
  --rank-by slot --map-by "ppr:${SLURM_NTASKS_PER_SOCKET}:socket:PE=${SLURM_CPUS_PER_TASK}" \
  "${ROOTDIR}/.local/bin/python" -u \
  "${ROOTDIR}/projects/des_3pcf/scripts/generate_ensemble_mpi.py" \
  --output "${GPTG_OUTPUT_DIR}" \
  --calibration-dir "${GPTG_CALIBRATION_DIR}" \
  --nside "${GPTG_NSIDE}" \
  --comparison-count "${GPTG_COMPARISON_COUNT}" \
  --validation-count "${GPTG_VALIDATION_COUNT}" \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --precision float64 --point-device cpu --gpu-workers 0

# Resubmit with the same output, calibration and counts after the time limit.
# Completed map IDs are skipped; an atomic save ahead of its journal is
# remeasured without regenerating the map. Never run simultaneous writers
# on the same output ensemble. No TreeCorr measurements run in this job.
