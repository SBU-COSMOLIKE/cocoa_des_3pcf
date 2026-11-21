#!/bin/bash
#SBATCH --job-name=DES3PCF_TREECORR
#SBATCH --output=DES3PCF_TREECORR-%j.out
#SBATCH --error=DES3PCF_TREECORR-%j.err
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
# One of the 960 ranks coordinates; the other 959 measure tomographic tasks.
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
set -e
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

# The existing launcher forwards Cocoa paths and numerical-thread settings to
# all nodes and binds each rank to one core. Do not wrap it in a second srun.
# It processes all 10000 comparison + 5000 validation maps. On resubmission,
# finite task entries in full_joint_480/map_*.txt are retained; missing/NaN
# entries are recomputed. Do not delete partial vectors between allocations.
bash "${ROOTDIR}/projects/des_3pcf/scripts/measure_gptg_hpc.sh" \
    "${SLURM_NTASKS}" 15000
