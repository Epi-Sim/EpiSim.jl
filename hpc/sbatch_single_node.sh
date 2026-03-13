#!/bin/bash
# Single-Node Maxed Out Synthetic Pipeline
# Runs entire pipeline on one GPP node using the Python orchestrator
#
# Usage: sbatch sbatch_single_node.sh [--n-jobs N] [--n-profiles M]
#   --n-jobs N: Number of parallel workers (default: 45)
#   --n-profiles M: Number of epidemiological profiles (default: 50)
#
# CPU Allocation Strategy:
# Uses a single --n-jobs parameter for both Python multiprocessing and Julia simulations.
# This provides better failure tolerance at the cost of some performance (Julia runs
# single-threaded with Python managing parallelism).
#
# RAM Usage (256GB GPP node):
# - Simulations: n_jobs × 4GB = ~180GB (with 45 parallel)
# - Processing: n_jobs × 4GB = ~180GB
# - Phases don't overlap, so memory usage stays manageable

#SBATCH --job-name=synth_single_node
#SBATCH --qos=gp_bscls
#SBATCH --account=bsc08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --time=01:00:00
#SBATCH --output=logs/synth_single_%j.out
#SBATCH --error=logs/synth_single_%j.err
#SBATCH --chdir=/gpfs/projects/bsc08/shared_projects/MePreCiSa/synthetic_episim

# Allow individual runs to fail without killing the whole job
set -uo pipefail

export PROJECT_DIR=/gpfs/projects/bsc08/shared_projects/MePreCiSa/synthetic_episim

# Parse command line arguments
N_JOBS=""
N_PROFILES=""
while [[ $# -gt 0 ]]; do
  case $1 in
  --n-jobs)
    N_JOBS="$2"
    shift 2
    ;;
  --n-profiles)
    N_PROFILES="$2"
    shift 2
    ;;
  *)
    echo "Unknown option: $1"
    echo "Usage: $0 [--n-jobs N] [--n-profiles M]"
    exit 1
    ;;
  esac
done

# Set defaults
AVAILABLE_CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
N_JOBS=${N_JOBS:-40}
N_PROFILES=${N_PROFILES:-50}
BATCH_SIZE=${N_JOBS} # Process N_JOBS profiles at a time

echo "=========================================="
echo "Single-Node Synthetic Pipeline"
echo "=========================================="
echo "Available CPUs: ${AVAILABLE_CPUS}"
echo "n_jobs (parallel workers): ${N_JOBS}"
echo "Profiles: ${N_PROFILES}"
echo "Batches: $(((N_PROFILES + BATCH_SIZE - 1) / BATCH_SIZE))"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Start: $(date)"
echo ""

# Configuration
DATASET="catalonia"
OUTPUT_BASE="$PROJECT_DIR/runs/synthetic_${DATASET}"

# Use NVMe for all temporary storage
NVME_BASE="${TMPDIR:-/tmp}/synthetic_pipeline"
mkdir -p "${NVME_BASE}"

cd "$PROJECT_DIR" || exit
source python/.venv/bin/activate

# Enable offline mode to prevent network access on worker nodes
# Packages must be pre-installed on login node with: julia --project=. -e 'using Pkg; Pkg.instantiate()'
export JULIA_PKG_OFFLINE=true
export JULIA_PROJECT="$PROJECT_DIR"

# ============================================================================
# RUN PYTHON PIPELINE (handles generation, simulation, processing, NVMe staging)
# ============================================================================
echo ""
echo "=========================================="
echo "RUNNING PYTHON PIPELINE"
echo "=========================================="
echo "Using unified Python orchestrator with NVMe staging..."

python python/run_synthetic_pipeline.py \
  --n-profiles ${N_PROFILES} \
  --n-jobs ${N_JOBS} \
  --batch-size ${BATCH_SIZE} \
  --nvme-base "${NVME_BASE}" \
  --dataset ${DATASET} \
  --failure-tolerance 10 \
  --intervention-profile-fraction 0.0 \
  --mobility-sigma-max 0.6 \
  --sparsity-mode tiers \
  --sparsity-tiers 0.05 0.20 0.40 0.60 0.80 \
  --sparsity-seed 42

PYTHON_EXIT=$?

# Cleanup NVMe
echo "Cleaning up NVMe..."
rm -rf "${NVME_BASE}"

if [ $PYTHON_EXIT -ne 0 ]; then
  echo ""
  echo "=========================================="
  echo "PIPELINE FAILED with exit code ${PYTHON_EXIT}"
  echo "=========================================="
  exit $PYTHON_EXIT
fi

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE!"
echo "=========================================="
echo "Output: ${OUTPUT_BASE}/raw_synthetic_observations.zarr"
echo "Baselines: ${OUTPUT_BASE}/baselines"
echo "End: $(date)"
