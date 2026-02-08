#!/bin/bash
# Single-Node Maxed Out Synthetic Pipeline
# Runs entire pipeline on one GPP node with phase-specific CPU allocation
#
# Usage: sbatch sbatch_single_node.sh [--julia-cpus N] [--python-cpus N]
#   --julia-cpus N: CPUs for Julia simulations (default: 45)
#   --python-cpus N: CPUs for Python processing (default: 25)
#
# CPU Allocation Strategy (Option 3):
# - Phase 1 (Julia sims): Uses JULIA_CPUS (45) - compute intensive
# - Phase 2a (Python processing): Uses PYTHON_CPUS (25) - I/O intensive
# - Phases don't overlap, so each gets optimal allocation
#
# RAM Usage (256GB GPP node):
# - Julia: 50 sims × 4GB = 200GB (with 45 parallel)
# - Python: 25 workers × 3.2GB = 80GB (mobility matrices)

#SBATCH --job-name=synth_single_node
#SBATCH --qos=gp_bscls
#SBATCH --account=bsc08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --time=04:00:00
#SBATCH --output=logs/synth_single_%j.out
#SBATCH --error=logs/synth_single_%j.err
#SBATCH --chdir=/home/bsc/bsc008913/EpiSim.jl

set -euo pipefail

# Parse command line arguments
JULIA_CPUS=""
PYTHON_CPUS=""
while [[ $# -gt 0 ]]; do
  case $1 in
  --julia-cpus)
    JULIA_CPUS="$2"
    shift 2
    ;;
  --python-cpus)
    PYTHON_CPUS="$2"
    shift 2
    ;;
  *)
    echo "Unknown option: $1"
    echo "Usage: $0 [--julia-cpus N] [--python-cpus N]"
    exit 1
    ;;
  esac
done

# Set defaults
AVAILABLE_CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
JULIA_CPUS=${JULIA_CPUS:-45}
PYTHON_CPUS=${PYTHON_CPUS:-45}

echo "=========================================="
echo "Single-Node Synthetic Pipeline"
echo "=========================================="
echo "Available CPUs: ${AVAILABLE_CPUS}"
echo "Julia CPUs (simulations): ${JULIA_CPUS}"
echo "Python CPUs (processing): ${PYTHON_CPUS}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Start: $(date)"
echo ""

# Configuration
N_PROFILES=50
BATCH_SIZE=${JULIA_CPUS} # Process JULIA_CPUS profiles at a time
DATASET="catalonia"
DATA_FOLDER="/home/bsc/bsc008913/EpiSim.jl/models/${DATASET}"
CONFIG_PATH="${DATA_FOLDER}/config_MMCACovid19.json"
METAPOP_CSV="${DATA_FOLDER}/metapopulation_data.csv"
OUTPUT_BASE="/home/bsc/bsc008913/EpiSim.jl/runs/synthetic_${DATASET}"
BASELINE_DIR="${OUTPUT_BASE}/baselines"
INTERVENTION_DIR="${OUTPUT_BASE}/interventions"
SPIKE_THRESHOLD=0.1

# Use NVMe for all temporary storage
NVME_BASE="${TMPDIR:-/tmp}/synthetic_pipeline"
mkdir -p "${NVME_BASE}"

cd /home/bsc/bsc008913/EpiSim.jl
source python/.venv/bin/activate

export JULIA_PROJECT=/home/bsc/bsc008913/EpiSim.jl

# Enable offline mode to prevent network access on worker nodes
# Packages must be pre-installed on login node with: julia --project=. -e 'using Pkg; Pkg.instantiate()'
export JULIA_PKG_OFFLINE=true

# ============================================================================
# PHASE 1: Baseline Generation (Julia Parallel)
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 1: Baseline Generation (Julia)"
echo "=========================================="
echo "Using ${JULIA_CPUS} CPUs for Julia simulations"

N_BATCHES=$(((N_PROFILES + BATCH_SIZE - 1) / BATCH_SIZE))
echo "Processing ${N_PROFILES} profiles in ${N_BATCHES} batches of up to ${BATCH_SIZE}..."

export JULIA_NUM_THREADS=${JULIA_CPUS}

for batch_idx in $(seq 0 $((N_BATCHES - 1))); do
  start_idx=$((batch_idx * BATCH_SIZE))
  end_idx=$((start_idx + BATCH_SIZE))
  if [ ${end_idx} -gt ${N_PROFILES} ]; then
    end_idx=${N_PROFILES}
  fi

  actual_batch_size=$((end_idx - start_idx))

  echo ""
  echo "--- Batch $((batch_idx + 1))/${N_BATCHES} (profiles ${start_idx}-$((end_idx - 1))) ---"
  echo "Generating ${actual_batch_size} configurations..."

  # Generate baseline configs for this batch
  python python/synthetic_generator.py \
    --n-profiles ${N_PROFILES} \
    --start-index ${start_idx} \
    --end-index ${end_idx} \
    --output-folder "${NVME_BASE}/baselines" \
    --data-folder "${DATA_FOLDER}" \
    --config "${CONFIG_PATH}" \
    --baseline-only \
    --failure-tolerance 10 \
    --intervention-profile-fraction 0.0 \
    --mobility-sigma-min 0.0 \
    --mobility-sigma-max 0.6

  # Run simulations for this batch in parallel
  echo "Running ${actual_batch_size} simulations with ${JULIA_CPUS} Julia threads..."
  julia --project=/home/bsc/bsc008913/EpiSim.jl \
    -t "${JULIA_CPUS}" \
    src/batch_run.jl \
    --batch-folder "${NVME_BASE}/baselines" \
    --data-folder "${DATA_FOLDER}"

  # Copy completed runs to GPFS
  echo "Copying results to GPFS..."
  mkdir -p "${BASELINE_DIR}"
  for run_dir in "${NVME_BASE}/baselines"/run_*_Baseline; do
    if [ -d "${run_dir}" ]; then
      run_name=$(basename "${run_dir}")
      rsync -av "${run_dir}/" "${BASELINE_DIR}/${run_name}/"
    fi
  done

  # Clear NVMe for next batch
  rm -rf "${NVME_BASE}/baselines"
done

echo ""
echo "Phase 1 complete!"

# ============================================================================
# PHASE 2a: Process Baselines (Python Parallel)
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 2a: Process Baselines (Python)"
echo "=========================================="
echo "Using ${PYTHON_CPUS} CPUs for Python processing"

mkdir -p "${NVME_BASE}/processing"
BASELINE_ZARR="${NVME_BASE}/processing/raw_synthetic_observations.zarr"

echo "Processing baselines into zarr with ${PYTHON_CPUS} parallel workers..."
python python/process_synthetic_outputs.py \
  --runs-dir "${BASELINE_DIR}" \
  --metapop-csv "${METAPOP_CSV}" \
  --output "${BASELINE_ZARR}" \
  --baseline-only \
  --sparsity-mode tiers \
  --sparsity-tiers 0.05 0.20 0.40 0.60 0.80 \
  --sparsity-seed 42 \
  --n-jobs "${PYTHON_CPUS}"

echo "Copying baseline zarr to GPFS..."
mkdir -p "${OUTPUT_BASE}"
rsync -av "${BASELINE_ZARR}/" "${OUTPUT_BASE}/raw_synthetic_observations.zarr/"

echo ""
echo "Phase 2a complete!"

# ============================================================================
# PHASE 2b: Generate and Run Interventions (if needed)
# ============================================================================
# Skip for 0 interventions
N_INTERVENTIONS=0

# ============================================================================
# PHASE 3: Final Processing (if needed)
# ============================================================================
# Skip for 0 interventions

# Cleanup NVMe
echo "Cleaning up NVMe..."
rm -rf "${NVME_BASE}"

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE!"
echo "=========================================="
echo "Output: ${OUTPUT_BASE}/raw_synthetic_observations.zarr"
echo "Baselines: ${BASELINE_DIR}"
echo "End: $(date)"
