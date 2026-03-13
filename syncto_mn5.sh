#!/bin/bash
# Sync EpiSim.jl codebase to MareNostrum 5
# Usage: ./syncto_mn5.sh [--dry-run]
# Requires: SSH access to MN5 (bsc008913@dt)

set -e

# Parse arguments
DRY_RUN=""
if [[ "$1" == "--dry-run" ]]; then
  DRY_RUN="--dry-run"
  echo "DRY RUN MODE - No files will be transferred"
  echo ""
fi

# MN5 destination (home directory)
DIR="/gpfs/projects/bsc08/shared_projects/MePreCiSa/synthetic_episim"
DEST="dt:$DIR"

echo "Syncing EpiSim.jl to MareNostrum 5..."
echo "Destination: ${DEST}"
echo ""

# Rsync filters: first matching rule wins, so order matters!
# Include python/data/ contents before excluding *.zarr
rsync -avz --progress . "${DEST}" \
  --include="python/data/**" \
  --exclude=".git/" \
  --exclude=".venv/" \
  --exclude="venv/" \
  --exclude="python/.venv/" \
  --exclude="python/venv/" \
  --exclude="__pycache__/" \
  --exclude=".cache/" \
  --exclude=".julia/" \
  --exclude="julia-depot/" \
  --exclude=".mypy_cache/" \
  --exclude=".pytest_cache/" \
  --exclude=".ruff_cache/" \
  --exclude="*.egg-info/" \
  --exclude="runs/" \
  --exclude="*.nc" \
  --exclude="*.zarr" \
  --exclude=".DS_Store" \
  --exclude="*.log" \
  ${DRY_RUN}

echo ""
if [[ -n "$DRY_RUN" ]]; then
  echo "Dry run complete!"
else
  echo "Sync complete!"
  echo ""
  echo "Next steps on MN5:"
  echo "  1. SSH to MN5: ssh bsc008913@dt"
  echo "  2. Navigate to project: cd $DIR"
  echo "  3. Setup Julia environment: module load julia/1.10 && julia --project=. -e 'using Pkg; Pkg.instantiate()'"
  echo "  4. Setup Python environment: cd python && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  echo "  5. Submit job: sbatch hpc/sbatch_single_node.sh"
fi
