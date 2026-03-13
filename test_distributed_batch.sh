#!/bin/bash
# Test script for Distributed.jl batch runner
# Run this to verify the new implementation works correctly

echo "============================================"
echo "Testing Distributed.jl Batch Runner"
echo "============================================"

# Create a minimal test setup
TEST_DIR=$(mktemp -d)
echo "Test directory: $TEST_DIR"

# Create a minimal batch folder structure
mkdir -p "$TEST_DIR/run_test_1"
mkdir -p "$TEST_DIR/run_test_2"

# Create minimal config files (these won't actually run simulations,
# but will test the directory scanning and worker spawning)
cat > "$TEST_DIR/run_test_1/config_auto_py.json" << 'EOF'
{
  "simulation": {
    "engine": "MMCACovid19",
    "start_date": "2020-03-01",
    "end_date": "2020-03-02"
  },
  "data": {
    "metapopulation_data_filename": "metapopulation_data.csv",
    "mobility_matrix_filename": "mobility_matrix.csv"
  },
  "epidemic_params": {},
  "population_params": {},
  "NPI": {}
}
EOF

cat > "$TEST_DIR/run_test_2/config_auto_py.json" << 'EOF'
{
  "simulation": {
    "engine": "MMCACovid19",
    "start_date": "2020-03-01",
    "end_date": "2020-03-02"
  },
  "data": {
    "metapopulation_data_filename": "metapopulation_data.csv",
    "mobility_matrix_filename": "mobility_matrix.csv"
  },
  "epidemic_params": {},
  "population_params": {},
  "NPI": {}
}
EOF

echo ""
echo "Test 1: Sequential mode (workers=1)"
echo "------------------------------------"
julia --project=. -t 1 src/batch_run.jl \
  --batch-folder "$TEST_DIR" \
  --data-folder "models/catalonia" \
  --workers 1 2>&1 | head -20

echo ""
echo "Test 2: Distributed mode with 2 workers"
echo "----------------------------------------"
julia --project=. -t 1 src/batch_run.jl \
  --batch-folder "$TEST_DIR" \
  --data-folder "models/catalonia" \
  --workers 2 2>&1 | head -30

echo ""
echo "Test 3: Auto mode (should detect CPU count)"
echo "-------------------------------------------"
julia --project=. -t 1 src/batch_run.jl \
  --batch-folder "$TEST_DIR" \
  --data-folder "models/catalonia" \
  --workers auto 2>&1 | head -20

# Cleanup
rm -rf "$TEST_DIR"

echo ""
echo "============================================"
echo "Test complete!"
echo "============================================"
echo ""
echo "Note: These tests verify the script launches correctly."
echo "To test full simulation runs, use the actual synthetic pipeline:"
echo "  ./hpc/sbatch_single_node.sh --n-profiles 5 --n-jobs 2"
