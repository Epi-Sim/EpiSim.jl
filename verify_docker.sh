#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
IMAGE_NAME="meprecisa/episim"
TEST_OUTPUT_DIR="test_output"
CONFIG_FILE="models/mitma/config_MMCACovid19.json"
DATA_DIR="models/mitma"
EXPECTED_OUTPUT_FILE="${TEST_OUTPUT_DIR}/output/observables.nc"

# --- Main Script ---
echo "--- Starting Docker Image Verification ---"

# 1. Build the Docker image
# echo "Building Docker image '$IMAGE_NAME'..."
# docker build -t ${IMAGE_NAME} .

# 2. Create a clean directory for test output
echo "Creating clean test output directory: '${TEST_OUTPUT_DIR}'..."
rm -rf "${TEST_OUTPUT_DIR}"
mkdir -p "${TEST_OUTPUT_DIR}"

# 3. Run the simulation inside the container
echo "Running simulation via Docker..."
docker run --rm \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/${TEST_OUTPUT_DIR}:/app/${TEST_OUTPUT_DIR}" \
  "${IMAGE_NAME}" \
  episim run -c "/app/${CONFIG_FILE}" -d "/app/${DATA_DIR}" -i "/app/${TEST_OUTPUT_DIR}"

# 4. Verify that the output file was created
echo "Verifying simulation output..."
if [ -f "${EXPECTED_OUTPUT_FILE}" ]; then
    echo "✅ SUCCESS: Expected output file '${EXPECTED_OUTPUT_FILE}' was created."
else
    echo "❌ FAILURE: Expected output file was not found at '${EXPECTED_OUTPUT_FILE}'."
    exit 1
fi

# 5. Clean up the test directory
echo "Cleaning up test directory..."
rm -rf "${TEST_OUTPUT_DIR}"

echo "--- Verification Complete. The image works as expected. ---"
