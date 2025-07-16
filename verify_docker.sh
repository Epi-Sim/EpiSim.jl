#!/bin/bash

set -e

IMAGE_NAME="episim-validation"
OUTPUT_DIR="test_output"

echo "Building docker image..."
docker build -t $IMAGE_NAME .

echo "Creating temporary output directory..."
mkdir -p $OUTPUT_DIR

echo "Running Julia tests"
# Mount the local models/mitma directory to /data in the container
# Mount the local test_output directory to /output in the container
docker run --rm \
    -v "$(pwd)/models/mitma:/data" \
    -v "$(pwd)/$OUTPUT_DIR:/output" \
    $IMAGE_NAME julia --project test/runtests.jl

if [ $? -eq 0 ]; then
    echo "Julia tests PASSED"
else
    echo "Julia tests FAILED"
    exit 1
fi

echo "Running Python test suite..."
# Run the full pytest suite in the Docker container
docker run --rm \
    -v "$(pwd):/workspace" \
    -w /workspace \
    $IMAGE_NAME \
    bash -c "cd python && uv run pytest tests/ -v"

if [ $? -eq 0 ]; then
    echo "Python test suite passed successfully."
else
    echo "Python test suite failed."
    exit 1
fi

echo "ALL TESTS PASSED"

