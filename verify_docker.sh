#!/bin/bash

# LOCAL test runner to build and run all tests inside the container.
# In CI, we separate build and testing into different steps using the runners.

set -e

# Configuration
IMAGE_NAME="${DOCKER_IMAGE_NAME:-episim-validation}"
OUTPUT_DIR="${TEST_OUTPUT_DIR:-test_output}"
CLEANUP_IMAGES="${CLEANUP_IMAGES:-true}"

# Function to cleanup Docker images
cleanup_docker() {
    if [ "$CLEANUP_IMAGES" = "true" ]; then
        echo "Cleaning up Docker images..."
        docker rmi $IMAGE_NAME 2>/dev/null || echo "Image $IMAGE_NAME not found for cleanup"
    fi
}

# Set up trap for cleanup on exit
trap cleanup_docker EXIT

echo "Building Docker image: $IMAGE_NAME"
if ! docker build -t $IMAGE_NAME .; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "Creating temporary output directory: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

echo "Running Julia tests..."
# Mount the local models/mitma directory to /data in the container
# Mount the local test_output directory to /output in the container
if docker run --rm \
    -v "$(pwd)/models/mitma:/data" \
    -v "$(pwd)/$OUTPUT_DIR:/output" \
    $IMAGE_NAME julia --project test/runtests.jl; then
    echo "✅ Julia tests PASSED"
else
    echo "❌ Julia tests FAILED"
    exit 1
fi

echo "Running Python test suite..."
# Run the full pytest suite in the Docker container
if docker run --rm \
    -v "$(pwd):/workspace" \
    -v "$(pwd)/models/mitma:/data" \
    -w /workspace \
    $IMAGE_NAME \
    bash -c "cd python && uv run pytest tests/ -v"; then
    echo "✅ Python test suite PASSED"
else
    echo "❌ Python test suite FAILED"
    exit 1
fi

echo "🎉 ALL TESTS PASSED"

