#!/bin/bash

# A simple script to verify the docker image is working correctly
#
# 1. build the image
# 2. run the image with a test simulation
# 3. check the output
# 4. cleanup
#

set -e

IMAGE_NAME="episim-validation"
OUTPUT_DIR="test_output"

echo "Building docker image..."
docker build -t $IMAGE_NAME .

echo "Creating temporary output directory..."
mkdir -p $OUTPUT_DIR

echo "Running docker image with test simulation..."
# Mount the local models/mitma directory to /data in the container
# Mount the local test_output directory to /output in the container
docker run --rm \
    -v "$(pwd)/models/mitma:/data" \
    -v "$(pwd)/$OUTPUT_DIR:/output" \
    $IMAGE_NAME episim run -c /data/config_MMCACovid19.json -d /data -i /output

echo "Checking output..."
if [ -f "$OUTPUT_DIR/output/compartments_full.nc" ]; then
    echo "Docker image verified successfully. Output file found."
    # Clean up the output directory
    rm -rf $OUTPUT_DIR
    exit 0
else
    echo "Docker image verification failed. Output file not found."
    # Clean up the output directory
    rm -rf $OUTPUT_DIR
    exit 1
fi
