#!/usr/bin/env python3
"""
Basic EpiSim simulation example

This script demonstrates how to run a basic epidemic simulation using EpiSim.jl
from Python with the default configuration.
"""

import os
import sys
import json
import logging

# Add the parent directory to the path to import episim_python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episim_python import EpiSim

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run a basic simulation"""

    # Paths relative to the EpiSim.jl root directory
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")

    # Configuration file
    config_file = os.path.join(base_dir, "models", "mitma", "config_MMCACovid19.json")

    # Data and instance folders
    data_folder = os.path.join(base_dir, "models", "mitma")
    instance_folder = os.path.join(base_dir, "runs")

    # Initial conditions (optional)
    initial_conditions = os.path.join(
        base_dir, "models", "mitma", "initial_conditions_MMCACovid19.nc"
    )

    logger.info("Loading configuration from: %s", config_file)

    # Load configuration
    with open(config_file, "r") as f:
        config = json.load(f)

    logger.info("Initializing EpiSim model")

    # Initialize the model
    model = EpiSim(
        config=config,
        data_folder=data_folder,
        instance_folder=instance_folder,
        initial_conditions=initial_conditions
        if os.path.exists(initial_conditions)
        else None,
    )

    # Setup execution environment
    # Use interpreter mode for easier debugging
    model.setup(executable_type="interpreter")

    logger.info("Running simulation")

    # Run the simulation
    try:
        uuid, output = model.run_model()
        logger.info("Simulation completed successfully")
        logger.info("Model instance UUID: %s", uuid)
        logger.info(
            "Output saved to: %s", os.path.join(instance_folder, uuid, "output")
        )

    except Exception as e:
        logger.error("Simulation failed: %s", str(e))
        raise


if __name__ == "__main__":
    main()
