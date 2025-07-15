#!/usr/bin/env python3
"""
EXPERIMENTAL: Step-by-step simulation with dynamic parameter updates

WARNING: This example uses the experimental step() method which is not officially supported.
This is intended for RL agent integration but is untested and may not work correctly.
Use basic_simulation.py for reliable single simulation execution.

This script demonstrates how to run a simulation in discrete time steps
with dynamic parameter updates based on simulation progress.
"""

import os
import sys
import logging

# Add the parent directory to the path to import episim_python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episim_python import EpiSim, EpiSimConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run step-by-step simulation with policy updates"""

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

    logger.info("Loading and validating configuration")

    # Load configuration using EpiSimConfig for easier manipulation
    config = EpiSimConfig.from_json(config_file)
    config.validate()

    # Update simulation dates for shorter demonstration
    config.update_param("simulation.start_date", "2020-02-09")
    config.update_param("simulation.end_date", "2020-04-30")

    logger.info("Initializing EpiSim model")

    # Initialize the model
    model = EpiSim(
        config=config.config,
        data_folder=data_folder,
        instance_folder=instance_folder,
        initial_conditions=initial_conditions
        if os.path.exists(initial_conditions)
        else None,
    )

    # Setup execution environment
    model.setup(executable_type="interpreter")

    logger.info("Starting step-by-step simulation")

    # Simulation parameters
    start_date = "2020-02-09"
    current_date = start_date
    step_days = 7  # One week steps
    max_steps = 10  # Maximum number of steps

    for step in range(max_steps):
        logger.info(
            "Step %d: Running simulation from %s for %d days",
            step + 1,
            current_date,
            step_days,
        )

        try:
            # Run one step
            new_state, next_date = model.step(current_date, step_days)

            logger.info("Step %d completed. Next date: %s", step + 1, next_date)

            # Update policy based on simulation progress
            if step == 3:  # Week 4: Implement lockdown
                logger.info("Implementing lockdown measures")
                config.update_param("NPI.κ₀s", [0.3])  # Reduce mobility to 30%
                config.update_param("NPI.ϕs", [0.1])  # Reduce household permeability
                config.update_param("NPI.δs", [0.5])  # Implement social distancing
                config.update_param("NPI.tᶜs", [step * step_days])  # Start intervention

            elif step == 6:  # Week 7: Relax some measures
                logger.info("Relaxing some measures")
                config.update_param("NPI.κ₀s", [0.6])  # Increase mobility to 60%
                config.update_param("NPI.ϕs", [0.2])  # Increase household permeability

            elif step == 8:  # Week 9: Implement stricter measures
                logger.info("Implementing stricter measures")
                config.update_param("NPI.κ₀s", [0.1])  # Very strict mobility reduction
                config.update_param("NPI.δs", [0.3])  # Stricter social distancing

            # Apply updated configuration
            model.update_config(config.config)

            # Move to next step
            current_date = next_date

        except Exception as e:
            logger.error("Step %d failed: %s", step + 1, str(e))
            break

    logger.info("Step-by-step simulation completed")
    logger.info("Model instance UUID: %s", model.uuid)
    logger.info(
        "Output saved to: %s", os.path.join(instance_folder, model.uuid, "output")
    )


if __name__ == "__main__":
    main()
