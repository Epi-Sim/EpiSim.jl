#!/usr/bin/env python3
"""
Parameter sensitivity analysis example

This script demonstrates how to run multiple simulations with different
parameter values to analyze sensitivity to key epidemiological parameters.
"""

import logging
import os
import sys

import numpy as np

# Add the parent directory to the path to import episim_python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episim_python import EpiSim, EpiSimConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_sensitivity_analysis():
    """Run parameter sensitivity analysis"""

    # Paths relative to the EpiSim.jl root directory
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")

    # Configuration file
    config_file = os.path.join(base_dir, "models", "mitma", "config_MMCACovid19.json")

    # Data and instance folders
    data_folder = os.path.join(base_dir, "models", "mitma")
    instance_folder = os.path.join(base_dir, "runs")

    # Initial conditions (optional)
    initial_conditions = os.path.join(
        base_dir,
        "models",
        "mitma",
        "initial_conditions_MMCACovid19.nc",
    )

    logger.info("Loading base configuration")

    # Load base configuration
    base_config = EpiSimConfig.from_json(config_file)
    base_config.validate()

    # Shorten simulation period for faster analysis
    base_config.update_param("simulation.start_date", "2020-02-09")
    base_config.update_param("simulation.end_date", "2020-03-31")

    # Parameter ranges for sensitivity analysis
    parameter_ranges = {
        "epidemic_params.βᴵ": np.linspace(0.05, 0.15, 5),  # Transmission rate
        "epidemic_params.scale_β": np.linspace(0.3, 0.7, 5),  # Asymptomatic scaling
        "NPI.κ₀s": [[x] for x in np.linspace(0.2, 0.8, 5)],  # Mobility reduction
    }

    results = {}

    for param_name, param_values in parameter_ranges.items():
        logger.info("Analyzing sensitivity to parameter: %s", param_name)
        results[param_name] = []

        for i, param_value in enumerate(param_values):
            logger.info(
                "Running simulation %d/%d for %s = %s",
                i + 1,
                len(param_values),
                param_name,
                param_value,
            )

            # Create a copy of the base configuration
            config = EpiSimConfig(base_config.config)

            # Update the parameter of interest
            config.update_param(param_name, param_value)

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

            try:
                # Run the simulation
                uuid, output = model.run_model()

                # Store results
                results[param_name].append(
                    {
                        "parameter_value": param_value,
                        "uuid": uuid,
                        "output_path": os.path.join(instance_folder, uuid, "output"),
                        "success": True,
                    },
                )

                logger.info("Simulation completed successfully. UUID: %s", uuid)

            except Exception as e:
                logger.error(
                    "Simulation failed for %s = %s: %s",
                    param_name,
                    param_value,
                    str(e),
                )
                results[param_name].append(
                    {
                        "parameter_value": param_value,
                        "uuid": None,
                        "output_path": None,
                        "success": False,
                        "error": str(e),
                    },
                )

    # Print summary
    logger.info("Sensitivity analysis completed")
    for param_name, param_results in results.items():
        successful_runs = sum(1 for r in param_results if r["success"])
        total_runs = len(param_results)
        logger.info(
            "%s: %d/%d simulations successful",
            param_name,
            successful_runs,
            total_runs,
        )

        if successful_runs > 0:
            logger.info("Successful runs for %s:", param_name)
            for result in param_results:
                if result["success"]:
                    logger.info(
                        "  %s = %s -> UUID: %s",
                        param_name,
                        result["parameter_value"],
                        result["uuid"],
                    )

    return results


def main():
    """Main function"""
    try:
        results = run_sensitivity_analysis()

        # You can add analysis of results here
        # For example, load the NetCDF output files and compare final outcomes

        logger.info(
            "Analysis complete. Results can be found in the respective UUID directories.",
        )

    except Exception as e:
        logger.error("Analysis failed: %s", str(e))
        raise


if __name__ == "__main__":
    main()
