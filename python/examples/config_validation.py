#!/usr/bin/env python3
"""
Configuration validation and manipulation example

This script demonstrates how to validate and manipulate EpiSim configuration
files using the EpiSimConfig class.
"""

import os
import sys
import logging

# Add the parent directory to the path to import episim_python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episim_python import EpiSimConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_config_validation():
    """Demonstrate configuration validation"""

    # Paths relative to the EpiSim.jl root directory
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")

    # Configuration files
    config_files = [
        os.path.join(base_dir, "models", "mitma", "config_MMCACovid19.json"),
        os.path.join(base_dir, "models", "mitma", "config_MMCACovid19-vac.json"),
    ]

    for config_file in config_files:
        if not os.path.exists(config_file):
            logger.warning("Config file not found: %s", config_file)
            continue

        logger.info("Validating configuration: %s", os.path.basename(config_file))

        try:
            # Load and validate configuration
            config = EpiSimConfig.from_json(config_file)
            config.validate(verbose=True)

            logger.info("Configuration is valid")

            # Print detected group parameters
            logger.info("Detected group parameters:")
            for param_path, is_grouped in config.group_params.items():
                if is_grouped:
                    value = config.get_param(param_path)
                    logger.info("  %s: %s", param_path, value)

        except Exception as e:
            logger.error("Configuration validation failed: %s", str(e))


def demonstrate_config_manipulation():
    """Demonstrate configuration manipulation"""

    # Paths relative to the EpiSim.jl root directory
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    config_file = os.path.join(base_dir, "models", "mitma", "config_MMCACovid19.json")

    if not os.path.exists(config_file):
        logger.error("Config file not found: %s", config_file)
        return

    logger.info("Demonstrating configuration manipulation")

    # Load configuration
    config = EpiSimConfig.from_json(config_file)

    logger.info("Original configuration:")
    logger.info("  βᴵ: %s", config.get_param("epidemic_params.βᴵ"))
    logger.info("  γᵍ: %s", config.get_param("epidemic_params.γᵍ"))
    logger.info("  NPI κ₀s: %s", config.get_param("NPI.κ₀s"))

    # Example 1: Update scalar parameter
    logger.info("\\nUpdating scalar parameter βᴵ")
    config.update_param("epidemic_params.βᴵ", 0.12)
    logger.info("  New βᴵ: %s", config.get_param("epidemic_params.βᴵ"))

    # Example 2: Update group parameter for specific age group
    logger.info("\\nUpdating group parameter γᵍ for age group 'O' (old)")
    config.update_group_param("epidemic_params.γᵍ", "O", 0.10)
    logger.info("  New γᵍ: %s", config.get_param("epidemic_params.γᵍ"))

    # Example 3: Batch parameter updates
    logger.info("\\nBatch parameter updates")
    updates = {
        "epidemic_params.βᴬ": 0.055,
        "epidemic_params.scale_β": 0.6,
        "NPI.κ₀s": [0.4, 0.2, 0.1],
        "NPI.tᶜs": [30, 60, 90],
        "NPI.ϕs": [0.3, 0.15, 0.05],
    }

    config.inject(updates)

    logger.info("  New βᴬ: %s", config.get_param("epidemic_params.βᴬ"))
    logger.info("  New scale_β: %s", config.get_param("epidemic_params.scale_β"))
    logger.info("  New NPI κ₀s: %s", config.get_param("NPI.κ₀s"))
    logger.info("  New NPI tᶜs: %s", config.get_param("NPI.tᶜs"))
    logger.info("  New NPI ϕs: %s", config.get_param("NPI.ϕs"))

    # Example 4: Group parameter vector injection
    logger.info("\\nGroup parameter vector injection")
    age_specific_hospitalization = {
        "Y": 0.002,  # Young: 0.2%
        "M": 0.015,  # Middle: 1.5%
        "O": 0.090,  # Old: 9.0%
    }

    config.inject_group_vector("epidemic_params.γᵍ", age_specific_hospitalization)
    logger.info("  New γᵍ: %s", config.get_param("epidemic_params.γᵍ"))

    # Example 5: Save modified configuration
    output_file = os.path.join(base_dir, "runs", "modified_config.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    config.to_json(output_file)
    logger.info("\\nModified configuration saved to: %s", output_file)

    # Example 6: Reset to original configuration
    logger.info("\\nResetting to original configuration")
    config.reset()
    logger.info("  Reset βᴵ: %s", config.get_param("epidemic_params.βᴵ"))
    logger.info("  Reset γᵍ: %s", config.get_param("epidemic_params.γᵍ"))


def demonstrate_error_handling():
    """Demonstrate error handling in configuration manipulation"""

    # Paths relative to the EpiSim.jl root directory
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    config_file = os.path.join(base_dir, "models", "mitma", "config_MMCACovid19.json")

    if not os.path.exists(config_file):
        logger.error("Config file not found: %s", config_file)
        return

    logger.info("Demonstrating error handling")

    config = EpiSimConfig.from_json(config_file)

    # Example 1: Invalid parameter path
    try:
        config.update_param("invalid.parameter", 0.5)
    except Exception as e:
        logger.info("Expected error for invalid parameter: %s", str(e))

    # Example 2: Wrong data type for group parameter
    try:
        config.update_param("epidemic_params.γᵍ", 0.5)  # Should be list
    except Exception as e:
        logger.info("Expected error for wrong data type: %s", str(e))

    # Example 3: Invalid group label
    try:
        config.update_group_param("epidemic_params.γᵍ", "X", 0.5)  # Invalid group
    except Exception as e:
        logger.info("Expected error for invalid group label: %s", str(e))

    # Example 4: Wrong length for group parameter
    try:
        config.update_param("epidemic_params.γᵍ", [0.1, 0.2])  # Should be length 3
    except Exception as e:
        logger.info("Expected error for wrong length: %s", str(e))


def main():
    """Main function"""
    try:
        demonstrate_config_validation()
        print("\\n" + "=" * 60 + "\\n")
        demonstrate_config_manipulation()
        print("\\n" + "=" * 60 + "\\n")
        demonstrate_error_handling()

    except Exception as e:
        logger.error("Demonstration failed: %s", str(e))
        raise


if __name__ == "__main__":
    main()
