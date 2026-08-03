"""
EpiSim Python Interface

A Python wrapper for the EpiSim.jl epidemic simulation package.
Provides high-level interfaces for configuration management,
input file manipulation, and step-by-step simulation execution.
"""

from .epi_sim import EpiSim
from .episim_utils import (
    EpiSimConfig,
    Metapopulation,
    compute_observables,
    update_params,
)
from .schema_validator import (
    EpiSimSchemaValidator,
    SchemaValidator,
    validate_episim_config,
    validate_episim_config_safe,
)

__version__ = "0.1.0"
__author__ = "EpiSim Development Team"

__all__ = [
    "EpiSim",
    "EpiSimConfig",
    "EpiSimSchemaValidator",
    "Metapopulation",
    "SchemaValidator",
    "compute_observables",
    "update_params",
    "validate_episim_config",
    "validate_episim_config_safe",
]
