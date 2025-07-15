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
    update_params,
    compute_observables,
)

# Import schema validation with graceful fallback
try:
    from .schema_validator import (
        SchemaValidator,
        EpiSimSchemaValidator,
        validate_episim_config,
        validate_episim_config_safe,
    )
    SCHEMA_VALIDATION_AVAILABLE = True
    __all_schema__ = [
        "SchemaValidator",
        "EpiSimSchemaValidator", 
        "validate_episim_config",
        "validate_episim_config_safe",
    ]
except ImportError:
    SCHEMA_VALIDATION_AVAILABLE = False
    __all_schema__ = []

__version__ = "0.1.0"
__author__ = "EpiSim Development Team"

__all__ = [
    "EpiSim",
    "EpiSimConfig",
    "Metapopulation",
    "update_params",
    "compute_observables",
] + __all_schema__
