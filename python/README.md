# EpiSim Python Interface

A Python wrapper for the EpiSim.jl epidemic simulation package that provides high-level interfaces for configuration management, input file manipulation, and step-by-step simulation execution.

## Features

- **Step-by-step simulation**: Execute simulations in discrete time steps with dynamic parameter updates
- **Configuration management**: Comprehensive configuration validation and manipulation
- **Input file handling**: Support for NetCDF, CSV, and JSON input formats
- **Metapopulation support**: Handle complex metapopulation structures with aggregation
- **Multiple engines**: Support for both MMCACovid19 and MMCACovid19Vac engines

## Installation

### Prerequisites

1. **Julia**: Install Julia (>= 1.6) from [julialang.org](https://julialang.org/)
2. **EpiSim.jl**: Install and compile the EpiSim.jl package (see parent directory)

### Install Python Package

```bash
# From the python directory
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

## Quick Start

```python
import json
from episim_python import EpiSim, EpiSimConfig

# Load configuration
with open("../models/mitma/config_MMCACovid19.json", 'r') as f:
    config = json.load(f)

# Initialize model
model = EpiSim(
    config=config,
    data_folder="../models/mitma",
    instance_folder="../runs",
    initial_conditions="../models/mitma/initial_conditions.nc"
)

# Setup execution (compiled or interpreter)
model.setup(executable_type='interpreter')

# Run step-by-step simulation
current_date = "2020-02-09"
for i in range(10):
    new_state, next_date = model.step(current_date, length_days=7)
    
    # Update parameters dynamically
    config["NPI"]["κ₀s"] = [config["NPI"]["κ₀s"][0] * 0.95]
    model.update_config(config)
    
    current_date = next_date
```

## Main Classes

### EpiSim
Main wrapper class for step-by-step simulation execution.

### EpiSimConfig
Configuration management with validation and group parameter handling.

### Metapopulation
Handles metapopulation data with aggregation capabilities.

## Configuration Management

```python
from episim_python import EpiSimConfig

# Load and validate configuration
config = EpiSimConfig.from_json("config.json")
config.validate()

# Update parameters
config.update_param("epidemic_params.βᴵ", 0.1)
config.update_group_param("epidemic_params.γᵍ", "Y", 0.005)

# Batch updates
config.inject({
    "epidemic_params.βᴵ": 0.1,
    "NPI.κ₀s": [0.8]
})
```

## Input File Formats

### Configuration (JSON)
Standard EpiSim.jl configuration format with sections for simulation, data, epidemic parameters, population parameters, and NPI.

### Initial Conditions (NetCDF)
Multi-dimensional arrays representing compartment populations by age group, metapopulation, and vaccination status.

### Metapopulation Data (CSV)
Population data by region and age group with area information.

### Mobility Matrix (CSV)
Connectivity matrix between metapopulations.

## Examples

See the `examples/` directory for:
- Basic simulation setup
- Parameter sensitivity analysis
- Step-by-step execution with policy updates
- Configuration validation and manipulation

## Dependencies

- numpy (>= 1.19.0)
- pandas (>= 1.2.0)
- xarray (>= 0.16.0)
- netcdf4 (>= 1.5.0)

## License

MIT License - see parent directory for details.