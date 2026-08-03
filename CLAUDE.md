# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EpiSim.jl is a Julia package for simulating epidemic spreading in metapopulations using different simulation engines. It provides a command-line interface for running epidemic simulations with standardized JSON configuration files and NetCDF output format.

## Common Commands

### Installation and Setup

```bash
# Install dependencies and optionally compile
julia ./install.jl -c -i     # Compile with incremental compilation
julia ./install.jl -c        # Compile without incremental compilation  
julia ./install.jl           # Install without compilation (creates wrapper script)
```

### Running Simulations

```bash
# Run simulation with config file
./episim run -c models/mitma/config_MMCACovid19.json -d models/mitma -i runs

# Run with custom dates
./episim run -c config.json -d data_folder -i instance_folder --start-date 2020-03-01 --end-date 2020-06-01

# Export compartments at specific time
./episim run -c config.json -d data_folder -i instance_folder --export-compartments-time-t 50
```

### Model Setup and Initialization

```bash
# Create model template
./episim setup -n model_name -M 10 -G 3 -e MMCACovid19 -o models

# Initialize simulation with seeds
./episim init -c config.json -d data_folder --seeds seeds.csv -o initial_conditions.nc
```

### Testing

```bash
# Run test suite
julia --project=. test/runtests.jl

# Or using Julia's built-in test runner
julia -e "using Pkg; Pkg.test()"
```

## Architecture

### Core Components

- **src/EpiSim.jl**: Main module with entry points and command-line interface
- **src/commands.jl**: Command-line argument parsing and command execution functions
- **src/engine.jl**: Engine abstraction layer supporting multiple simulation engines
- **src/io.jl**: Input/output functions for NetCDF and CSV formats
- **src/run.jl**: Entry point for non-compiled execution

### Supported Engines

1. **MMCACovid19**: Basic SEIR model with age stratification
2. **MMCACovid19Vac**: Extended model with vaccination support

Each engine has its own parameter structure and validation requirements.

### Key Data Structures

- **Population_Params**: Metapopulation structure, contact matrices, mobility networks
- **Epidemic_Params**: Disease parameters (transition rates, infectivity, etc.)
- **NPI_Params**: Non-pharmaceutical intervention parameters
- **Vaccination_Params**: Vaccination strategy parameters (MMCACovid19Vac only)

### Configuration Format

The system uses JSON configuration files with these sections:

- `simulation`: Engine type, dates, output settings
- `data`: File paths for input data
- `epidemic_params`: Disease-specific parameters
- `population_params`: Population structure and contact patterns
- `NPI`: Non-pharmaceutical intervention parameters
- `vaccination`: Vaccination parameters (MMCACovid19Vac only)

### Input/Output

- **Input formats**: NetCDF for initial conditions, CSV for seeds and data files
- **Output format**: NetCDF with multi-dimensional arrays for compartments and observables
- **Data structure**: Time × Age Groups × Metapopulations × (Vaccination Status)

## Development Notes

### Code Organization

- Engine-specific functions use multiple dispatch on engine types
- Configuration validation is engine-specific
- I/O operations are centralized in `io.jl`
- All simulation engines follow the same interface pattern

### Key Functions

- `execute_run()`: Main simulation execution
- `validate_config()`: Engine-specific config validation
- `run_engine_io()`: File-based simulation runner
- `init_*_struct()`: Parameter structure initialization
- `set_compartments!()`: Initial condition setup

### Dependencies

- Core Julia packages: DataFrames, CSV, JSON, NetCDF, NCDatasets
- Engine packages: MMCAcovid19, MMCACovid19Vac
- Build tools: PackageCompiler for standalone executable creation

### Testing

Tests are located in `test/runtests.jl` and run simulations with both engines using the MITMA model configuration.

## Python Interface

The `python/` directory contains a Python package (`episim_python`) that provides a high-level interface to EpiSim.jl for step-by-step simulation execution, configuration management, and input file manipulation.

### Features

- **Step-by-step simulation**: Execute simulations in discrete time steps with dynamic parameter updates
- **Configuration management**: Comprehensive configuration validation and manipulation
- **Input file handling**: Support for NetCDF, CSV, and JSON input formats
- **Metapopulation support**: Handle complex metapopulation structures with aggregation
- **Multiple engines**: Support for both MMCACovid19 and MMCACovid19Vac engines

### Installation

```bash
cd python
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Usage

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

### Configuration Management

The Python interface provides comprehensive configuration management utilities through the `EpiSimConfig` class and `update_params()` function. For complete documentation, see [Python Configuration Utilities Guide](docs/PYTHON_CONFIG_UTILS.md).

```python
from episim_python import EpiSimConfig
from episim_python.episim_utils import update_params

# High-level interface with type safety and validation
config = EpiSimConfig.from_json("config.json")
config.validate()

# Update parameters with automatic type checking
config.update_param("epidemic_params.βᴵ", 0.1)
config.update_group_param("epidemic_params.γᵍ", "Y", 0.005)

# Batch updates
config.inject({
    "epidemic_params.βᴵ": 0.1,
    "NPI.κ₀s": [0.8]
})

# Low-level interface with parameter aliases and conversions
base_config = {"epidemic_params": {}, "NPI": {}}
updated_config = update_params(base_config, {
    "β": 0.12,           # Alias for βᴵ
    "τ_inc": 5.2,        # Incubation period -> ηᵍ
    "scale_ea": 0.4,     # E->A fraction
    "ϕs": 0.3            # Contact reduction
})
```

#### Key Configuration Utilities

- **Parameter Aliases**: Use intuitive names like `β` instead of `βᴵ`
- **Automatic Conversions**: Convert epidemiological timescales to rates
- **Group Parameter Handling**: Type-safe updates for age-stratified parameters
- **Schema Validation**: Comprehensive configuration validation
- **Complex Parameter Derivation**: Compute interdependent parameters automatically

### Input File Resolution

The Python interface handles file paths as follows:

1. Configuration file paths are relative to the data folder
2. If a file is not found in the data folder, it searches the instance folder
3. Absolute paths in the configuration are used as-is

Required input files in the data folder:

- **`Metapopulation_data.csv`**: Population data by region and age group
- **`Mobility_Network.csv`**: Connectivity matrix between metapopulations  
- **`Contact_Matrices_data.csv`**: Age-stratified contact patterns

Optional files:

- **Initial conditions**: NetCDF file with compartment populations
- **Seeds file**: CSV with initial infection seeds by location

### Output Structure

All outputs are written to: `<instance_folder>/<output_folder>/`

Primary output files:

- **`episim_output.nc`**: Main NetCDF file containing all compartments and observables
- **`observables.nc`**: Observable quantities only (when `save_full_output: false`)

NetCDF structure includes:

- **Compartments**: Population in each disease state by time, age group, metapopulation
- **Observables**: Daily new infections, hospitalizations, deaths, mobility flow

### Environment Variables

The Python interface supports these environment variables:

- **`JULIA_PROJECT`**: Sets the Julia project directory (equivalent to `--project` flag)
- **`EPISIM_EXECUTABLE_PATH`**: Path to compiled EpiSim executable (for compiled mode)

Example:

```bash
export JULIA_PROJECT=/path/to/EpiSim.jl
export EPISIM_EXECUTABLE_PATH=/path/to/compiled/episim
python your_script.py
```

### Dependencies

- numpy (>= 1.19.0)
- pandas (>= 1.2.0)  
- xarray (>= 0.16.0)
- netcdf4 (>= 1.5.0)

## Preferences and Recommendations

### Python Development

- Prefer `uv` for running python code eg. `uv run pytest`
- Use uv to run python commands with the venv eg. `uv run pytest`

## Input File Manipulation

EpiSim.jl uses multiple input file formats that can be manipulated both manually and programmatically.

