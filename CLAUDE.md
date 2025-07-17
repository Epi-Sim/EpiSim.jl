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

The `python/` directory contains a Python package (`episim_python`) that provides a high-level interface to EpiSim.jl.

### Installation
```bash
cd python
pip install -e .
```

### Usage
```python
from episim_python import EpiSim, EpiSimConfig
import json

# Load configuration
config = EpiSimConfig.from_json("config.json")
config.validate()

# Initialize model
model = EpiSim(config.config, data_folder, instance_folder)
model.setup(executable_type='interpreter')

# Run step-by-step
current_date = "2020-02-09"
new_state, next_date = model.step(current_date, length_days=7)
```

## Preferences and Recommendations

### Python Development
- Prefer `uv` for running python code eg. `uv run pytest`
- Use uv to run python commands with the venv eg. `uv run pytest`

## Input File Manipulation

EpiSim.jl uses multiple input file formats that can be manipulated both manually and programmatically.