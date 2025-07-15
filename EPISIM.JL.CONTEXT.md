# EpiSim.jl - Context Documentation

## Overview

EpiSim.jl is a Julia package for simulating epidemic spreading in metapopulations using different simulation engines/models. It implements MMCA (Microscopic Markov Chain Approach) for simulating extended SEIR models in metapopulations with different agent types representing age strata, connected through mobility networks.

## Project Structure

```
EpiSim.jl/
├── src/
│   ├── EpiSim.jl           # Main module file
│   ├── engine.jl           # Core simulation engine logic
│   ├── io.jl               # Input/output operations, NetCDF/HDF5 handling
│   ├── commands.jl         # Command-line interface functions
│   ├── common.jl           # Common utilities
│   ├── run.jl              # Simple run interface
│   ├── epi_sim.py          # Python wrapper
│   └── epi_sim/            # Python module directory
└── models/
    └── mitma/              # Example MITMA dataset
        ├── config_*.json   # Configuration files
        └── [data files]    # Required input data
```

## Supported Engines

1. **MMCACovid19** - Basic epidemic model without vaccination
2. **MMCACovid19Vac** - Extended model with vaccination capabilities

## Configuration System

### config.json Structure
```json
{
  "simulation": {
    "engine": "MMCACovid19|MMCACovid19Vac",
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD", 
    "save_full_output": boolean,
    "save_observables": boolean,
    "save_time_step": integer|null,
    "input_format": "csv|netcdf",
    "output_folder": "output",
    "output_format": "netcdf|hdf5"
  },
  "data": {
    "initial_condition_filename": "filename",
    "metapopulation_data_filename": "filename",
    "mobility_matrix_filename": "filename", 
    "kappa0_filename": "filename"
  },
  "epidemic_params": { /* engine-specific parameters */ },
  "population_params": { /* population structure parameters */ },
  "NPI": { /* non-pharmaceutical interventions */ },
  "vaccination": { /* vaccination parameters (MMCACovid19Vac only) */ }
}
```

## Required Data Files

All data files referenced in `config["data"]` must be present in the data directory specified when running simulations.

### 1. Initial Conditions

**Purpose**: Define the initial state of epidemic compartments

**Formats**:
- **CSV format** (`input_format: "csv"`):
  - File: `A0_initial_conditions_seeds.csv`
  - Columns: `name,id,idx,Y,M,O` (Y=Young, M=Middle, O=Old age groups)
  - Contains initial infected individuals by region and age group
  
- **NetCDF format** (`input_format: "netcdf"`):
  - File: `initial_conditions_MMCACovid19[-vac].nc`
  - Contains all compartments (S,E,A,I,PH,PD,HR,HD,R,D,CH) as multidimensional arrays
  - Dimensions: [G,M,V] for vaccination model, [G,M] for basic model

### 2. Metapopulation Data

**File**: `metapopulation_data.csv`
**Purpose**: Define population structure and demographics
**Columns**:
- `id`: Region identifier (string)
- `area`: Geographic area (km²)
- `Y,M,O`: Population by age group (Young, Middle, Old)
- `total`: Total population

### 3. Mobility Matrix

**File**: `R_mobility_matrix.csv` 
**Purpose**: Define daily mobility patterns between regions
**Structure**: CSV with mobility flows between region pairs
**Format**: Adjacency list format with columns for origin, destination, and flow strength

### 4. Mobility Reduction (Kappa0)

**File**: `kappa0_from_mitma.csv`
**Purpose**: Time-varying mobility reduction factors (e.g., due to lockdowns)
**Columns**:
- `date`: Date (YYYY-MM-DD)
- `reduction`: Mobility reduction factor (0-1, where 1 = no reduction)
- `datetime`: Date timestamp
- `time`: Time step number

### 5. Optional: Translation Files

**File**: `rosetta.csv` (optional)
**Purpose**: Maps region IDs to sequential indices
**Columns**: `id,idx`

## Epidemic Compartments

The model uses 11 compartments:
- **S**: Susceptible
- **E**: Exposed  
- **A**: Asymptomatic
- **I**: Infected (symptomatic)
- **PH**: Pre-hospitalized
- **PD**: Pre-deceased
- **HR**: Hospitalized (recovering)
- **HD**: Hospitalized (deteriorating)
- **R**: Recovered
- **D**: Dead
- **CH**: Confined

## Age Groups (G dimension)

Standard configuration uses 3 age strata:
- **Y**: Young (typically 0-39 years)
- **M**: Middle (typically 40-64 years)  
- **O**: Old (typically 65+ years)

## Vaccination States (V dimension, MMCACovid19Vac only)

- **NV**: Non-vaccinated
- **V**: Vaccinated
- **PV**: Post-vaccination (residual immunity)

## Key Parameters

### Epidemic Parameters
- **β**: Transmission rates (βᴬ for asymptomatic, βᴵ for symptomatic)
- **η,α,μ,θ,γ,ζ,λ,ω,ψ,χ**: Age-stratified transition rates between compartments
- **scale_β**: Global transmission scaling factor

### Population Parameters  
- **C**: Contact matrix between age groups
- **k**: Average contacts per age group (total, home, work)
- **p**: Mobility propensity by age group
- **ξ**: Population density factor
- **σ**: Average household size

### Non-Pharmaceutical Interventions (NPIs)
- **κ₀s**: Confinement levels
- **ϕs**: Household permeability during confinement
- **δs**: Social distancing factors
- **tᶜs**: Implementation time steps

## Output Files

### NetCDF Outputs
- `compartments_full.nc`: Complete time series of all compartments
- `compartments_t_DATE.nc`: Snapshot at specific time step
- `observables.nc`: Derived quantities (new infections, hospitalizations, deaths, R_eff)

### Dimensions in Output Files
- **G**: Age strata
- **M**: Metapopulation regions
- **T**: Time steps
- **V**: Vaccination status (MMCACovid19Vac only)

## Data Processing Pipeline

1. **Configuration Loading**: Parse config.json
2. **Data File Reading**: Load all required CSV/NetCDF files
3. **Structure Initialization**: Create population and epidemic parameter structures
4. **Compartment Initialization**: Set initial conditions from data
5. **Simulation Execution**: Run MMCA epidemic spreading
6. **Output Generation**: Save results in specified format

## Command Line Usage

```bash
# Run simulation
episim run -c config.json -d data_folder -i output_folder

# Generate initial conditions  
episim init -c config.json -d data_folder -s seeds.csv -o initial_conditions.nc

# Create config template
episim template -e MMCACovid19 -o config_template.json
```

## Data Validation Requirements

- Region IDs must be consistent across all data files
- Population totals should match between metapopulation_data.csv and initial conditions
- Mobility matrix must reference valid region IDs
- Date ranges in kappa0 file should cover simulation period
- Age group labels must match between config and data files
- NetCDF initial conditions must contain all required compartments

## Common Data Issues

1. **Missing regions**: Mobility matrix references non-existent regions
2. **Date mismatches**: Simulation dates outside kappa0 time range  
3. **Format inconsistencies**: Mixed string/numeric region IDs
4. **Scale problems**: Population numbers not matching expected magnitudes
5. **Missing compartments**: NetCDF files missing required variables

## Example MITMA Dataset

The `models/mitma/` directory contains a complete example dataset for Spanish regions:
- 2,851 geographic units
- 3 age groups (Young, Middle, Old)
- Mobility data from Spanish MITMA transport authority
- COVID-19 lockdown mobility reductions (March-September 2020)
- Both basic and vaccination model configurations 