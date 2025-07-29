# EpiSim Python Examples

This directory contains example scripts demonstrating different aspects of the EpiSim Python interface.

## Prerequisites

1. **Julia Environment**: Ensure Julia is installed and EpiSim.jl is properly set up
2. **Python Environment**: Install the episim_python package:
   ```bash
   cd ../
   pip install -e .
   ```

## Examples

### 1. Basic Simulation (`basic_simulation.py`)

Demonstrates the simplest way to run an EpiSim simulation from Python.

```bash
python basic_simulation.py
```

**Key features:**
- Load configuration from JSON file
- Initialize EpiSim model
- Run complete simulation
- Handle output

### 2. Step-by-step Simulation (`step_by_step_simulation.py`)

Shows how to run simulations in discrete time steps with dynamic parameter updates.

```bash
python step_by_step_simulation.py
```

**Key features:**
- Step-by-step execution
- Dynamic policy updates based on simulation progress
- Intervention scheduling (lockdowns, relaxations)
- Configuration updates between steps

### 3. Parameter Sensitivity Analysis (`parameter_sensitivity.py`)

Demonstrates how to run multiple simulations with different parameter values for sensitivity analysis.

```bash
python parameter_sensitivity.py
```

**Key features:**
- Multiple simulation runs with varying parameters
- Systematic parameter space exploration
- Result organization and tracking
- Error handling for failed simulations

### 4. Configuration Validation (`config_validation.py`)

Shows comprehensive configuration file validation and manipulation techniques.

```bash
python config_validation.py
```

**Key features:**
- Configuration file validation
- Parameter type detection (scalar vs. group parameters)
- Individual parameter updates
- Batch parameter updates
- Group-specific parameter manipulation
- Error handling and validation

## Common Usage Patterns

### Loading and Validating Configuration

```python
from episim_python import EpiSimConfig

# Load configuration
config = EpiSimConfig.from_json("config.json")

# Validate configuration
config.validate(verbose=True)

# Access parameters
beta_I = config.get_param("epidemic_params.βᴵ")
gamma_g = config.get_param("epidemic_params.γᵍ")
```

### Running Simulations

```python
from episim_python import EpiSim

# Initialize model
model = EpiSim(config.config, data_folder, instance_folder)

# Setup execution
model.setup(executable_type='interpreter')

# Run simulation
uuid, output = model.run_model()
```

### Step-by-step Execution

```python
# Run one week at a time
current_date = "2020-02-09"
new_state, next_date = model.step(current_date, length_days=7)

# Update parameters
config.update_param("NPI.κ₀s", [0.5])
model.update_config(config.config)

# Continue simulation
current_date = next_date
```

### Parameter Updates

```python
# Scalar parameter
config.update_param("epidemic_params.βᴵ", 0.1)

# Group parameter for specific age group
config.update_group_param("epidemic_params.γᵍ", "O", 0.08)

# Batch updates
config.inject({
    "epidemic_params.βᴵ": 0.1,
    "NPI.κ₀s": [0.8, 0.6, 0.4],
    "NPI.tᶜs": [30, 60, 90]
})
```

## File Structure

```
examples/
├── README.md                     # This file
├── basic_simulation.py           # Basic simulation example
├── step_by_step_simulation.py    # Step-by-step execution
├── parameter_sensitivity.py      # Sensitivity analysis
└── config_validation.py          # Configuration manipulation
```

## Output

All examples create output in the `../../runs/` directory with unique UUID identifiers. Each simulation creates:

- `output/compartments_full.nc`: Complete simulation results
- `output/observables.nc`: Computed observables (if enabled)
- `output/compartments_t_*.nc`: Time-specific snapshots (if enabled)

## Troubleshooting

### Common Issues

1. **Julia not found**: Ensure Julia is in your PATH
2. **EpiSim.jl not compiled**: Run `julia install.jl` in the parent directory
3. **Missing initial conditions**: Some examples require initial condition files
4. **Permission errors**: Ensure write permissions to the runs directory

### Debug Mode

Enable debug logging by modifying the logging level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Using Compiled Executable

For better performance, use the compiled executable:

```python
model.setup(executable_type='compiled', executable_path='../../episim')
```

## Next Steps

After running these examples, you can:

1. Modify parameters to explore different scenarios
2. Implement custom analysis of output NetCDF files
3. Create reinforcement learning environments using the step-by-step interface
4. Develop automated parameter calibration workflows
5. Build web interfaces or dashboards using the Python API