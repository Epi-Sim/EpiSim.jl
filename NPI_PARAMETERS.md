# NPI Parameters Documentation

This document describes the Non-Pharmaceutical Intervention (NPI) parameter system, including the κ₀ (mobility reduction) parameter formats and the `init_NPI_parameters_struct` function contract.

---

## κ₀ (Mobility Reduction) Parameter

### Semantics

- **κ₀ = 0.0**: No confinement (100% mobility preserved)
- **κ₀ = 0.5**: 50% confinement (50% mobility reduction)
- **κ₀ = 1.0**: Full confinement (0% mobility, household-only)

**Key relationship**: As κ₀ ↑ → mobility ↓ → infections ↓

### Supported Formats

Both formats support time-series application, but they work differently:

---

### Format 1: JSON Config (No CSV)

**Use case**: Manual control of intervention events at specific timesteps

```json
{
  "data": {
    "kappa0_filename": null
  },
  "NPI": {
    "κ₀s": [0.8, 0.5, 0.0],
    "ϕs":  [0.2, 0.1, 0.0],
    "δs":  [0.8, 0.5, 0.0],
    "tᶜs": [50, 100, 150],
    "are_there_npi": true
  }
}
```

**How it works**:
- `tᶜs` = timesteps when interventions change (e.g., day 50, 100, 150)
- `κ₀s` = confinement level at each timestep
- `ϕs` = household permeability at each timestep
- `δs` = social distancing factor at each timestep
- All arrays must have the **same length**
- Values persist until the next timestep

**Example interpretation**:
| Day (t) | κ₀ | Meaning |
|----------|-----|---------|
| 1-49 | 0.0 | No confinement (default) |
| 50-99 | 0.8 | 80% confinement |
| 100-149 | 0.5 | 50% confinement (relaxed) |
| 150+ | 0.0 | No confinement (ended) |

---

### Format 2: CSV Time-Series + JSON

**Use case**: External mobility data (e.g., Google Mobility, Apple Mobility) with daily values

**Config**:
```json
{
  "data": {
    "kappa0_filename": "kappa0.csv"
  },
  "NPI": {
    "κ₀s": [0.0],
    "ϕs":  [0.2],
    "δs":  [0.8]
  }
}
```

**CSV format** (`kappa0.csv`):
```csv
date,reduction,datetime,time
2020-03-16,0.505,2020-03-16,36
2020-03-17,0.528,2020-03-17,37
2020-03-18,0.533,2020-03-18,38
2020-03-19,0.573,2020-03-19,39
...
```

**How it works**:
- Each row is a day with its own κ₀ value
- The `time` column is automatically synchronized to simulation dates
- `ϕs` and `δs` from JSON are applied as **constant values** across all timesteps
- When `κ₀ = 0.0`, `δ` is automatically set to `0.0` (no social distancing when no confinement)

---

### Key Differences

| Feature | JSON Only | CSV + JSON |
|---------|-----------|------------|
| **κ₀ values** | Manual array | Read from `reduction` column |
| **timesteps** | Explicit `tᶜs` | Auto-generated from dates |
| **ϕs, δs** | Per-timestep arrays | Single value, broadcast to all days |
| **Best for** | Scenario planning, what-if | Real-world mobility data |
| **Data source** | User-defined | External (Google/Apple mobility) |

---

## `init_NPI_parameters_struct` Function Contract

### Overview

Two overloaded methods that initialize NPI (Non-Pharmaceutical Intervention) parameters from either a JSON config or a CSV time-series file.

---

### Method 1: High-Level Interface (Recommended)

```julia
function init_NPI_parameters_struct(
    data_path::String,
    npi_params_dict::Dict,
    kappa0_filename::Union{String, Nothing},
    first_day::Date
)::NPI_Params
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `data_path` | `String` | Directory path containing the kappa0 CSV file (if used) |
| `npi_params_dict` | `Dict` | JSON config parsed as dict, must contain keys: `"κ₀s"`, `"ϕs"`, `"δs"`, `"tᶜs"` |
| `kappa0_filename` | `String \| Nothing` | Relative path to CSV time-series file, or `nothing` for JSON-only |
| `first_day` | `Date` | Simulation start date (for CSV date synchronization) |

**Returns:** `NPI_Params` struct with fields `(κ₀s, ϕs, δs, tᶜs)`

**Behavior:**
- If `kappa0_filename === nothing`: Uses JSON arrays directly
- If `kappa0_filename` is a string: Loads CSV and synchronizes dates to simulation timesteps

---

### Method 2: Low-Level Interface

```julia
function init_NPI_parameters_struct(
    κ₀_df::Union{DataFrame, Nothing},
    npi_params_dict::Dict,
    first_day::Date
)::NPI_Params
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `κ₀_df` | `DataFrame \| Nothing` | Pre-loaded kappa0 DataFrame with columns `date`, `reduction`, `time`, or `nothing` |
| `npi_params_dict` | `Dict` | JSON config dict (must contain `"ϕs"`, `"δs"` at minimum) |
| `first_day` | `Date` | Simulation start date for time index calculation |

**Returns:** `NPI_Params` struct

---

### NPI_Params Struct

```julia
struct NPI_Params
    κ₀s::Array{Float64, 1}  # Confinement levels (0=none, 1=full)
    ϕs::Array{Float64, 1}  # Household permeabilities (0=impermeable, 1=permeable)
    δs::Array{Float64, 1}  # Social distancing factors (0=none, 1=full distancing)
    tᶜs::Array{Int64, 1}   # Timesteps when interventions change
end
```

**Contract:** All four arrays must have the same length `n`, representing `n` intervention change points.

---

### Usage Examples

**Example 1: JSON-only (manual timesteps)**
```julia
using MMCACovid19Vac
using JSON

config = JSON.parsefile("config.json")
npi = init_NPI_parameters_struct(
    "data/",
    config["NPI"],
    nothing,              # No CSV
    Date("2020-03-01")
)
# Returns: NPI_Params([0.8], [0.2], [0.8], [50])
```

**Example 2: CSV time-series**
```julia
using MMCACovid19Vac
using JSON

config = JSON.parsefile("config.json")
npi = init_NPI_parameters_struct(
    "data/",
    config["NPI"],
    "kappa0.csv",         # Load from file
    Date("2020-03-01")
)
# Returns: NPI_Params with daily κ₀ values synchronized to dates
```

---

### Edge Cases & Special Behavior

1. **Pre-simulation dates**: CSV entries with dates before `first_day` are removed with a warning
2. **δ override**: When using CSV, `δs[κ₀s .== 0.] .= 0.0` sets social distancing to 0 when confinement is 0
3. **Constant ϕ/δ**: CSV mode broadcasts the first value from JSON config across all timesteps

---

### Export Status

**Yes** - This function is exported via the module export list:

```julia
include("utils.jl")
export init_NPI_parameters_struct
```

So calling libraries can use:
```julia
using MMCACovid19Vac

npi = init_NPI_parameters_struct(...)
```

---

### Code Reference

The branching logic at `src/utils.jl:610-646`:

```julia
if isnothing(κ₀_df)
    # JSON path: read arrays directly
    tᶜs = npi_params_dict["tᶜs"]
    κ₀s = npi_params_dict["κ₀s"]
    ϕs = npi_params_dict["ϕs"]
    δs = npi_params_dict["δs"]
else
    # CSV path: read reduction, broadcast ϕ/δ
    tᶜs = κ₀_df.time[:]
    κ₀s = κ₀_df.reduction[:]
    ϕs = fill(ϕs_aux[1], length(tᶜs))  # constant
    δs = fill(δs_aux[1], length(tᶜs))  # constant
    δs[κ₀s .== 0.] .= 0.0              # special case
end

return NPI_Params(κ₀s, ϕs, δs, tᶜs)
```
