# Mobility Configuration Guide

This document details how to configure time-varying mobility in the MMCACovid19Vac model.

## Overview

In the standard MMCA model, the mobility matrix $R_{ij}$ (fraction of people living in $i$ who visit $j$) is static. This extension allows $R_{ij}$ to vary at each timestep $t$, simulating the stochastic nature of daily human movement while maintaining physical consistency (population conservation).

## Configuration Parameters

These parameters are set in the `population_params` section of your JSON config file.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `mobility_variation_type` | String | `"none"` | The method for generating mobility variation. Options: `"none"`, `"ipfp_simple"`, `"external_netcdf"`. |
| `mobility_sigma_O` | Float | `0.0` | Noise level for total outflows (Origin strength). Used by `ipfp_simple` mode. |
| `mobility_sigma_D` | Float | `0.0` | Noise level for total inflows (Destination attraction). Used by `ipfp_simple` mode. |
| `mobility_rng_seed` | Int | `null` | Seed for reproducibility. Used by `ipfp_simple` mode. |
| `mobility_external_file` | String | `null` | Path to NetCDF file with mobility time series. Required for `external_netcdf` mode. |
| `mobility_external_variable` | String | `"mobility"` | Name of mobility variable in NetCDF file. |
| `mobility_validate` | Boolean | `true` | Whether to validate loaded mobility series. |
| `mobility_fill_missing` | Boolean | `false` | Whether to interpolate missing dates. |
| `mobility_time_alignment` | String | `"start"` | Date alignment mode: `"start"`, `"end"`, or `"nearest"`. |

## Mobility Variation Modes

### Mode 1: Static (`"none"`)
Mobility is static. The model uses the input `R_mobility_matrix.csv` for the entire simulation.

### Mode 2: Stochastic IPFP (`"ipfp_simple"`)
Enables the stochastic engine. At each timestep $t$, the model generates a perturbed matrix $R(t)$ based on the baseline matrix $R_{base}$.

**How it works:**
1. **Perturbation**: At each step $t$, we take the baseline flows $F_{ij} = N_i R_{ij}$ and calculate the baseline marginals (total outflows $O_i$ and total inflows $D_j$).
2. **Noise Injection**: We perturb these marginals:
   $$ O_i(t) = O_i \cdot \text{Lognormal}(0, \sigma_O) $$
   $$ D_j(t) = D_j \cdot \text{Lognormal}(0, \sigma_D) $$
3. **Consistency**: We adjust $D(t)$ slightly so that $\sum O(t) = \sum D(t)$ (global population conservation).
4. **Reconstruction (IPFP)**: We use the **Iterative Proportional Fitting Procedure (RAS Algorithm)** to find a new matrix $R_{ij}(t)$ that:
   - Matches the new noisy marginals $O(t)$ and $D(t)$.
   - Preserves the "structure" (zero/non-zero pattern and relative weights) of the baseline matrix.
5. **Update**: The model recalculates all density-dependent factors (effective population $\tilde{n}_i$, normalization $z^g$) using this new $R_{ij}(t)$ before computing infection probabilities for the day.

**Parameter guidance:**
- `mobility_sigma_O` & `mobility_sigma_D`: Control the variance of the lognormal noise applied to row sums (origins) and column sums (destinations). Higher values (e.g., `0.2` vs `0.05`) result in more erratic daily changes. The noise is unbiased (mean 1.0) so the *average* mobility over time remains close to the baseline.
- `mobility_rng_seed`: If provided, the sequence of random matrices will be identical across runs, allowing for reproducible debugging or sensitivity analysis.

### Mode 3: External NetCDF (`"external_netcdf"`)
Load pre-computed mobility time series from an external NetCDF file.

**NetCDF file format:**
```
Dimensions:
  date: T (timesteps)
  origin: M (patches)
  destination: M (patches)

Variables:
  mobility(date, origin, destination): Float64
    description: "Mobility flow probability"
    units: "probability"
  date: String
    format: "yyyy-mm-dd"
```

**Example configuration:**
```json
{
  "population_params": {
    "mobility_variation_type": "external_netcdf",
    "mobility_external_file": "data/mobility_timeseries.nc",
    "mobility_external_variable": "mobility",
    "mobility_validate": true,
    "mobility_fill_missing": false,
    "mobility_time_alignment": "start"
  }
}
```

**Creating NetCDF files:**
Use the `save_mobility_netcdf()` function to create properly formatted NetCDF files:

```julia
using MMCACovid19Vac
using Dates

# R_series: (T, L) array where T = timesteps, L = number of edges
# edgelist: (L, 2) array of [origin, destination] pairs
# dates: Vector of Date objects

save_mobility_netcdf(R_series, edgelist, dates, "output_mobility.nc")
```

**Date alignment modes:**
- `"start"`: Align NetCDF dates to start of simulation period
- `"end"`: Align to end of simulation period
- `"nearest"`: Use nearest available date

When external data is exhausted (beyond the available timesteps in the NetCDF file), the model automatically falls back to the baseline mobility matrix.

## Usage in Julia

### Loading External Mobility
```julia
using MMCACovid19Vac

# Configure external mobility
config = Dict(
    "mobility_variation_type" => "external_netcdf",
    "mobility_external_file" => "data/mobility.nc",
    "start_date" => Date(2020, 3, 1),
    "end_date" => Date(2020, 5, 1)
)

# Load mobility into population
population = load_external_mobility!(population, config; verbose=true)

# During simulation, mobility updates automatically
for t in 1:T
    update_mobility!(population, t)
    # ... run epidemic step
end
```

### Converting from Other Formats
If you have mobility data in sparse format (edges × timesteps), you may need to transpose it:

```julia
# If your data is (L, T) instead of (T, L):
R_series_TL = permutedims(R_sparse, (2, 1))  # Convert (L, T) to (T, L)
```

## Notes

* **Computational Cost**: The `ipfp_simple` mode adds a small computational overhead per timestep due to the IPFP matrix balancing step. The `external_netcdf` mode has minimal overhead.
* **ARM64 (Apple Silicon)**: The NetCDF implementation uses NCDatasets.jl which wraps HDF5 for better compatibility with Apple Silicon (M1/M2/M3/M4) processors.
