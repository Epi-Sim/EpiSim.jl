# EpiSim.jl Examples

This directory contains example data files demonstrating expected input formats for EpiSim.jl and related MMCA epidemic models.

## Mobility Time-Varying Example

### File: `mobility_time_varying_example.nc`

This file demonstrates the expected NetCDF structure for time-varying mobility data to be used for external mobility support in MMCA epidemic models. Uses MMCA-compatible dense format.

#### Purpose

This example file is provided to MMCA repository maintainers as a reference implementation for:
- Understanding the dense NetCDF format for time-varying mobility
- Implementing external mobility input support
- Testing mobility data parsing and validation

#### File Structure

```
mobility_time_varying_example.nc
├── Dimensions:
│   ├── date: 10
│   ├── origin: 3
│   └── destination: 3
│
├── Coordinates:
│   ├── date: ["2020-03-01", "2020-03-02", ..., "2020-03-10"]
│   ├── origin: ["08001", "08002", "08003"] - Municipality IDs
│   └── destination: ["08001", "08002", "08003"] - Municipality IDs
│
├── Data Variables:
│   └── mobility: (date, origin, destination) - Float64
│       ├── description: "Mobility flow probability"
│       └── units: "probability"
│
└── Global Attributes
    ├── title: "Time-Varying Mobility Example for MMCA External Mobility Support"
    ├── format_version: "2.0"
    ├── conventions: "CF-1.8"
    └── ... (see file for complete list)
```

#### MMCA Expected Format

The dense format stores mobility as a full 3D array:

```
mobility[date, origin, destination] = probability
```

Where:
- `date`: Timestep index (string coordinate in "yyyy-mm-dd" format)
- `origin`: Origin region index (where people live)
- `destination`: Destination region index (where people visit)

**Properties:**
- **Row-stochastic**: For each origin at each date, `sum(mobility[date, origin, :]) == 1.0`
- **Non-negative**: All values >= 0
- **Self-loops dominant**: In realistic mobility, `mobility[date, i, i]` >> `mobility[date, i, j]` for i != j

#### Key Design Decisions

1. **Dense Format**
   - Stores full `(T, M, M)` array of mobility matrices
   - Directly compatible with MMCA's expected format
   - Simpler data access: `mobility[t, i, j]` gives flow from i to j at timestep t
   - More memory-intensive but easier to use

2. **String Date Coordinates**
   - Dates stored as strings in "yyyy-mm-dd" format
   - Matches MMCA configuration date format
   - Enables direct date-based lookup and alignment

3. **Region ID Coordinates**
   - Origin and destination coordinates use region IDs
   - Enables semantic access: `mobility.sel(origin="08001", destination="08002")`
   - Supports both integer and string-based indexing

#### Loading the File

##### Python (xarray)

```python
import xarray as xr

# Load the dataset
ds = xr.open_dataset("mobility_time_varying_example.nc")

# Access mobility data
mobility = ds.mobility.values  # (T, M, M) numpy array
dates = ds.date.values  # (T,) array of date strings
origins = ds.origin.values  # (M,) array of region IDs
destinations = ds.destination.values  # (M,) array of region IDs

# Get mobility matrix for specific date
date_idx = 0
mobility_t = mobility[date_idx]  # (M, M) matrix

# Verify row-stochastic property
row_sums = mobility_t.sum(axis=1)
print(f"Row sums: {row_sums}")  # Should all be ~1.0

# Access using region IDs (xarray feature)
flow_08001_to_08002 = ds.mobility.sel(date="2020-03-01", origin="08001", destination="08002")

ds.close()
```

##### Julia (NCDatasets)

```julia
using NCDatasets

# Open the dataset
ds = NCDataset("mobility_time_varying_example.nc")

# Access components
mobility = ds["mobility"][:]  # (T, M, M) array
dates = ds["date"][:]  # (T,) array of date strings
origins = ds["origin"][:]  # (M,) array of region IDs

# Get mobility matrix for timestep t (1-indexed in Julia)
t = 1
mobility_t = mobility[t, :, :]  # (M, M) matrix

# Verify row-stochastic property
row_sums = sum(mobility_t, dims=2)
println("Row sums: ", row_sums)

close(ds)
```

#### Expected Properties

1. **Row-Stochastic**: For each origin at each date, the sum of outgoing flows should equal 1.0
2. **Non-Negative**: All mobility values should be >= 0
3. **Self-Loops Dominant**: In realistic mobility, self-loop weights are typically much higher than cross-region flows
4. **Time Variation**: Mobility matrices may vary across dates (e.g., due to lockdowns, holidays)

#### Example Data Patterns

The example file demonstrates:
- **3 regions**: Barcelona municipalities (08001, 08002, 08003)
- **10 timesteps**: March 1-10, 2020
- **Dense (3×3) matrices**: All possible connections including self-loops
- **Mobility reduction**: Days 0-3 have normal mobility; days 4-9 show gradual reduction (simulating lockdown)

```
Day 0: Diagonal = [0.863, 0.804, 0.824]  (normal mobility)
Day 4: Diagonal = [0.878, 0.824, 0.837]  (reduction begins)
Day 9: Diagonal = [0.942, 0.896, 0.906]  (significant reduction)
```

#### MMCA Configuration

To use this file with MMCA external mobility support, configure your JSON file:

```json
{
  "population_params": {
    "mobility_variation_type": "external_netcdf",
    "mobility_external_file": "examples/mobility_time_varying_example.nc",
    "mobility_external_variable": "mobility",
    "mobility_validate": true,
    "mobility_fill_missing": false,
    "mobility_time_alignment": "start"
  }
}
```

See [MMCA_mobility_configuration.md](../MMCA_mobility_configuration.md) for complete configuration reference.

#### Regenerating the Example

To regenerate this file with different parameters:

```bash
# From project root
python3 python/generate_mobility_example.py
```

Or modify parameters in `python/generate_mobility_example.py`:
- `num_regions`: Number of regions
- `timesteps`: Number of timesteps
- `reduction_start`: Day when mobility reduction starts
- `reduction_rate`: Daily reduction rate
- `region_ids`: List of region ID codes

#### Integration with MMCA

When implementing external mobility support in MMCA models:

1. **Parse the NetCDF file** to extract the `mobility` variable
2. **For each timestep**:
   - Get `M_t = mobility[t, :, :]` (full matrix at timestep t)
   - Validate row-stochastic property: `all(sum(M_t, dims=2) ≈ 1.0)`
3. **Use M_t** as the mobility matrix for epidemic dynamics at timestep t
4. **Date alignment**: Match simulation dates to NetCDF date coordinates

#### Related Files

- `python/generate_mobility_example.py`: Script that generates this example file
- `MMCA_mobility_configuration.md`: Complete MMCA mobility configuration guide

#### Contact

For questions about this format or implementation support:
- EpiSim.jl repository: https://github.com/your-org/EpiSim.jl
- MMCA repository: https://github.com/MMCA-epidemic-models/MMCA
