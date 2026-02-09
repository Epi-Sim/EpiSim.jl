# Time-Varying Mobility Validation - Test Results & Findings

## Summary

This document summarizes the findings from testing time-varying mobility validation with EpiSim.jl.

## Test Results

### Created Test Files

Both test files were successfully created at:
```
test_data/mobility_validation/mobility_valid.nc    (9.4 KB)
test_data/mobility_validation/mobility_invalid.nc  (9.6 KB)
```

#### Valid File (`mobility_valid.nc`)
- **Dimensions**: (10 dates, 3 origins, 3 destinations)
- **Format**: Dense `mobility(date, origin, destination)`
- **Property**: Row-stochastic (each row sums to 1.0)
- **Result**: ✓ PASSED validation

#### Invalid File (`mobility_invalid.nc`)
- **Dimensions**: (10 dates, 3 origins, 3 destinations)
- **Format**: Dense `mobility(date, origin, destination)`
- **Property**: NOT row-stochastic (rows sum to 0.7-0.8 instead of 1.0)
- **Result**: ✓ FAILED validation (as expected)

### Validation Logic Implemented

The Python validation script (`python/test_mobility_validation.py`) implements:
1. File existence and readability check
2. Required dimensions check: `(date, origin, destination)`
3. Required variable check: `mobility`
4. Row-stochastic property: `sum(mobility[date, origin, :]) == 1.0`
5. Non-negative values check

## Key Finding: External NetCDF Mobility Not Yet Implemented

After thorough exploration of the EpiSim.jl codebase, I found that:

### What IS Documented
- `MMCA_mobility_configuration.md` describes `external_netcdf` mode
- Configuration parameters are defined:
  - `mobility_variation_type`: `"none"`, `"ipfp_simple"`, `"external_netcdf"`
  - `mobility_external_file`: Path to NetCDF file
  - `mobility_validate`: Whether to validate
  - `mobility_fill_missing`: Whether to interpolate missing dates
  - `mobility_time_alignment`: Date alignment mode

### What IS Currently Implemented
The current EpiSim.jl code (`src/engine.jl`) **ONLY** supports:
- Loading mobility from **CSV files** (`R_mobility_matrix.csv`)
- Static mobility matrices for entire simulation

### What is NOT Implemented
- Loading external NetCDF mobility files
- Time-varying mobility from external sources
- Validation of external mobility data
- Date-based mobility alignment

### Evidence from Code

**src/engine.jl:68-70** (current implementation):
```julia
# Loading mobility network
mobility_matrix_filename = joinpath(data_path, data_dict["mobility_matrix_filename"])
network_df  = CSV.read(mobility_matrix_filename, DataFrame)
```

The mobility is loaded as a static CSV DataFrame and passed to the MMCA engine.

## Implications

### For Testing Valid/Invalid Mobility
The validation logic I created demonstrates what **should** happen when external NetCDF mobility is implemented:
1. Load NetCDF file with `mobility(date, origin, destination)` variable
2. Validate row-stochastic property at load time
3. Reject files with row sums != 1.0 (within tolerance)
4. Provide clear error messages for invalid data

### To Run Actual Simulation with Time-Varying Mobility

You would need to implement one of:

**Option A: Use MMCA packages directly**
- The external NetCDF mobility support might be in `MMCACovid19Vac.jl` or `MMCAcovid19.jl` packages
- Bypass EpiSim.jl and use the MMCA packages directly

**Option B: Implement in EpiSim.jl**
- Add NetCDF loading to `src/engine.jl`
- Parse `mobility_variation_type` from config
- Implement date-based mobility lookup
- Add validation logic at load time
- Pass time-varying mobility to MMCA engine

**Option C: Use ipfp_simple mode (if available)**
- Set `mobility_variation_type: "ipfp_simple"`
- This generates time-varying mobility internally using IPFP
- No external file needed

## Test Script Usage

To run the validation tests:

```bash
# Run the test script
python3 python/test_mobility_validation.py

# Inspect the created files
python3 -c "
import xarray as xr
ds = xr.open_dataset('test_data/mobility_validation/mobility_valid.nc')
print('Valid file:')
print(f'  Shape: {ds.mobility.shape}')
print(f'  Row sums (t=0): {ds.mobility[0].sum(axis=1).values}')

ds = xr.open_dataset('test_data/mobility_validation/mobility_invalid.nc')
print('Invalid file:')
print(f'  Shape: {ds.mobility.shape}')
print(f'  Row sums (t=0): {ds.mobility[0].sum(axis=1).values}')
"
```

## Recommendations

1. **Verify MMCA Package Support**: Check if `MMCACovid19Vac.jl` actually supports external NetCDF mobility. The `MMCA_mobility_configuration.md` document might describe features that exist in the MMCA packages but not exposed through EpiSim.jl.

2. **Implement External Mobility Loading**: To enable this feature in EpiSim.jl, add NetCDF loading logic to `src/engine.jl` with validation.

3. **Test with ipfp_simple Mode**: If the MMCA engine supports `ipfp_simple` mode, test that first to verify time-varying mobility works at all.

## Files Created

| File | Purpose | Status |
|------|---------|--------|
| `python/test_mobility_validation.py` | Test script with validation logic | ✓ Created |
| `test_data/mobility_validation/mobility_valid.nc` | Valid row-stochastic mobility | ✓ Created |
| `test_data/mobility_validation/mobility_invalid.nc` | Invalid non-row-stochastic mobility | ✓ Created |
