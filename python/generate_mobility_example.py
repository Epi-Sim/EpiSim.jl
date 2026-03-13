#!/usr/bin/env python3
"""
Generate Example Time-Varying Mobility NetCDF File

This script creates an example NetCDF file demonstrating the expected structure
for time-varying mobility data to be provided to MMCA repository maintainers
for implementing external mobility support.

Uses MMCA-compatible dense format: mobility(date, origin, destination)

Output: examples/mobility_time_varying_example.nc
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def create_base_mobility_matrix(num_regions: int = 3) -> np.ndarray:
    """
    Create a base mobility matrix (row-stochastic).

    Design:
    - High self-loops (0.8-0.9): most people stay in place
    - Low cross-region flows (0.1-0.2): distributed among other regions
    - Each row sums to 1.0 (row-stochastic)

    Args:
        num_regions: Number of regions in the mobility matrix

    Returns:
        num_regions × num_regions mobility matrix
    """
    mobility = np.zeros((num_regions, num_regions))

    for i in range(num_regions):
        # High self-loop: 0.8 to 0.9
        self_loop = 0.8 + np.random.rand() * 0.1
        mobility[i, i] = self_loop

        # Distribute remaining probability among other regions
        remaining = 1.0 - self_loop
        num_others = num_regions - 1

        if num_others > 0:
            # Equal distribution to other regions with slight variation
            cross_flows = np.random.dirichlet(np.ones(num_others)) * remaining
            other_indices = [j for j in range(num_regions) if j != i]
            mobility[i, other_indices] = cross_flows

    return mobility


def apply_time_varying_reduction(
    mobility_matrix: np.ndarray,
    timesteps: int,
    reduction_start: int = 4,
    reduction_rate: float = 0.1
) -> np.ndarray:
    """
    Apply gradual mobility reduction over time (simulating lockdown).

    Design:
    - Days 0 to reduction_start-1: normal mobility
    - Days reduction_start onwards: gradual reduction
    - Self-loops increase as cross-region mobility decreases

    Args:
        mobility_matrix: Base mobility matrix (M × M)
        timesteps: Number of timesteps
        reduction_start: Day when reduction starts
        reduction_rate: Daily reduction rate for cross-region flows

    Returns:
        T × M × M array of time-varying mobility matrices
    """
    num_regions = mobility_matrix.shape[0]
    mobility_series = np.zeros((timesteps, num_regions, num_regions))

    for t in range(timesteps):
        if t < reduction_start:
            # Normal mobility
            mobility_series[t] = mobility_matrix.copy()
        else:
            # Apply reduction
            days_of_reduction = t - reduction_start + 1
            reduction_factor = max(0.3, 1.0 - reduction_rate * days_of_reduction)

            mobility_series[t] = mobility_matrix.copy()

            # Reduce cross-region flows
            for i in range(num_regions):
                # Redistribute: reduce cross-flows proportionally, increase self-loop
                for j in range(num_regions):
                    if i != j:
                        mobility_series[t, i, j] = mobility_matrix[i, j] * reduction_factor

                mobility_series[t, i, i] = 1.0 - mobility_series[t, i].sum() + mobility_series[t, i, i]

    return mobility_series


def generate_mobility_netcdf(
    output_path: Path,
    num_regions: int = 3,
    timesteps: int = 10,
    start_date: str = "2020-03-01",
    reduction_start: int = 4,
    reduction_rate: float = 0.1,
    region_ids: list[str] | None = None
) -> None:
    """
    Generate the complete example mobility NetCDF file using MMCA dense format.

    Format: mobility(date, origin, destination) - dense (T, M, M) array

    Args:
        output_path: Path for output NetCDF file
        num_regions: Number of regions
        timesteps: Number of timesteps
        start_date: Start date (YYYY-MM-DD)
        reduction_start: Day when mobility reduction starts
        reduction_rate: Daily reduction rate
        region_ids: Optional list of region IDs (defaults to Barcelona municipalities)
    """
    # Default to Barcelona municipalities if not provided
    if region_ids is None:
        region_ids = ["08001", "08002", "08003"]  # Barcelona: Barcelona, Badalona, Sabadell

    if len(region_ids) != num_regions:
        raise ValueError(f"Expected {num_regions} region IDs, got {len(region_ids)}")

    # Generate date range as strings
    dates = pd.date_range(start=start_date, periods=timesteps, freq="D")
    date_strings = dates.strftime("%Y-%m-%d").tolist()

    # Create base mobility and apply time-varying reduction
    base_mobility = create_base_mobility_matrix(num_regions)
    mobility_series = apply_time_varying_reduction(
        base_mobility, timesteps, reduction_start, reduction_rate
    )

    # Create xarray Dataset with MMCA-compatible dense structure
    data_vars = {
        "mobility": (["date", "origin", "destination"], mobility_series.astype(np.float64), {
            "long_name": "Mobility flow probability",
            "units": "probability",
            "description": "Time-varying mobility matrix. mobility[date, origin, destination] "
                          "gives the fraction of people living in 'origin' who visit 'destination'. "
                          "Each origin row sums to 1.0 (row-stochastic)."
        }),
    }

    coords = {
        "date": (["date"], date_strings, {
            "long_name": "Calendar date",
            "description": "Date corresponding to each timestep (yyyy-mm-dd format)"
        }),
        "origin": (["origin"], np.array(region_ids), {
            "long_name": "Origin region identifier",
            "description": "Municipality or region ID code for origin (home location)"
        }),
        "destination": (["destination"], np.array(region_ids), {
            "long_name": "Destination region identifier",
            "description": "Municipality or region ID code for destination (visited location)"
        }),
    }

    dataset = xr.Dataset(data_vars, coords)

    # Add global attributes
    dataset.attrs = {
        "title": "Time-Varying Mobility Example for MMCA External Mobility Support",
        "description": (
            "Example time-varying mobility data demonstrating the expected NetCDF format "
            "for external mobility input to MMCA epidemic models. Uses MMCA-compatible dense "
            "format with mobility(date, origin, destination) dimensions. "
            "mobility[date, origin, destination] gives the fraction of people living in "
            "'origin' who visit 'destination' on the given date."
        ),
        "format_version": "2.0",
        "conventions": "CF-1.8",
        "creation_date": datetime.now().isoformat(),
        "num_regions": str(num_regions),
        "num_timesteps": str(timesteps),
        "mobility_reduction_start": str(reduction_start),
        "mobility_reduction_rate": str(reduction_rate),
        "contact": "EpiSim.jl Contributors",
        "institution": "EpiSim.jl - Epidemic Simulation Framework",
    }

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to NetCDF
    dataset.to_netcdf(output_path)

    print(f"Created mobility example file: {output_path}")
    print("  Format: Dense MMCA-compatible (date, origin, destination)")
    print(f"  Regions: {num_regions}")
    print(f"  Timesteps: {timesteps}")
    print(f"  Mobility shape: {mobility_series.shape}")
    print(f"  Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")


def main():
    """Main entry point for script execution."""
    output_path = Path(__file__).parent.parent / "examples" / "mobility_time_varying_example.nc"

    generate_mobility_netcdf(
        output_path=output_path,
        num_regions=3,
        timesteps=10,
        start_date="2020-03-01",
        reduction_start=4,
        reduction_rate=0.1,
        region_ids=["08001", "08002", "08003"]  # Barcelona municipalities
    )

    print("\nTo inspect the file:")
    print(f"  uv run python -c 'import xarray as xr; ds = xr.open_dataset(\"{output_path}\"); print(ds); print(ds.mobility.shape)'")


if __name__ == "__main__":
    main()
