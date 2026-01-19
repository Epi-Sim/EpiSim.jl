#!/usr/bin/env python3
"""Plot zarr-based epicurves with wastewater and lockdown highlighting.

This script visualizes consolidated zarr observations containing:
- Ground truth infections
- Hospitalizations
- Deaths
- Reported cases
- Wastewater observations
- Mobility reduction (kappa0) with lockdown highlighting
"""

import argparse
import logging
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ZarrPlotter")

# Constants
DATE_ORIGIN = "2020-02-09"
LOCKDOWN_THRESHOLD = 0.01
DPI = 150

# Color scheme
COLOR_INFECTIONS = "#1f77b4"  # blue
COLOR_CASES = "#ff7f0e"  # orange
COLOR_WASTEWATER = "#2ca02c"  # green
COLOR_HOSPITALIZATIONS = "#d62728"  # red
COLOR_DEATHS = "#9467bd"  # purple


def load_zarr_data(zarr_path):
    """Load the consolidated zarr dataset.

    Args:
        zarr_path: Path to zarr dataset

    Returns:
        xarray.Dataset with variables:
            - ground_truth_infections: (run_id, region_id, date)
            - obs_hospitalizations: (run_id, region_id, date)
            - obs_deaths: (run_id, region_id, date)
            - obs_cases: (run_id, region_id, date)
            - obs_wastewater: (run_id, region_id, date)
            - ascertainment_rate: (run_id, date)
            - mobility_reduction: (run_id, date)
    """
    logger.info(f"Loading zarr data from {zarr_path}")
    ds = xr.open_zarr(zarr_path)
    logger.info(f"Loaded {len(ds.run_id)} runs, {len(ds.region_id)} regions, {len(ds.date)} dates")
    return ds


def convert_dates(ds):
    """Convert dates to pandas datetime.

    Args:
        ds: xarray Dataset with date coordinate

    Returns:
        pandas.DatetimeIndex
    """
    dates = ds["date"].values
    # xarray may return datetime64, strings, or numeric
    # pd.to_datetime handles all cases automatically
    return pd.to_datetime(dates)


def detect_lockdown_periods(mobility_reduction, dates, threshold=LOCKDOWN_THRESHOLD):
    """Detect contiguous lockdown periods from mobility reduction series.

    Args:
        mobility_reduction: Array of mobility reduction values (0-1)
        dates: pandas DatetimeIndex
        threshold: Minimum value to consider as lockdown

    Returns:
        List of (start_date, end_date, max_severity) tuples
    """
    lockdown_mask = mobility_reduction > threshold

    if not lockdown_mask.any():
        return []

    # Find transitions in/out of lockdown
    # Check for start of lockdown (mask goes from False to True)
    starts = np.where(lockdown_mask & ~np.roll(lockdown_mask, 1))[0]
    # Handle case where lockdown starts at index 0
    if lockdown_mask[0]:
        starts = np.concatenate([[0], starts])

    # Check for end of lockdown (mask goes from True to False)
    ends = np.where(lockdown_mask & ~np.roll(lockdown_mask, -1))[0]
    # Handle case where lockdown ends at last index
    if lockdown_mask[-1]:
        ends = np.concatenate([ends, [len(lockdown_mask) - 1]])

    # Pair start/end events and extract severity
    periods = []
    for start, end in zip(starts, ends):
        period_values = mobility_reduction[start:end + 1]
        max_severity = float(period_values.max())
        periods.append((dates[start], dates[end], max_severity))

    return periods


def plot_run_epicurves(ds, run_id, output_dir, aggregate_regions=True):
    """Plot epicurves for a single run with lockdown highlighting.

    Panel layout (2 subplots, stacked vertically):
    1. Ground truth infections, reported cases, hospitalizations, deaths (left y-axis)
       Wastewater observations (right y-axis, twin)
    2. Mobility reduction (kappa0) bar chart showing severity

    Args:
        ds: xarray Dataset
        run_id: Run ID to plot
        output_dir: Directory to save output
        aggregate_regions: If True, sum across all regions
    """
    run_data = ds.sel(run_id=run_id)
    dates = convert_dates(ds)

    # Extract data for this run
    mobility = run_data["mobility_reduction"].values

    if aggregate_regions:
        infections = run_data["ground_truth_infections"].sum(dim="region_id").values
        hospitalizations = run_data["obs_hospitalizations"].sum(dim="region_id").values
        deaths = run_data["obs_deaths"].sum(dim="region_id").values
        cases = run_data["obs_cases"].sum(dim="region_id").values
        wastewater = run_data["obs_wastewater"].sum(dim="region_id").values
        region_label = "All Regions"
    else:
        infections = run_data["ground_truth_infections"].values
        hospitalizations = run_data["obs_hospitalizations"].values
        deaths = run_data["obs_deaths"].values
        cases = run_data["obs_cases"].values
        wastewater = run_data["obs_wastewater"].values
        region_label = "Per Region"

    # Get scenario info
    scenario = str(run_data["scenario_type"].values)
    strength = run_data["strength"].values

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={'height_ratios': [3, 1]})

    # --- Top subplot: Epicurves with twin axis ---
    # Plot infections, cases, hospitalizations, deaths on left axis
    ax1.plot(dates, infections, color=COLOR_INFECTIONS, linestyle='-',
             linewidth=2, label='Ground Truth Infections')
    ax1.plot(dates, cases, color=COLOR_CASES, linestyle='--',
             linewidth=2, label='Reported Cases')
    ax1.plot(dates, hospitalizations, color=COLOR_HOSPITALIZATIONS, linestyle='-.',
             linewidth=2, label='Hospitalizations')
    ax1.plot(dates, deaths, color=COLOR_DEATHS, linestyle=':',
             linewidth=2, label='Deaths')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Count (Infections/Cases/Hospitalizations/Deaths)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Twin axis for wastewater
    ax1_twin = ax1.twinx()
    ax1_twin.plot(dates, wastewater, color=COLOR_WASTEWATER,
                  linestyle='-.', linewidth=2, label='Wastewater')
    ax1_twin.set_ylabel('Wastewater Signal')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper left', bbox_to_anchor=(1.02, 1))

    # Title with scenario info
    title = f'Epicurves: {scenario}'
    if not np.isnan(strength):
        title += f' (Strength: {strength:.0f})'
    ax1.set_title(title)

    # --- Bottom subplot: Mobility reduction bar chart ---
    # Create color map based on severity
    cmap = plt.get_cmap('RdYlGn_r')  # Red = high, Green = low
    colors = cmap(mobility)

    ax2.bar(dates, mobility, color=colors, width=1)
    ax2.axhline(y=LOCKDOWN_THRESHOLD, color='black', linestyle=':',
                linewidth=1, label=f'Lockdown Threshold ({LOCKDOWN_THRESHOLD})')
    ax2.set_xlabel('Date')
    ax2.set_ylabel(r'Mobility Reduction ($\kappa_0$)')
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='upper right')

    # Format dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.tick_params(axis='x', rotation=45)

    # Highlight lockdown periods
    periods = detect_lockdown_periods(mobility, dates)
    if periods:
        logger.info(f"Detected {len(periods)} lockdown period(s)")
        for start, end, severity in periods:
            ax2.axvspan(start, end, alpha=0.2, color='red')

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / f"epicurves_{run_id}.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved single-run plot to {output_path}")


def plot_faceted_grid(ds, run_ids, output_dir, scenarios=None, n_runs=None):
    """Plot epicurves in a faceted grid layout.

    Layout:
    - Rows: Scenario types (Baseline, Global_Const, Global_Timed, etc.)
    - Columns: Different runs/strengths within each scenario
    - Each subplot: twin-axis (infections/cases/hospitalizations/deaths left, wastewater right)

    Args:
        ds: xarray Dataset
        run_ids: List of run IDs to plot
        output_dir: Directory to save output
        scenarios: List of scenario types to filter by (None = all)
        n_runs: Limit number of runs per scenario (None = all)
    """
    dates = convert_dates(ds)

    # Filter by scenarios if specified
    if scenarios:
        run_ids = [r for r in run_ids if str(ds.sel(run_id=r)["scenario_type"].values) in scenarios]

    # Group runs by scenario
    scenario_groups = {}
    for run_id in run_ids:
        scenario = str(ds.sel(run_id=run_id)["scenario_type"].values)
        if scenario not in scenario_groups:
            scenario_groups[scenario] = []
        scenario_groups[scenario].append(run_id)

    # Limit runs per scenario if specified
    if n_runs:
        for scenario in scenario_groups:
            scenario_groups[scenario] = scenario_groups[scenario][:n_runs]

    # Determine grid size
    n_scenarios = len(scenario_groups)
    max_runs = max(len(runs) for runs in scenario_groups.values())

    # Create figure
    fig_size = (4 * max_runs, 3 * n_scenarios)
    fig, axes = plt.subplots(n_scenarios, max_runs, figsize=fig_size)

    # Handle case of single row/column
    if n_scenarios == 1:
        axes = axes.reshape(1, -1)
    if max_runs == 1:
        axes = axes.reshape(-1, 1)

    # Plot each scenario/run
    for row_idx, (scenario, runs) in enumerate(sorted(scenario_groups.items())):
        for col_idx, run_id in enumerate(runs):
            ax = axes[row_idx, col_idx]
            run_data = ds.sel(run_id=run_id)

            # Aggregate across regions
            infections = run_data["ground_truth_infections"].sum(dim="region_id").values
            hospitalizations = run_data["obs_hospitalizations"].sum(dim="region_id").values
            deaths = run_data["obs_deaths"].sum(dim="region_id").values
            cases = run_data["obs_cases"].sum(dim="region_id").values
            wastewater = run_data["obs_wastewater"].sum(dim="region_id").values
            mobility = run_data["mobility_reduction"].values

            # Plot epicurves
            ax.plot(dates, infections, color=COLOR_INFECTIONS, linestyle='-',
                    linewidth=1.5, label='Infections')
            ax.plot(dates, cases, color=COLOR_CASES, linestyle='--',
                    linewidth=1.5, label='Cases')
            ax.plot(dates, hospitalizations, color=COLOR_HOSPITALIZATIONS, linestyle='-.',
                    linewidth=1.5, label='Hospitalizations')
            ax.plot(dates, deaths, color=COLOR_DEATHS, linestyle=':',
                    linewidth=1.5, label='Deaths')

            # Twin axis for wastewater
            ax_twin = ax.twinx()
            ax_twin.plot(dates, wastewater, color=COLOR_WASTEWATER,
                        linestyle='-.', linewidth=1.5, label='Wastewater')

            # Shade lockdown periods
            periods = detect_lockdown_periods(mobility, dates)
            for start, end, _ in periods:
                ax.axvspan(start, end, alpha=0.2, color='red')

            # Formatting
            ax.set_xlabel('Date')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

            # Title with run info
            strength = run_data["strength"].values
            title = f'{scenario}'
            if not np.isnan(strength):
                title += f'\nStrength: {strength:.0f}'
            ax.set_title(title)

            # Only label y-axes on left column and right column
            if col_idx == 0:
                ax.set_ylabel('Infections/Cases')
            if col_idx == max_runs - 1:
                ax_twin.set_ylabel('Wastewater')

            # Only add legend for first plot
            if row_idx == 0 and col_idx == 0:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax_twin.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2,
                          loc='upper left', bbox_to_anchor=(1.02, 1))

    # Hide unused subplots
    for row_idx in range(n_scenarios):
        for col_idx in range(max_runs):
            if col_idx >= len(list(scenario_groups.values())[row_idx]):
                axes[row_idx, col_idx].set_visible(False)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / "epicurves_faceted.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved faceted plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot zarr-based epicurves with lockdown highlighting"
    )
    parser.add_argument(
        "--zarr",
        default="../runs/synthetic_test/synthetic_observations.zarr",
        help="Path to zarr dataset"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Specific run IDs to plot (by default plots all runs)"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["Baseline", "Global_Const", "Global_Timed", "Local_Timed"],
        help="Filter by scenario type"
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        help="Limit number of runs per scenario (for faceted view)"
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        default=True,
        help="Aggregate across regions (default: True)"
    )
    parser.add_argument(
        "--no-aggregate",
        dest="aggregate",
        action="store_false",
        help="Do not aggregate across regions"
    )
    parser.add_argument(
        "--output-dir",
        default="../runs/synthetic_test",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--format",
        choices=["single", "faceted"],
        default="faceted",
        help="Output format: single plot per run or faceted grid"
    )

    args = parser.parse_args()

    # Load data
    ds = load_zarr_data(args.zarr)

    # Determine which runs to plot
    if args.runs:
        run_ids = args.runs
    else:
        run_ids = list(ds.run_id.values)

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate plots
    if args.format == "single":
        for run_id in run_ids:
            plot_run_epicurves(ds, run_id, args.output_dir, args.aggregate)
    else:
        plot_faceted_grid(ds, run_ids, args.output_dir, args.scenarios, args.n_runs)


if __name__ == "__main__":
    main()
