#!/usr/bin/env python3
"""Analyze synthetic data outputs from EpiSim.jl"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load the zarr dataset
zarr_path = Path('runs/synthetic_catalonia/raw_synthetic_observations.zarr')
ds = xr.open_zarr(zarr_path)

print('=' * 60)
print('EpiSim.jl Synthetic Data Analysis')
print('=' * 60)
print()

print('=== Dataset Overview ===')
print(f'Dimensions: {dict(ds.dims)}')
print()

print('=== Variables ===')
for var in ds.data_vars:
    print(f'  {var}: {ds[var].dims}')
print()

print('=== Scenario Types ===')
scenarios = ds['synthetic_scenario_type'].values
unique, counts = np.unique(scenarios, return_counts=True)
for s, c in zip(unique, counts):
    print(f'  {s}: {c} runs')
print()

# Get baseline run IDs
baseline_mask = scenarios == 'Baseline'
baseline_ids = np.where(baseline_mask)[0]
print('=== Baseline Runs ===')
print(f'Number of baselines: {len(baseline_ids)}')
print(f'Baseline run IDs: {ds.run_id.values[baseline_ids]}')
print()

# Intervention scenarios
intervention_mask = scenarios != 'Baseline'
intervention_types = scenarios[intervention_mask]
unique_int, counts_int = np.unique(intervention_types, return_counts=True)
print('=== Intervention Scenarios ===')
for s, c in zip(unique_int, counts_int):
    print(f'  {s}: {c} runs')
print()

# Intervention strengths
if 'synthetic_strength' in ds and intervention_mask.any():
    strengths = ds['synthetic_strength'].values[intervention_mask]
    print('=== Intervention Strengths ===')
    print(f'Min: {np.min(strengths):.2f}, Max: {np.max(strengths):.2f}, Mean: {np.mean(strengths):.2f}')
    print()

# Date range
print('=== Date Range ===')
print(f'Start: {ds.date.values[0]}')
print(f'End: {ds.date.values[-1]}')
print(f'Days: {len(ds.date)}')
print()

# Regional info
print('=== Regional Info ===')
print(f'Number of regions: {len(ds.region_id)}')
print(f'Region IDs (first 10): {ds.region_id.values[:10]}')
print()

# Population
if 'population' in ds:
    total_pop = ds['population'].sum().values
    print('=== Population ===')
    print(f'Total population: {total_pop:,.0f}')
    print()

# Statistics by scenario type
print('=== Summary Statistics (Ground Truth) ===')
stats_data = []
for scenario_type in np.unique(scenarios):
    mask = scenarios == scenario_type
    subset = ds.isel(run_id=np.where(mask)[0])

    if 'infections_true' in subset:
        inf = subset['infections_true'].sum(dim=['run_id', 'region_id', 'date']).values
        hosp = subset['hospitalizations_true'].sum(dim=['run_id', 'region_id', 'date']).values
        deaths = subset['deaths_true'].sum(dim=['run_id', 'region_id', 'date']).values

        print(f'{scenario_type}:')
        print(f'  Total Infections:       {inf:,.0f}')
        print(f'  Total Hospitalizations: {hosp:,.0f}')
        print(f'  Total Deaths:           {deaths:,.0f}')
        print()

        stats_data.append({
            'scenario': scenario_type,
            'infections': inf,
            'hospitalizations': hosp,
            'deaths': deaths
        })

# Calculate intervention effectiveness
if intervention_mask.any():
    print('=' * 60)
    print('Intervention Effectiveness Analysis')
    print('=' * 60)
    print()

    # Group by profile to compare twins
    # Profile is encoded in run_id (e.g., "profile_0_baseline")
    profiles = {}
    for run_id in ds.run_id.values:
        parts = run_id.strip().split('_')
        if len(parts) >= 2:
            profile = f"profile_{parts[1]}"
            scenario = parts[-1] if len(parts) > 2 else 'baseline'
            if profile not in profiles:
                profiles[profile] = {}
            profiles[profile][scenario] = run_id

    print('=== Profile Grouping ===')
    print(f'Number of unique profiles: {len(profiles)}')
    print()

    # Calculate effectiveness per profile
    effectiveness_data = []
    baseline_infs = []
    scenario_infs = []

    for profile, scenarios_dict in sorted(profiles.items()):
        if 'baseline' in scenarios_dict:
            baseline_run = scenarios_dict['baseline']
            baseline_idx = np.where(ds.run_id.values == baseline_run.strip())[0][0]
            baseline_inf = ds['infections_true'].isel(run_id=baseline_idx).sum().values
            baseline_infs.append(baseline_inf)

            for scenario, run_id in scenarios_dict.items():
                if scenario != 'baseline':
                    idx = np.where(ds.run_id.values == run_id.strip())[0][0]
                    scenario_type = ds['synthetic_scenario_type'].values[idx]
                    scenario_inf = ds['infections_true'].isel(run_id=idx).sum().values
                    scenario_infs.append(scenario_inf)

                    effectiveness = 1 - (scenario_inf / baseline_inf)
                    strength = ds['synthetic_strength'].values[idx]

                    effectiveness_data.append({
                        'profile': profile,
                        'scenario_type': scenario_type,
                        'strength': strength,
                        'baseline_inf': baseline_inf,
                        'scenario_inf': scenario_inf,
                        'effectiveness': effectiveness
                    })

    if effectiveness_data:
        df_eff = pd.DataFrame(effectiveness_data)

        print('=== Effectiveness by Scenario Type ===')
        for scenario_type in df_eff['scenario_type'].unique():
            subset = df_eff[df_eff['scenario_type'] == scenario_type]
            print(f'{scenario_type}:')
            print(f'  Mean effectiveness: {subset["effectiveness"].mean():.1%}')
            print(f'  Std effectiveness:  {subset["effectiveness"].std():.1%}')
            print(f'  Range: [{subset["effectiveness"].min():.1%}, {subset["effectiveness"].max():.1%}]')
            print()

        print('=== Effectiveness by Strength ===')
        strength_groups = df_eff.groupby('strength')['effectiveness'].agg(['mean', 'std', 'min', 'max'])
        print(strength_groups)
        print()
else:
    print('=' * 60)
    print('No Intervention Scenarios Found')
    print('=' * 60)
    print('This dataset contains only baseline runs (no interventions).')
    print()

    # Still organize baselines by profile
    profiles = {}
    for run_id in ds.run_id.values:
        run_id_clean = run_id.strip()
        parts = run_id_clean.split('_')
        if len(parts) >= 2 and parts[0].isdigit():
            profile = parts[0]
            if profile not in profiles:
                profiles[profile] = []
            profiles[profile].append(run_id_clean)

    print('=== Profile Grouping ===')
    print(f'Number of unique profiles: {len(profiles)}')
    print(f'Profiles: {sorted(profiles.keys())}')
    print()

    # Calculate baseline statistics
    baseline_stats = []
    for profile, run_ids in sorted(profiles.items()):
        for run_id in run_ids:
            idx = np.where([x.strip() == run_id for x in ds.run_id.values])[0][0]
            inf = ds['infections_true'].isel(run_id=idx).sum().values
            hosp = ds['hospitalizations_true'].isel(run_id=idx).sum().values
            deaths = ds['deaths_true'].isel(run_id=idx).sum().values

            baseline_stats.append({
                'profile': profile,
                'run_id': run_id,
                'infections': inf,
                'hospitalizations': hosp,
                'deaths': deaths
            })

    if baseline_stats:
        df_baseline = pd.DataFrame(baseline_stats)

        print('=== Baseline Statistics Summary ===')
        print(f'Mean Infections:       {df_baseline["infections"].mean():,.0f} (±{df_baseline["infections"].std():,.0f})')
        print(f'Mean Hospitalizations: {df_baseline["hospitalizations"].mean():,.0f} (±{df_baseline["hospitalizations"].std():,.0f})')
        print(f'Mean Deaths:           {df_baseline["deaths"].mean():,.0f} (±{df_baseline["deaths"].std():,.0f})')
        print()

        print('=== Range Across Baselines ===')
        print(f'Infections:       [{df_baseline["infections"].min():,.0f}, {df_baseline["infections"].max():,.0f}]')
        print(f'Hospitalizations: [{df_baseline["hospitalizations"].min():,.0f}, {df_baseline["hospitalizations"].max():,.0f}]')
        print(f'Deaths:           [{df_baseline["deaths"].min():,.0f}, {df_baseline["deaths"].max():,.0f}]')
        print()

# Spike detection
print('=' * 60)
print('Spike Detection Analysis')
print('=' * 60)
print()

# Calculate daily infections across all baselines
baseline_ds = ds.isel(run_id=baseline_ids)
if 'infections_true' in baseline_ds:
    # infections_true is (run_id, region_id, date)
    daily_inf_baseline = baseline_ds['infections_true'].sum(dim=['run_id', 'region_id']).values

    # Find spike periods using percentile threshold
    threshold = np.percentile(daily_inf_baseline, 90)
    spike_mask = daily_inf_baseline >= threshold

    # Find contiguous spike periods
    spike_starts = []
    spike_ends = []
    in_spike = False

    for i, is_spike in enumerate(spike_mask):
        if is_spike and not in_spike:
            spike_starts.append(i)
            in_spike = True
        elif not is_spike and in_spike:
            spike_ends.append(i - 1)
            in_spike = False

    if in_spike:
        spike_ends.append(len(spike_mask) - 1)

    print('=== Spike Periods (90th percentile threshold) ===')
    print(f'Threshold: {threshold:,.0f} infections/day')
    print(f'Number of spike periods: {len(spike_starts)}')

    for start, end in zip(spike_starts, spike_ends):
        start_date = str(ds.date.values[start])[:10]
        end_date = str(ds.date.values[end])[:10]
        duration = end - start + 1
        peak = daily_inf_baseline[start:end+1].max()
        print(f'  {start_date} to {end_date} ({duration} days) - Peak: {peak:,.0f}')
    print()

# Generate epicurves plot
print('=' * 60)
print('Generating Epicurves Plot')
print('=' * 60)
print()

if 'infections_true' in ds:
    if intervention_mask.any():
        # With interventions - show twin comparisons
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Select a few representative profiles for plotting
        profiles_to_plot = sorted(profiles.keys())[:3]

        for i, profile in enumerate(profiles_to_plot):
            ax = axes[i]
            scenarios_dict = profiles[profile]

            # Plot baseline
            if 'baseline' in scenarios_dict:
                baseline_run = scenarios_dict['baseline']
                baseline_idx = np.where(ds.run_id.values == baseline_run.strip())[0][0]
                baseline_inf = ds['infections_true'].isel(run_id=baseline_idx).sum(dim='region_id').values
                dates = pd.to_datetime(ds.date.values)
                ax.plot(dates, baseline_inf, label='Baseline', linewidth=2, color='black')

            # Plot interventions
            for scenario, run_id in scenarios_dict.items():
                if scenario != 'baseline':
                    idx = np.where(ds.run_id.values == run_id.strip())[0][0]
                    scenario_type = ds['synthetic_scenario_type'].values[idx]
                    strength = ds['synthetic_strength'].values[idx]
                    scenario_inf = ds['infections_true'].isel(run_id=idx).sum(dim='region_id').values

                    label = f'{scenario_type} (strength={strength:.1f})'
                    alpha = 0.7 if 'Global' in scenario_type else 0.5
                    ax.plot(dates, scenario_inf, label=label, alpha=alpha, linewidth=1.5)

            ax.set_title(f'{profile} - Daily Infections')
            ax.set_xlabel('Date')
            ax.set_ylabel('Infections')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = 'runs/synthetic_catalonia/epicurves_analysis.png'
        plt.savefig(output_path, dpi=150)
        print(f'Saved epicurves plot to: {output_path}')
        print()
    else:
        # Baseline only - show multiple baselines
        fig, ax = plt.subplots(figsize=(14, 8))

        # Get run IDs sorted
        run_ids_sorted = sorted(ds.run_id.values, key=lambda x: int(x.strip().split('_')[0]))

        # Plot each baseline
        for run_id in run_ids_sorted[:10]:  # Plot first 10
            idx = np.where([x.strip() == run_id.strip() for x in ds.run_id.values])[0][0]
            infections = ds['infections_true'].isel(run_id=idx).sum(dim='region_id').values
            dates = pd.to_datetime(ds.date.values)
            label = run_id.strip()
            ax.plot(dates, infections, label=label, alpha=0.7, linewidth=1.5)

        ax.set_title('Baseline Epicurves - Daily Infections')
        ax.set_xlabel('Date')
        ax.set_ylabel('Infections')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = 'runs/synthetic_catalonia/epicurves_baseline_only.png'
        plt.savefig(output_path, dpi=150)
        print(f'Saved baseline epicurves plot to: {output_path}')
        print()

# Create effectiveness vs strength plot (only if interventions exist)
if intervention_mask.any() and 'effectiveness_data' in locals() and effectiveness_data:
    fig, ax = plt.subplots(figsize=(10, 6))

    for scenario_type in df_eff['scenario_type'].unique():
        subset = df_eff[df_eff['scenario_type'] == scenario_type]
        ax.scatter(subset['strength'], subset['effectiveness'],
                   label=scenario_type, alpha=0.6, s=50)

    ax.set_xlabel('Intervention Strength')
    ax.set_ylabel('Effectiveness (1 - infections/baseline)')
    ax.set_title('Intervention Effectiveness vs Strength')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    output_path = 'runs/synthetic_catalonia/effectiveness_vs_strength.png'
    plt.savefig(output_path, dpi=150)
    print(f'Saved effectiveness plot to: {output_path}')
    print()

print('=' * 60)
print('Analysis Complete')
print('=' * 60)
