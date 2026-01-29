import xarray as xr
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path


def analyze_interventions():
    zarr_path = "runs/synthetic_test/synthetic_observations.zarr"
    ds = xr.open_zarr(zarr_path)

    run_ids = ds.run_id.values
    dates = pd.to_datetime(ds.date.values)

    # Storage for stats
    stats = []

    print(f"Analyzing {len(run_ids)} runs...")

    for run_id in run_ids:
        # Reconstruct folder path
        # run_id in Zarr is sanitized.
        # The generator creates folders like "run_0_Global_Timed_s05"
        # The zarr run_id likely matches the folder name suffix.

        # We need to find the config.
        # Let's search for the folder that ends with this run_id
        # Actually, sanitized run_id is: run_dir_name[4:] if startswith run_ else run_dir_name
        # So folder is likely "run_" + run_id

        run_dir = Path(f"runs/synthetic_test/run_{run_id}")
        config_path = run_dir / "config_auto_py.json"

        if not config_path.exists():
            print(f"Config not found for {run_id}")
            continue

        with open(config_path) as f:
            config = json.load(f)

        # Extract Parameters
        # R0 is scale_beta? Or close to it.
        # Actually R0 depends on beta, gamma, etc.
        # The generator sets "epidemic_params.scale_β". Let's use that as proxy for R0.
        r0_scale = config["epidemic_params"].get("scale_β", 1.0)

        # Seed Size: We can't easily get it from config if it's a file path.
        # But we can read the seed file!
        seed_path = config["data"].get("initial_condition_filename")
        if seed_path and os.path.exists(seed_path):
            seed_df = pd.read_csv(seed_path)
            seed_size = seed_df["M"].sum()
        else:
            seed_size = np.nan

        # Intervention Timing
        # We can look at mobility_reduction in the Zarr
        mobility = ds.mobility_reduction.sel(run_id=run_id).values

        # Find start day (first day where reduction > 0.001)
        intervention_indices = np.where(mobility > 0.001)[0]

        if len(intervention_indices) == 0:
            # Baseline or no intervention
            intervention_day = -1
            cases_at_start = np.nan
            infections_at_start = np.nan
        else:
            start_idx = intervention_indices[0]
            intervention_day = start_idx
            intervention_date = dates[start_idx]

            # Get Cases/Infections at that time
            # ground_truth_infections is daily new infections.
            # We want accumulated infections or daily infections at that specific day.
            # Let's use daily infections (sum over regions)
            daily_infections = (
                ds.ground_truth_infections.sel(run_id=run_id)
                .sum(dim="region_id")
                .values
            )
            infections_at_start = daily_infections[start_idx]

            # Cumulative infections
            cum_infections = daily_infections[: start_idx + 1].sum()

            cases_at_start = cum_infections  # using cumulative as "state"

        stats.append(
            {
                "run_id": run_id,
                "r0_scale": r0_scale,
                "seed_size": seed_size,
                "intervention_day": intervention_day,
                "daily_infections_at_start": infections_at_start,
                "cumulative_infections_at_start": cases_at_start,
                "is_intervention": intervention_day >= 0,
            }
        )

    df = pd.DataFrame(stats)

    # Filter for intervention scenarios
    df_int = df[df["is_intervention"]]

    print("\n--- Analysis Results ---")
    print(f"Total Intervention Scenarios: {len(df_int)}")

    if len(df_int) == 0:
        print("No interventions found.")
        return

    print("\nInfections at Start of Intervention:")
    print(
        df_int[
            ["daily_infections_at_start", "cumulative_infections_at_start"]
        ].describe()
    )

    # Check for "Zero Case" Interventions (Unrealistic)
    # Define "Zero" as very low compared to seed (e.g. < seed_size)
    # Actually, if cumulative < seed_size * 1.5, it hasn't grown much.

    # Let's count how many interventions happened with < 100 cumulative infections
    low_case_interventions = df_int[df_int["cumulative_infections_at_start"] < 100]
    print(
        f"\nInterventions with < 100 cumulative infections: {len(low_case_interventions)} / {len(df_int)}"
    )

    if len(low_case_interventions) > 0:
        print("\nSample of Low Case Interventions:")
        print(
            low_case_interventions[
                [
                    "run_id",
                    "r0_scale",
                    "seed_size",
                    "intervention_day",
                    "cumulative_infections_at_start",
                ]
            ].head()
        )

    # High R0 stats
    high_r0 = df_int[df_int["r0_scale"] > 2.0]
    print(
        f"\nMean Cumulative Cases at Intervention (High R0 > 2.0): {high_r0['cumulative_infections_at_start'].mean():.1f}"
    )

    # Low R0 stats
    low_r0 = df_int[df_int["r0_scale"] < 1.2]
    print(
        f"Mean Cumulative Cases at Intervention (Low R0 < 1.2): {low_r0['cumulative_infections_at_start'].mean():.1f}"
    )

    # Correlation
    # We expect positive correlation between R0 and cases?
    # Or negative?
    # High R0 -> Fast growth -> We detect it at 100 cases -> Intervention.
    # But because of delay, high R0 might overshoot 100 cases more before intervention hits.
    # So High R0 -> Higher Cases at Intervention.

    # Save CSV
    df.to_csv("intervention_analysis.csv", index=False)
    print("\nDetailed stats saved to intervention_analysis.csv")


if __name__ == "__main__":
    analyze_interventions()
