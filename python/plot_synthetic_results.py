import glob
import json
import logging
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Plotter")


def parse_run_dir(run_dir_name):
    # Expected format: run_{pid}_{scenario}_s{strength} or run_{pid}_{scenario}
    # Example: run_0_Baseline, run_0_Global_Const_s50
    match = re.match(r"run_(\d+)_(.+)", run_dir_name)
    if match:
        pid = int(match.group(1))
        scen_str = match.group(2)

        # Clean scenario string by removing suffix _s\d+ if present
        # e.g. Global_Const_s50 -> Global_Const
        s_match = re.search(r"(.+)_s\d+$", scen_str)
        scenario = s_match.group(1) if s_match else scen_str

        return pid, scenario
    return None, None


def load_run_data(run_dir):
    """
    Load configuration and simulation results for a single run.
    Returns a dictionary with parameters and time series data.
    """
    run_path = Path(run_dir)
    pid, scenario_name = parse_run_dir(run_path.name)

    if pid is None:
        logger.warning(f"Could not parse run directory name: {run_path.name}")
        return None

    # Find the UUID folder (assuming only one per run)
    subdirs = [d for d in run_path.iterdir() if d.is_dir()]
    if not subdirs:
        logger.warning(f"No UUID folder found in {run_dir}")
        return None

    uuid_dir = subdirs[0]
    config_path = uuid_dir / "config_auto_py.json"
    output_path = uuid_dir / "output" / "observables.nc"

    if not config_path.exists() or not output_path.exists():
        logger.warning(f"Missing config or output in {uuid_dir}")
        return None

    # Load Config
    with open(config_path) as f:
        config = json.load(f)

    # Extract Parameters
    try:
        r0 = config["epidemic_params"].get("scale_β", 0)
        npi = config.get("NPI", {})

        # Check for custom direct mobility reduction (used by Local_Static scenarios)
        custom = npi.get("custom", {})
        if (
            "direct_mobility_reduction" in custom
            and custom["direct_mobility_reduction"] > 0
        ):
            global_strength = custom["direct_mobility_reduction"]
            min_kappa = 1.0 - global_strength
            kappas = [min_kappa]
            times = [0]
            logger.info(f"Using custom mobility reduction: {global_strength}")
        else:
            # Standard kappa0 processing
            kappas = npi.get("κ₀s", None)
            times = npi.get("tᶜs", [0])

            # If κ₀s is not in config, check for kappa0_filename
            if kappas is None or len(kappas) == 0:
                kappa0_file = config.get("data", {}).get("kappa0_filename", None)
                if kappa0_file and os.path.exists(kappa0_file):
                    kappa0_df = pd.read_csv(kappa0_file)
                    kappas = kappa0_df["reduction"].values.tolist()
                    times = kappa0_df["time"].values.tolist()
                    logger.info(f"Loaded kappa0 from file: {kappa0_file}")
                else:
                    kappas = [1.0]

            # Calculate NPI metrics
            min_kappa = min(kappas)
            global_strength = 1.0 - min_kappa

        # Calculate duration of reduction
        duration = 0
        if len(kappas) >= 3 and kappas[1] < 1.0:
            duration = times[2] - times[1]
        elif len(kappas) > 1:
            # Calculate duration from kappa0 file: number of days where kappa < 1.0
            reduced_days = sum(1 for k in kappas if k < 1.0)
            if reduced_days > 0:
                duration = reduced_days
        elif len(kappas) == 1 and kappas[0] < 1.0:
            duration = 999  # Constant reduction

    except Exception as e:
        logger.error(f"Error parsing config {config_path}: {e}")
        return None

    # Load Simulation Results
    try:
        ds = xr.open_dataset(output_path)

        # Aggregate New Infections over G (Age) and M (Region)
        daily_infections = ds["new_infected"].sum(dim=["G", "M"]).values
        daily_hospitalized = ds["new_hospitalized"].sum(dim=["G", "M"]).values
        daily_deaths = ds["new_deaths"].sum(dim=["G", "M"]).values

        # Derived Cumulative metrics
        cumulative_infections = np.cumsum(daily_infections)
        cumulative_deaths = np.cumsum(daily_deaths)

        time_steps = np.arange(len(daily_infections))

        total_infections = np.sum(daily_infections)
        peak_infections = np.max(daily_infections)

        ds.close()

    except Exception as e:
        logger.error(f"Error reading NetCDF {output_path}: {e}")
        return None

    return {
        "run_id": run_path.name,
        "Profile_ID": pid,
        "Scenario": scenario_name,
        "R0": r0,
        "Strength": global_strength,
        "Duration": duration,
        "Total_Infections": total_infections,
        "Peak_Infections": peak_infections,
        "Time": time_steps,
        "Daily Infections": daily_infections,
        "Daily Hospitalized": daily_hospitalized,
        "Daily Deaths": daily_deaths,
        "Cumulative Infections": cumulative_infections,
        "Cumulative Deaths": cumulative_deaths,
    }


def calculate_relative_metrics(results):
    """Group by Profile ID and calculate metrics relative to Baseline."""
    profiles = {}
    for r in results:
        pid = r["Profile_ID"]
        if pid not in profiles:
            profiles[pid] = []
        profiles[pid].append(r)

    enriched_results = []

    for pid, runs in profiles.items():
        # Find Baseline
        baseline = next((r for r in runs if r["Scenario"] == "Baseline"), None)

        if not baseline:
            logger.warning(
                f"No Baseline found for Profile {pid}, skipping relative metrics."
            )
            continue

        baseline_total_inf = baseline["Total_Infections"]

        for r in runs:
            # Avoid division by zero
            if baseline_total_inf > 0:
                r["Rel_Total_Infections"] = r["Total_Infections"] / baseline_total_inf
                r["Reduction_Total_Infections"] = 1.0 - r["Rel_Total_Infections"]
            else:
                r["Rel_Total_Infections"] = 0.0
                r["Reduction_Total_Infections"] = 0.0

            enriched_results.append(r)

    return enriched_results


def plot_comparative_results(results, output_dir):
    """Generate and save comparative plots."""

    # Create DataFrame excluding arrays for plotting
    df = pd.DataFrame(
        [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in results]
    )

    # Filter out Baseline from the plot (Reduction is 0.0)
    df_intervention = df[df["Scenario"] != "Baseline"]

    if df_intervention.empty:
        logger.warning("No intervention data to plot.")
        return

    # 1. Relative Effectiveness Plot (Boxplot + StripPlot)
    plt.figure(figsize=(12, 6))
    # Removed alpha from boxplot as it's not supported in all versions/backends directly via kwarg in this context
    sns.boxplot(
        data=df_intervention,
        x="Scenario",
        y="Reduction_Total_Infections",
        color="lightgray",
        fliersize=0,
    )
    sns.stripplot(
        data=df_intervention,
        x="Scenario",
        y="Reduction_Total_Infections",
        hue="Profile_ID",
        palette="tab10",
        s=10,
    )
    plt.title("Effectiveness of Interventions Relative to Baseline (Twin Profiles)")
    plt.ylabel("Reduction in Total Infections (1 - Scenario/Baseline)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(title="Profile ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "synthetic_relative_effectiveness.png"))
    plt.close()

    # 2. Scatter: Strength vs Relative Reduction with Lines
    plt.figure(figsize=(10, 6))

    # Draw lines (aggregated with confidence interval)
    sns.lineplot(
        data=df_intervention,
        x="Strength",
        y="Reduction_Total_Infections",
        hue="Scenario",
        style="Scenario",
        markers=True,
        dashes=False,
        palette="viridis",
        linewidth=2.5,
        err_style="band",  # Draw confidence bands
        alpha=0.8,
    )

    # Draw individual points for detail
    sns.scatterplot(
        data=df_intervention,
        x="Strength",
        y="Reduction_Total_Infections",
        hue="Scenario",
        style="Scenario",
        s=60,
        palette="viridis",
        legend=False,
        alpha=0.6,
    )

    plt.title("Impact of Reduction Strength on Relative Infection Reduction")
    plt.xlabel("Global Reduction Strength (1 - κ₀)")
    plt.ylabel("Reduction in Total Infections")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "synthetic_relative_scatter.png"))
    plt.close()

    logger.info(f"Comparative plots saved to {output_dir}")


def create_timeseries_dataframe(results):
    """Convert list of result dicts to a long-form DataFrame for time series plotting."""
    dfs = []
    # Plotting Daily Infections, Hospitalized, and Deaths
    metrics = ["Daily Infections", "Daily Hospitalized", "Daily Deaths"]

    for r in results:
        base_data = {
            "Time": r["Time"],
            "Run": r["run_id"],
            "Profile_ID": r["Profile_ID"],
            "Scenario": r["Scenario"],
        }

        for metric in metrics:
            if metric in r:
                df = pd.DataFrame(base_data)
                df["Value"] = r[metric]
                df["Metric"] = metric
                dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def plot_epicurves_by_profile(results, output_dir):
    """Plot epicurves faceted by Profile ID (Rows) and Metric (Cols)."""
    df_ts = create_timeseries_dataframe(results)

    # Filter to ensure we have data
    if df_ts.empty:
        logger.warning("No time series data to plot.")
        return

    # FacetGrid: Rows = Profile, Cols = Metric
    g = sns.FacetGrid(
        df_ts,
        row="Profile_ID",
        col="Metric",
        hue="Scenario",
        height=2.5,
        aspect=1.5,
        palette="tab10",
        sharey=False,  # Different scales for metrics and profiles
        margin_titles=True,
    )

    g.map(sns.lineplot, "Time", "Value", alpha=0.8)
    g.add_legend()
    g.set_axis_labels("Days", "Count")
    g.set_titles(row_template="Profile {row_name}", col_template="{col_name}")
    g.fig.suptitle("Epicurves by Profile and Observable (Twin Scenarios)", y=1.01)

    plt.savefig(
        os.path.join(output_dir, "synthetic_profile_epicurves.png"), bbox_inches="tight"
    )
    plt.close()
    logger.info(f"Profile epicurves saved to {output_dir}")


if __name__ == "__main__":
    runs_dir = "../runs/synthetic_test"
    output_dir = "../runs/synthetic_test"

    logger.info(f"Scanning runs in {runs_dir}...")
    run_folders = glob.glob(os.path.join(runs_dir, "run_*"))

    results = []
    for run in run_folders:
        data = load_run_data(run)
        if data:
            results.append(data)

    if results:
        logger.info(
            f"Loaded data for {len(results)} runs. Calculating relative metrics..."
        )
        enriched_results = calculate_relative_metrics(results)

        if enriched_results:
            plot_comparative_results(enriched_results, output_dir)
            plot_epicurves_by_profile(enriched_results, output_dir)
        else:
            logger.warning("Could not calculate relative metrics (missing baselines?).")
    else:
        logger.warning("No valid run data found.")
