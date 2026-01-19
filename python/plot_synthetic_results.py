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


def parse_intervention_window_from_csv(kappa0_file):
    """Extract intervention window info from kappa0 CSV.

    Returns:
        event_start_day: int (day index)
        event_end_day: int (day index)
        strength: float (κ₀ value during intervention)
    """
    df = pd.read_csv(kappa0_file)

    # Access columns by index to avoid special character issues
    kappa_vals = df.iloc[:, 1].values  # Second column (reduction)
    times = df.iloc[:, 3].values  # Fourth column (time)

    # Find intervention window: where κ₀ > 0
    intervention_mask = kappa_vals > 0.01

    if intervention_mask.any():
        intervention_times = times[intervention_mask]
        event_start = int(intervention_times.min())
        event_end = int(intervention_times.max()) + 1
        strength = float(kappa_vals[intervention_mask].mean())
    else:
        event_start = 0
        event_end = 0
        strength = 0.0

    return event_start, event_end, strength


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

    Directory structure (new): run_{id}/config_auto_py.json, run_{id}/output/observables.nc
    Directory structure (old): run_{id}/{uuid}/config_auto_py.json, run_{id}/{uuid}/output/observables.nc
    """
    run_path = Path(run_dir)
    pid, scenario_name = parse_run_dir(run_path.name)

    if pid is None:
        logger.warning(f"Could not parse run directory name: {run_path.name}")
        return None

    # Try new structure first (no UUID nesting)
    config_path = run_path / "config_auto_py.json"
    output_path = run_path / "output" / "observables.nc"

    # Fall back to old structure (UUID subdirectory)
    if not config_path.exists() or not output_path.exists():
        subdirs = [d for d in run_path.iterdir() if d.is_dir()]
        if subdirs:
            uuid_dir = subdirs[0]
            config_path = uuid_dir / "config_auto_py.json"
            output_path = uuid_dir / "output" / "observables.nc"

    if not config_path.exists() or not output_path.exists():
        logger.warning(f"Missing config or output in {run_dir}")
        return None

    # Load Config
    with open(config_path) as f:
        config = json.load(f)

    # Extract Parameters
    try:
        r0 = config["epidemic_params"].get("scale_β", 0)
        npi = config.get("NPI", {})

        # CSV is now single source of truth - check kappa0_filename FIRST
        kappa0_file = config.get("data", {}).get("kappa0_filename", None)
        if kappa0_file and os.path.exists(kappa0_file):
            kappa0_df = pd.read_csv(kappa0_file)
            kappas = kappa0_df["reduction"].values.tolist()
            times = kappa0_df["time"].values.tolist()
            min_kappa = min(kappas)
            global_strength = min_kappa
            logger.info(f"Loaded kappa0 from file: {kappa0_file}")
        else:
            # Fallback to JSON κ₀s if no CSV (should not happen after generator fix)
            kappas = npi.get("κ₀s", [0.0])
            times = npi.get("tᶜs", [0])
            min_kappa = min(kappas)
            global_strength = min_kappa
            logger.info(f"Using JSON κ₀s: {kappas} (no CSV found)")

        # Check for custom direct mobility reduction (used by Local_Static scenarios)
        # This modifies mobility matrix, not κ₀
        custom = npi.get("custom", {})
        if (
            "direct_mobility_reduction" in custom
            and custom["direct_mobility_reduction"] > 0
        ):
            global_strength = custom["direct_mobility_reduction"]
            min_kappa = 0.0  # κ₀ stays 0.0 for Local scenarios
            logger.info(f"Using custom mobility reduction: {global_strength}")
        elif scenario_name == "Global_Timed" and len(kappas) > 1:
            # For Global_Timed, strength is max kappa (intervention level)
            global_strength = max(kappas)
            logger.debug(
                f"Global_Timed: using max_kappa={global_strength} as strength (scenario_name={scenario_name})"
            )
        else:
            logger.debug(f"Scenario {scenario_name}: using min_kappa={global_strength}")

        # Debug print kappa values before processing
        logger.debug(
            f"Scenario {scenario_name}: kappas range [{min(kappas):.3f}, {max(kappas):.3f}], final strength={global_strength:.3f}"
        )

        # Calculate duration of reduction and extract intervention window
        event_start = 0
        event_end = 0
        duration = 0

        if (
            scenario_name == "Global_Timed"
            and kappa0_file
            and os.path.exists(kappa0_file)
        ):
            # Parse intervention window from CSV
            event_start, event_end, _ = parse_intervention_window_from_csv(kappa0_file)
            duration = event_end - event_start
        elif len(kappas) >= 3 and kappas[1] < 1.0:
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

    result = {
        "run_id": run_path.name,
        "Profile_ID": pid,
        "Scenario": scenario_name,
        "R0": r0,
        "Strength": global_strength,
        "Duration": duration,
        "Event_Start": event_start,
        "Event_End": event_end,
        "Window_Center": (event_start + event_end) / 2 if event_end > 0 else 0,
        "Total_Infections": total_infections,
        "Peak_Infections": peak_infections,
        "Time": time_steps,
        "Daily Infections": daily_infections,
        "Daily Hospitalized": daily_hospitalized,
        "Daily Deaths": daily_deaths,
        "Cumulative Infections": cumulative_infections,
        "Cumulative Deaths": cumulative_deaths,
    }

    # Debug print for each run
    print(
        f"[DEBUG] {result['run_id']}: Profile={pid}, Scenario={scenario_name}, "
        f"kappas=[{min(kappas):.2f}, {max(kappas):.2f}], Strength={global_strength:.2f}, "
        f"Infections={total_infections:.1f}"
    )

    return result


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


def print_summary_stats(results):
    """Print summary statistics for all runs."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Group by scenario
    by_scenario = {}
    for r in results:
        scen = r["Scenario"]
        if scen not in by_scenario:
            by_scenario[scen] = []
        by_scenario[scen].append(r)

    for scen, runs in sorted(by_scenario.items()):
        infections = [r["Total_Infections"] for r in runs]
        strengths = [r["Strength"] for r in runs]
        print(f"\n{scen}:")
        print(f"  N runs: {len(runs)}")
        print(f"  Strengths: [{min(strengths):.2f}, {max(strengths):.2f}]")
        print(f"  Infections: [{min(infections):.1f}, {max(infections):.1f}]")
        print(f"  Mean infections: {np.mean(infections):.1f}")

    print("\n" + "=" * 80)


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


def plot_intervention_bubble(results, output_dir):
    """Bubble plot showing intervention window vs effectiveness.

    Dimensions:
        - X-axis: Intervention Start Day
        - Y-axis: Duration (days)
        - Bubble size: Intervention strength (κ₀: 0.0 to 1.0)
        - Bubble color: Infection reduction (green=effective, red=ineffective)
    """
    # Filter Global_Timed and get baseline for relative reduction
    global_timed = [r for r in results if r["Scenario"] == "Global_Timed"]
    baselines = {r["Profile_ID"]: r for r in results if r["Scenario"] == "Baseline"}

    if not global_timed or not baselines:
        logger.warning("No Global_Timed or Baseline data for bubble plot")
        return

    # Prepare plot data
    plot_data = []
    for r in global_timed:
        baseline = baselines.get(r["Profile_ID"])
        if baseline and baseline["Total_Infections"] > 0:
            # Calculate relative reduction (positive = fewer infections)
            rel_reduction = 1.0 - (r["Total_Infections"] / baseline["Total_Infections"])

            plot_data.append({
                "Start_Day": r["Event_Start"],
                "End_Day": r["Event_End"],
                "Duration": r["Duration"],
                "Strength": r["Strength"],
                "Relative_Reduction": rel_reduction,
                "Total_Infections": r["Total_Infections"],
            })

    if not plot_data:
        logger.warning("No plot data for bubble plot")
        return

    df = pd.DataFrame(plot_data)

    # Create bubble plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color map: RdYlGn (red=low reduction/ineffective, green=high reduction/effective)
    cmap = plt.get_cmap("RdYlGn")

    # Plot bubbles with horizontal error bars for window extent
    for _, row in df.iterrows():
        # Position: X = Window Center, Y = Duration
        # Size = Strength (log scale for better visibility)
        # Color = Relative Reduction
        window_center = row["Start_Day"] + row["Duration"] / 2
        y_pos = row["Duration"]

        # Bubble size: use log scale for strength (0.0 -> small, 1.0 -> large)
        bubble_size = (row["Strength"] * 500 + 50) ** 1.5

        # Color based on reduction (clip to valid range)
        color_val = np.clip(row["Relative_Reduction"], -1.0, 1.0)
        color = cmap((color_val + 1.0) / 2.0)  # Map [-1,1] to [0,1]

        scatter = ax.scatter(
            window_center, y_pos,
            s=bubble_size,
            c=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

        # Add horizontal error bar showing full window extent [start, end]
        ax.errorbar(
            window_center, y_pos,
            xerr=row["Duration"] / 2,
            fmt="none",
            ecolor="black",
            alpha=0.4,
            linewidth=1.5,
            capsize=3,
            capthick=1.5,
            zorder=2,
        )

    # Add colorbar
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=-1.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("Relative Infection Reduction\n(1 - Interventions/Baseline)", fontsize=10)

    # Labels and title
    ax.set_xlabel("Intervention Window Center (Day)", fontsize=12)
    ax.set_ylabel("Intervention Duration (days)", fontsize=12)
    ax.set_title(
        "Intervention Window Analysis: Timing vs Effectiveness\n(Bubble Size = Strength κ₀, Color = Effectiveness, Error Bars = Window Extent)",
        fontsize=14,
        pad=20,
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-5, right=120)  # Simulation is ~114 days, interventions start up to day 90
    ax.set_ylim(bottom=0, top=65)  # Duration max is ~60 days

    # Add size legend manually
    legend_elements = [
        plt.scatter([], [], s=(0 * 500 + 50) ** 1.5, c="gray", alpha=0.7, edgecolor="black", label=f"κ₀=0.0"),
        plt.scatter([], [], s=(0.5 * 500 + 50) ** 1.5, c="gray", alpha=0.7, edgecolor="black", label=f"κ₀=0.5"),
        plt.scatter([], [], s=(1.0 * 500 + 50) ** 1.5, c="gray", alpha=0.7, edgecolor="black", label=f"κ₀=1.0"),
    ]
    ax.legend(handles=legend_elements, title="Intervention Strength", loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "intervention_bubble_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Bubble plot saved to {output_dir}")


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
            plot_intervention_bubble(enriched_results, output_dir)
            print_summary_stats(results)
        else:
            logger.warning("Could not calculate relative metrics (missing baselines?).")
    else:
        logger.warning("No valid run data found.")
