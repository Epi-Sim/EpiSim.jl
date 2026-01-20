import argparse
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from synthetic_observations import (
    DEFAULT_REPORTED_CASES_CONFIG,
    DEFAULT_WASTEWATER_CONFIG,
    generate_reported_cases,
    generate_wastewater_stratified,
)

# EDAR-municipality edges for wastewater aggregation
EDAR_MUNI_EDGES_PATH = "edar_muni_edges.nc"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SyntheticOutputs")


def sanitize_run_id(run_dir_name: str) -> str:
    run_id = run_dir_name[4:] if run_dir_name.startswith("run_") else run_dir_name
    run_id = re.sub(r"[^A-Za-z0-9_-]+", "_", run_id).strip("_")
    return run_id or "run"


def parse_run_metadata(run_dir_name: str):
    raw_id = run_dir_name[4:] if run_dir_name.startswith("run_") else run_dir_name
    scenario_type = raw_id
    strength = np.nan

    strength_match = re.search(r"_s(\d+)$", raw_id)
    if strength_match:
        strength = int(strength_match.group(1)) / 100.0
        scenario_type = raw_id[: strength_match.start()]

    scenario_match = re.match(r"\d+_(.+)", scenario_type)
    if scenario_match:
        scenario_type = scenario_match.group(1)

    if scenario_type == "Baseline" and np.isnan(strength):
        strength = 0.0

    return scenario_type, strength


def load_run_artifacts(run_dir: Path):
    """
    Load config and observables from a run directory.

    Directory structure (new): run_{id}/config_auto_py.json, run_{id}/output/observables.nc
    Directory structure (old): run_{id}/{uuid}/config_auto_py.json, run_{id}/{uuid}/output/observables.nc
    """
    # Try new structure first (no UUID nesting)
    config_path = run_dir / "config_auto_py.json"
    output_path = run_dir / "output" / "observables.nc"

    # Fall back to old structure (UUID subdirectory)
    if not config_path.exists() or not output_path.exists():
        subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
        if subdirs:
            uuid_dir = subdirs[0]
            config_path = uuid_dir / "config_auto_py.json"
            output_path = uuid_dir / "output" / "observables.nc"

    if not config_path.exists() or not output_path.exists():
        logger.warning("Missing config or observables in %s", run_dir)
        return None

    with open(config_path, encoding="utf-8") as file_handle:
        config = json.load(file_handle)

    return config, output_path


def load_infections(observables_path: Path):
    ds = xr.open_dataset(observables_path)
    try:
        infections = ds["new_infected"].sum(dim="G")
        infections = infections.transpose("T", "M")
        infections_array = infections.values
    finally:
        ds.close()

    return infections_array


def load_infections_stratified(observables_path: Path):
    """Load infections keeping Age Group dimension (T, M, G)."""
    ds = xr.open_dataset(observables_path)
    try:
        # Check dimensions order in NetCDF
        # Typically: (G, M, T) or (T, M, G)
        # We need (T, M, G) for processing
        infections = ds["new_infected"]

        # Ensure correct dimension order: Time, Metapopulation, Groups
        if infections.dims != ("T", "M", "G"):
            infections = infections.transpose("T", "M", "G")

        infections_array = infections.values
    finally:
        ds.close()

    return infections_array


def load_hospitalizations(observables_path: Path):
    ds = xr.open_dataset(observables_path)
    try:
        hospitalizations = ds["new_hospitalized"].sum(dim="G")
        hospitalizations = hospitalizations.transpose("T", "M")
        hospitalizations_array = hospitalizations.values
    finally:
        ds.close()

    return hospitalizations_array


def load_deaths(observables_path: Path):
    ds = xr.open_dataset(observables_path)
    try:
        deaths = ds["new_deaths"].sum(dim="G")
        deaths = deaths.transpose("T", "M")
        deaths_array = deaths.values
    finally:
        ds.close()

    return deaths_array


def resolve_kappa0_path(config: dict, run_dir: Path):
    kappa0_path = config.get("data", {}).get("kappa0_filename")
    if not kappa0_path:
        return None
    if os.path.exists(kappa0_path):
        return kappa0_path
    candidate = run_dir / kappa0_path
    if candidate.exists():
        return str(candidate)
    return None


def load_kappa0_series(kappa0_path, dates):
    if not kappa0_path:
        return np.zeros(len(dates), dtype=float)

    kappa_df = pd.read_csv(kappa0_path)

    if "date" in kappa_df.columns:
        kappa_df["date"] = pd.to_datetime(kappa_df["date"])
        kappa_df = kappa_df.set_index("date").sort_index()
        series = kappa_df["reduction"].reindex(dates).fillna(0.0)
        return series.to_numpy(dtype=float)

    if "time" in kappa_df.columns:
        reductions = kappa_df.sort_values("time")["reduction"].to_numpy(dtype=float)
        if len(reductions) >= len(dates):
            return reductions[: len(dates)]
        padded = np.zeros(len(dates), dtype=float)
        padded[: len(reductions)] = reductions
        return padded

    logger.warning("kappa0 CSV missing date/time column: %s", kappa0_path)
    return np.zeros(len(dates), dtype=float)


def build_date_range(config: dict, time_len: int):
    start_date = config.get("simulation", {}).get("start_date")
    if not start_date:
        raise ValueError("Missing simulation.start_date in config")
    return pd.date_range(start=start_date, periods=time_len)


def load_edar_muni_mapping(metapop_df, edar_nc_path=None):
    """Load EDAR-municipality edges and build region-to-EDAR mapping.

    Args:
        metapop_df: DataFrame with metapopulation data (must have 'id' column)
        edar_nc_path: Path to EDAR-municipality edges NetCDF file

    Returns:
        dict: Mapping from metapopulation region_id to (edar_id, contribution_ratio)
              Only includes regions with EDAR coverage.
    """
    if edar_nc_path is None:
        edar_nc_path = EDAR_MUNI_EDGES_PATH

    if not os.path.exists(edar_nc_path):
        logger.warning(f"EDAR-municipality edges file not found: {edar_nc_path}")
        logger.info("Wastewater will be generated for all metapopulation regions")
        return {}

    ds = xr.open_dataset(edar_nc_path)
    edar_ids = ds["edar_id"].values
    home_ids = ds["home"].values  # Municipality codes
    contribution_matrix = ds["contribution_ratio"].values  # (EDAR, home)
    ds.close()

    # Build mapping from metapopulation region_id to (edar_id, contribution)
    region_to_edar = {}
    metapop_ids = set(metapop_df["id"].astype(str).values)

    for edar_idx, edar_id in enumerate(edar_ids):
        for home_idx, home_id in enumerate(home_ids):
            contrib = contribution_matrix[edar_idx, home_idx]
            if not np.isnan(contrib) and contrib > 0:
                # Check if this municipality is in our metapopulation
                if home_id in metapop_ids:
                    region_to_edar[home_id] = (edar_id, contrib)

    logger.info(
        f"Loaded EDAR mapping: {len(region_to_edar)} metapopulation regions have EDAR coverage"
    )
    logger.info(f"Total EDARs in file: {len(edar_ids)}")

    return region_to_edar


def aggregate_wastewater_by_edar(ww_by_region, region_ids, region_to_edar, edar_ids):
    """Aggregate wastewater signal from metapopulation regions to EDARs.

    Args:
        ww_by_region: Wastewater signal array (Time, Region) or (Time, Region, Target)
        region_ids: List of metapopulation region IDs
        region_to_edar: Mapping from region_id to (edar_id, contribution_ratio)
        edar_ids: List of all EDAR IDs

    Returns:
        Wastewater aggregated by EDAR with same shape as input but last dim is EDAR
        For (T, M) input: returns (T, EDAR)
        For (T, M, Target) input: returns (T, EDAR, Target)
    """
    # Build mapping from region index to (edar_index, contribution)
    region_idx_to_edar = []
    for i, region_id in enumerate(region_ids):
        region_id_str = str(region_id)
        if region_id_str in region_to_edar:
            edar_id, contrib = region_to_edar[region_id_str]
            try:
                edar_idx = edar_ids.index(edar_id)
                region_idx_to_edar.append((i, edar_idx, contrib))
            except ValueError:
                # EDAR ID not in list - should not happen if mapping is consistent
                pass

    if not region_idx_to_edar:
        logger.warning("No regions have EDAR coverage, returning zeros")
        n_time = ww_by_region.shape[0]
        if ww_by_region.ndim == 2:
            return np.zeros((n_time, len(edar_ids)))
        else:
            n_target = ww_by_region.shape[2]
            return np.zeros((n_time, len(edar_ids), n_target))

    # Aggregate by EDAR
    if ww_by_region.ndim == 2:
        # (Time, Region) -> (Time, EDAR)
        n_time = ww_by_region.shape[0]
        n_edar = len(edar_ids)
        ww_by_edar = np.zeros((n_time, n_edar))

        for region_idx, edar_idx, contrib in region_idx_to_edar:
            ww_by_edar[:, edar_idx] += ww_by_region[:, region_idx] * contrib

        return ww_by_edar
    else:
        # (Time, Region, Target) -> (Time, EDAR, Target)
        n_time, n_region, n_target = ww_by_region.shape
        n_edar = len(edar_ids)
        ww_by_edar = np.zeros((n_time, n_edar, n_target))

        for region_idx, edar_idx, contrib in region_idx_to_edar:
            for t in range(n_target):
                ww_by_edar[:, edar_idx, t] += ww_by_region[:, region_idx, t] * contrib

        return ww_by_edar


def plot_preview(run_id, dates, infections, wastewater_stack, output_dir, targets):
    import matplotlib.pyplot as plt

    infections_total = infections.sum(axis=1)

    # wastewater_stack is (Time, Region, Target) -> sum regions -> (Time, Target)
    ww_total_by_target = wastewater_stack.sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        dates,
        infections_total,
        label="Infections",
        color="black",
        linewidth=2,
        linestyle="--",
    )

    for i, target in enumerate(targets):
        sig = ww_total_by_target[:, i]
        if sig.max() > 0:
            # Scale for visualization against infections
            sig_scaled = sig / sig.max() * infections_total.max()
        else:
            sig_scaled = sig
        ax.plot(dates, sig_scaled, label=f"{target} (scaled)", alpha=0.7)

    ax.set_title(f"Observation Preview: {run_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Count (scaled)")
    ax.legend()
    fig.autofmt_xdate(rotation=30)

    output_path = os.path.join(output_dir, f"observation_preview_{run_id}.png")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved preview plot to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Process synthetic simulation outputs into a consolidated zarr dataset."
    )
    parser.add_argument(
        "--runs-dir",
        default="../runs/synthetic_test",
        help="Directory with run_* folders",
    )
    parser.add_argument(
        "--metapop-csv",
        required=True,
        help="Path to metapopulation_data.csv",
    )
    parser.add_argument(
        "--output",
        default="../runs/synthetic_test/synthetic_observations.zarr",
        help="Output path for zarr store",
    )
    parser.add_argument(
        "--min-rate", type=float, default=DEFAULT_REPORTED_CASES_CONFIG["min_rate"]
    )
    parser.add_argument(
        "--max-rate", type=float, default=DEFAULT_REPORTED_CASES_CONFIG["max_rate"]
    )
    parser.add_argument(
        "--inflection-day",
        type=int,
        default=DEFAULT_REPORTED_CASES_CONFIG["inflection_day"],
    )
    parser.add_argument(
        "--slope", type=float, default=DEFAULT_REPORTED_CASES_CONFIG["slope"]
    )
    parser.add_argument(
        "--gamma-shape", type=float, default=DEFAULT_WASTEWATER_CONFIG["gamma_shape"]
    )
    parser.add_argument(
        "--gamma-scale", type=float, default=DEFAULT_WASTEWATER_CONFIG["gamma_scale"]
    )
    parser.add_argument(
        "--noise-sigma", type=float, default=DEFAULT_WASTEWATER_CONFIG["noise_sigma"]
    )
    parser.add_argument(
        "--kernel-quantile",
        type=float,
        default=DEFAULT_WASTEWATER_CONFIG["kernel_quantile"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible observation noise",
    )
    parser.add_argument(
        "--preview-plot",
        action="store_true",
        help="Save a preview plot comparing infections vs wastewater",
    )
    parser.add_argument(
        "--preview-max",
        type=int,
        default=1,
        help="Number of runs to plot when --preview-plot is set",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size for region/date dimensions",
    )
    parser.add_argument(
        "--edar-edges",
        default=None,
        help="Path to EDAR-municipality edges NetCDF file for wastewater aggregation",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing Zarr store instead of overwriting",
    )

    args = parser.parse_args()

    # Define Gene Targets with biological properties
    # N1: Reference (High sensitivity, Low decay, Low LoD)
    # N2: Lower sensitivity, Higher noise, Medium LoD
    # IP4: Lowest sensitivity, High decay, High LoD
    #
    # Calibrated to Physical Units (Copies/L)
    # Scale = 500,000 (10^8 copies/day / 200L flow)
    # LoD = 375 Copies/L (Median from dataset)
    # Transport Loss = 50 Copies/L (Background decay)
    GENE_TARGETS = {
        "N1": {
            "sensitivity_scale": 500000.0,
            "noise_sigma": 0.5,
            "limit_of_detection": 375.0,
            "transport_loss": 50.0,
            "lod_probabilistic": True,
            "lod_slope": 1.5,
        },
        "N2": {
            "sensitivity_scale": 400000.0,
            "noise_sigma": 0.8,
            "limit_of_detection": 500.0,
            "transport_loss": 100.0,
            "lod_probabilistic": True,
            "lod_slope": 1.5,
        },
        "IP4": {
            "sensitivity_scale": 250000.0,
            "noise_sigma": 0.6,
            "limit_of_detection": 800.0,
            "transport_loss": 200.0,
            "lod_probabilistic": True,
            "lod_slope": 1.5,
        },
    }
    TARGET_NAMES = list(GENE_TARGETS.keys())

    runs_dir = Path(args.runs_dir)
    run_dirs = sorted([d for d in runs_dir.glob("run_*") if d.is_dir()])

    if not run_dirs:
        raise ValueError(f"No run_* folders found in {runs_dir}")

    metapop_df = pd.read_csv(args.metapop_csv)
    # Ensure ID is string to match
    metapop_df["id"] = metapop_df["id"].astype(str)
    region_ids = metapop_df["id"].tolist()

    # Extract Population Vector for Dilution
    # Aligned with region_ids
    population_vector = metapop_df["total"].values.astype(float)

    # Load EDAR-municipality mapping for wastewater aggregation
    # Check if EDAR edges file is provided or use default path
    edar_path = args.edar_edges
    if edar_path is None:
        # Try default location (project root)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        default_edar_path = os.path.join(project_root, EDAR_MUNI_EDGES_PATH)
        if os.path.exists(default_edar_path):
            edar_path = default_edar_path

    region_to_edar = load_edar_muni_mapping(metapop_df, edar_nc_path=edar_path)
    # Extract EDAR IDs from the mapping
    if region_to_edar:
        edar_ids = sorted(set(edar_id for edar_id, _ in region_to_edar.values()))
    else:
        # Fallback to region-based wastewater if no EDAR mapping
        edar_ids = region_ids[:]
        logger.warning("No EDAR mapping found, using region-based wastewater")

    infections_runs = []
    hospitalizations_runs = []
    deaths_runs = []
    cases_runs = []
    wastewater_runs = []
    ascertainment_runs = []
    mobility_runs = []
    scenario_types = []
    strengths = []
    run_ids = []

    dates_ref = None

    rng = np.random.default_rng(args.seed)

    reported_cfg = {
        "min_rate": args.min_rate,
        "max_rate": args.max_rate,
        "inflection_day": args.inflection_day,
        "slope": args.slope,
    }
    wastewater_cfg = {
        "gamma_shape": args.gamma_shape,
        "gamma_scale": args.gamma_scale,
        "noise_sigma": args.noise_sigma,
        "kernel_quantile": args.kernel_quantile,
    }

    preview_count = 0

    for run_dir in run_dirs:
        artifacts = load_run_artifacts(run_dir)
        if artifacts is None:
            continue

        config, observables_path = artifacts
        infections_stratified = load_infections_stratified(observables_path)
        # Sum over ages for reporting cases (standard reporting doesn't distinguish source)
        infections_total = infections_stratified.sum(axis=2)

        hospitalizations = load_hospitalizations(observables_path)
        deaths = load_deaths(observables_path)

        if infections_total.shape[1] != len(region_ids):
            raise ValueError(
                f"Region count mismatch for {run_dir}: "
                f"expected {len(region_ids)}, got {infections_total.shape[1]}"
            )

        dates = build_date_range(config, infections_total.shape[0])
        if dates_ref is None:
            dates_ref = dates
        elif not dates.equals(dates_ref):
            raise ValueError(f"Date range mismatch for {run_dir}")

        case_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        ww_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

        reported_cases, ascertainment_rate = generate_reported_cases(
            infections_total, config=reported_cfg, rng=case_rng
        )

        # Generate Wastewater for each target using Stratified Model
        ww_target_layers = []
        for target in TARGET_NAMES:
            t_cfg = GENE_TARGETS[target]
            run_ww_cfg = wastewater_cfg.copy()
            run_ww_cfg.update(t_cfg)

            # Use Stratified Generator
            # Passes: Infections(T, M, G), Population(M)
            # Returns: (T, M) concentration in Copies/L
            ww = generate_wastewater_stratified(
                infections_stratified,
                population=population_vector,
                config=run_ww_cfg,
                rng=ww_rng,
            )
            ww_target_layers.append(ww)

        # Stack targets: (Time, Region, Target)
        ww_stacked = np.stack(ww_target_layers, axis=-1)

        # Aggregate wastewater by EDAR using contribution ratios
        # ww_stacked is (Time, Region, Target) -> (Time, EDAR, Target)
        if region_to_edar:
            ww_by_edar = aggregate_wastewater_by_edar(
                ww_stacked, region_ids, region_to_edar, edar_ids
            )
        else:
            # Fallback: use region-based wastewater
            ww_by_edar = ww_stacked  # (Time, Region, Target)

        if args.preview_plot and preview_count < args.preview_max:
            # For preview, ww_by_edar is already in the correct format (Time, Spatial, Target)
            # whether using EDAR or region-based aggregation
            plot_preview(
                sanitize_run_id(run_dir.name),
                dates,
                infections_total,
                ww_by_edar,
                runs_dir,
                TARGET_NAMES,
            )
            preview_count += 1

        infections_runs.append(infections_total.T.astype(int))
        hospitalizations_runs.append(hospitalizations.T.astype(int))
        deaths_runs.append(deaths.T.astype(int))
        cases_runs.append(reported_cases.T.astype(int))

        # wastewater_runs wants (EDAR, Time, Target) when using EDAR, or (Region, Time, Target) otherwise
        # ww_by_edar is (Time, EDAR, Target).
        # Transpose: 1 -> 0, 0 -> 1, 2 -> 2
        ww_transposed = ww_by_edar.transpose(1, 0, 2)
        wastewater_runs.append(ww_transposed.astype(float))

        ascertainment_runs.append(ascertainment_rate.astype(float))

        kappa0_path = resolve_kappa0_path(config, run_dir)
        mobility_reduction = load_kappa0_series(kappa0_path, dates)
        mobility_runs.append(mobility_reduction.astype(float))

        scenario_type, strength = parse_run_metadata(run_dir.name)
        if np.isnan(strength):
            strength = float(np.nanmax(mobility_reduction))
        scenario_types.append(scenario_type)
        strengths.append(strength)
        run_ids.append(sanitize_run_id(run_dir.name))

    if not infections_runs:
        raise ValueError("No valid runs were processed")

    # Stack runs: (Run, Region, Time, ...) or (Run, EDAR, Time, ...) for wastewater
    infections_arr = np.stack(infections_runs, axis=0)
    hospitalizations_arr = np.stack(hospitalizations_runs, axis=0)
    deaths_arr = np.stack(deaths_runs, axis=0)
    cases_arr = np.stack(cases_runs, axis=0)
    # wastewater_arr: (Run, EDAR, Time, Target) when using EDAR, else (Run, Region, Time, Target)
    wastewater_arr = np.stack(wastewater_runs, axis=0)
    ascertainment_arr = np.stack(ascertainment_runs, axis=0)
    mobility_arr = np.stack(mobility_runs, axis=0)

    # Determine wastewater spatial dimension
    if region_to_edar:
        ww_spatial_dim = "edar_id"
        logger.info(f"Using EDAR-based wastewater: {len(edar_ids)} EDARs")
    else:
        ww_spatial_dim = "region_id"
        logger.info(f"Using region-based wastewater: {len(region_ids)} regions")

    dataset = xr.Dataset(
        {
            "ground_truth_infections": (
                ("run_id", "region_id", "date"),
                infections_arr,
            ),
            "obs_hospitalizations": (
                ("run_id", "region_id", "date"),
                hospitalizations_arr,
            ),
            "obs_deaths": (
                ("run_id", "region_id", "date"),
                deaths_arr,
            ),
            "obs_cases": (
                ("run_id", "region_id", "date"),
                cases_arr,
            ),
            "obs_wastewater": (
                ("run_id", ww_spatial_dim, "date", "target"),
                wastewater_arr,
            ),
            "ascertainment_rate": (
                ("run_id", "date"),
                ascertainment_arr,
            ),
            "mobility_reduction": (
                ("run_id", "date"),
                mobility_arr,
            ),
        },
        coords={
            "run_id": run_ids,
            "region_id": region_ids,
            "date": dates_ref,
            "target": TARGET_NAMES,
            "edar_id": edar_ids if region_to_edar else region_ids,
        },
    )

    dataset = dataset.assign_coords(
        {
            "scenario_type": ("run_id", scenario_types),
            "strength": ("run_id", strengths),
        }
    )

    if dates_ref is None:
        raise ValueError("No valid dates found in run outputs")

    chunk_size = args.chunk_size
    region_chunk = min(chunk_size, len(region_ids))
    date_chunk = min(chunk_size, len(dates_ref))

    for var_name in [
        "ground_truth_infections",
        "obs_hospitalizations",
        "obs_deaths",
        "obs_cases",
    ]:
        dataset[var_name].encoding = {"chunksizes": (1, region_chunk, date_chunk)}

    # Wastewater chunking depends on spatial dimension (EDAR or region)
    if region_to_edar:
        edar_chunk = min(chunk_size, len(edar_ids))
        dataset["obs_wastewater"].encoding = {
            "chunksizes": (1, edar_chunk, date_chunk, len(TARGET_NAMES))
        }
    else:
        dataset["obs_wastewater"].encoding = {
            "chunksizes": (1, region_chunk, date_chunk, len(TARGET_NAMES))
        }

    for var_name in ["ascertainment_rate", "mobility_reduction"]:
        dataset[var_name].encoding = {"chunksizes": (1, date_chunk)}

    output_path = args.output

    if args.append and os.path.exists(output_path):
        logger.info(
            "Appending to observation zarr at %s (runs=%s)",
            output_path,
            len(run_ids),
        )
        dataset.to_zarr(output_path, mode="a", append_dim="run_id")
    else:
        if os.path.exists(output_path):
            logger.info("Removing existing output at %s", output_path)
            if os.path.isdir(output_path):
                import shutil

                shutil.rmtree(output_path)
            else:
                os.remove(output_path)

        logger.info(
            "Writing observation zarr to %s (runs=%s, regions=%s, dates=%s, targets=%s, edars=%s)",
            output_path,
            len(run_ids),
            len(region_ids),
            len(dates_ref),
            len(TARGET_NAMES),
            len(edar_ids) if region_to_edar else 0,
        )
        dataset.to_zarr(output_path, mode="w")


if __name__ == "__main__":
    main()
