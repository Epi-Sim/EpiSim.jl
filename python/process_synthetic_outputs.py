"""
Process synthetic simulation outputs into a raw observation zarr dataset.

This script generates "raw-ish" synthetic observations that mimic real-world data sources,
with configurable noise, gaps, and censoring patterns. The output is designed to be
consumed by EpiForecaster's preprocessing pipeline (Tobit Kalman, alignment, etc.).

Key Design Principle:
    The output should be raw enough that EpiForecaster's preprocessing pipeline
    does meaningful work (interpolation, smoothing, censoring), but structured
    enough to match the expected input format.

Output format:
    cases: (run_id, date, region_id) - Raw cases with NaN for missing (matches EpiForecaster input)
    edar_biomarker_N1, edar_biomarker_N2, edar_biomarker_IP4: (run_id, date, edar_id) - Wastewater in log1p space
        Note: Raw wastewater values are log1p-transformed (log1p(concentration)) to enable efficient
        float16 storage. Raw values can be 150k+ which exceeds float16 max (65504).
        Example: log1p(150000) ≈ 11.9, log1p(375) ≈ 5.93 (LoD for N1)
    edar_biomarker_*_censor_hints: (run_id, date, edar_id) - Censoring hints (0=observed, 1=censored, 2=missing)

    # Mobility: Two storage formats depending on mobility type
    # When all runs are static (default, memory-efficient):
    mobility_base: (origin, target) - Base mobility matrix (shared across all runs)
    mobility_kappa0: (run_id, date) - Mobility reduction factor per run and date
        Reconstructed as: mobility[run, date] = mobility_base * (1 - mobility_kappa0[run, date])
    synthetic_mobility_type: (run_id,) - All "static"

    # When any run has time-varying mobility (mobility_sigma_O/D > 0):
    mobility_time_varying: (run_id, origin, target, date) - Dense OD matrices for each timestep
    synthetic_mobility_type: (run_id,) - "static" or "time_varying" per run

    population: (run_id, region_id) - Static population (matches EpiForecaster input)

    # Ground truth for evaluation (not passed to preprocessor)
    infections_true: (run_id, date, region_id)
    hospitalizations_true: (run_id, date, region_id)
    deaths_true: (run_id, date, region_id)
    latent_S_true, latent_E_true, latent_A_true, latent_I_true, latent_R_true, latent_D_true:
        (run_id, region_id, date) - Optional latent simulator states for hybrid supervision
    latent_CH_true, latent_hospitalized_true, latent_active_true:
        (run_id, region_id, date) - Optional auxiliary latent targets

    # Synthetic metadata (for reference, not used by preprocessor)
    synthetic_scenario_type: (run_id,) - Scenario type (Baseline, Global_Timed, Local_Static)
    synthetic_strength: (run_id,) - Intervention strength
    synthetic_sparsity_level: (run_id,) - Missing data rate used
"""

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit

from synthetic_observations import (
    DEFAULT_DEATHS_REPORT_CONFIG,
    DEFAULT_HOSP_REPORT_CONFIG,
    DEFAULT_REPORTED_CASES_CONFIG,
    DEFAULT_WASTEWATER_CONFIG,
    generate_reported_cases,
    generate_reported_with_delay,
    generate_wastewater_stratified,
)

# EDAR-municipality edges for wastewater aggregation
# Path relative to project root, placed in python/data/
EDAR_MUNI_EDGES_PATH = os.path.join("python", "data", "edar_muni_edges.nc")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SyntheticOutputs")

LATENT_ZARR_VARS = (
    "latent_S_true",
    "latent_E_true",
    "latent_A_true",
    "latent_I_true",
    "latent_R_true",
    "latent_D_true",
    "latent_CH_true",
    "latent_hospitalized_true",
    "latent_active_true",
)


def safe_convert_dtype(arr: np.ndarray, dtype_str: str, var_name: str) -> np.ndarray:
    """
    Safely convert array to target dtype, validating no overflow for float16.

    Args:
        arr: Input array to convert
        dtype_str: Target dtype string ("float16", "float32", "float64")
        var_name: Variable name for error messages

    Returns:
        Converted array

    Raises:
        ValueError: If float16 overflow would occur
    """
    if dtype_str == "float16":
        max_val = np.finfo(np.float16).max  # 65504
        arr_max = np.nanmax(np.abs(arr))
        if arr_max > max_val:
            raise ValueError(
                f"Cannot convert {var_name} to float16: "
                f"max absolute value {arr_max} exceeds limit {max_val}"
            )
    return arr.astype(dtype_str)


def transform_wastewater_log1p(arr: np.ndarray) -> np.ndarray:
    """
    Apply log1p transformation and convert to float16 for efficient storage.

    This transformation compresses large wastewater concentration values (e.g., 150k)
    into a range suitable for float16 (0-20). Log1p(150000) ≈ 11.9.

    Args:
        arr: Raw wastewater array with values in Copies/L (can be large, up to ~500k)

    Returns:
        Log1p-transformed array as float16, suitable for efficient storage
    """
    log_arr = np.log1p(arr)
    return log_arr.astype(np.float16)


def sanitize_run_id(run_dir_name: str, max_length: int = 50) -> str:
    """
    Sanitize run directory name into a valid run_id.

    Pads to max_length to avoid dtype mismatches when appending to zarr stores,
    which have fixed-size string dtypes.
    """
    run_id = run_dir_name[4:] if run_dir_name.startswith("run_") else run_dir_name
    run_id = re.sub(r"[^A-Za-z0-9_-]+", "_", run_id).strip("_")
    run_id = run_id or "run"
    # Pad to fixed length to avoid dtype mismatches when appending
    return run_id.ljust(max_length)[:max_length]


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
    compartments_path = run_dir / "output" / "compartments_full.nc"

    # Fall back to old structure (UUID subdirectory)
    if not config_path.exists() or not output_path.exists():
        subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
        if subdirs:
            uuid_dir = subdirs[0]
            config_path = uuid_dir / "config_auto_py.json"
            output_path = uuid_dir / "output" / "observables.nc"
            compartments_path = uuid_dir / "output" / "compartments_full.nc"

    if not config_path.exists() or not output_path.exists():
        logger.warning("Missing config or observables in %s", run_dir)
        return None

    with open(config_path, encoding="utf-8") as file_handle:
        config = json.load(file_handle)

    return config, output_path, compartments_path if compartments_path.exists() else None


def load_infections_stratified(observables_path: Path):
    """Load infections keeping Age Group dimension (T, M, G)."""
    ds = xr.open_dataset(observables_path)
    try:
        infections = ds["new_infected"]
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


def load_compartment_latents(compartments_path: Path) -> dict[str, np.ndarray]:
    """Load latent compartment trajectories aggregated to (region, date)."""
    if compartments_path is None or not compartments_path.exists():
        raise FileNotFoundError(
            f"Missing compartments_full.nc required for latent export: {compartments_path}"
        )

    with xr.open_dataset(compartments_path) as ds:
        required = {"S", "E", "A", "I", "PH", "PD", "HR", "HD", "R", "D", "CH"}
        missing = sorted(required.difference(ds.data_vars))
        if missing:
            raise ValueError(
                f"Missing latent compartment variables in {compartments_path}: {missing}"
            )

        def load_state(var_name: str) -> np.ndarray:
            arr = ds[var_name]
            if set(arr.dims) != {"G", "M", "T"}:
                raise ValueError(
                    f"Unexpected dims for {var_name} in {compartments_path}: {arr.dims}"
                )
            return arr.transpose("M", "T", "G").values

        states = {name: load_state(name) for name in required}

    hospitalized = states["HR"] + states["HD"]
    active = (
        states["E"]
        + states["A"]
        + states["I"]
        + states["PH"]
        + states["PD"]
        + states["HR"]
        + states["HD"]
    )

    latents = {
        "latent_S_true": states["S"].sum(axis=2),
        "latent_E_true": states["E"].sum(axis=2),
        "latent_A_true": states["A"].sum(axis=2),
        "latent_I_true": states["I"].sum(axis=2),
        "latent_R_true": states["R"].sum(axis=2),
        "latent_D_true": states["D"].sum(axis=2),
        "latent_CH_true": states["CH"].sum(axis=2),
        "latent_hospitalized_true": hospitalized.sum(axis=2),
        "latent_active_true": active.sum(axis=2),
    }
    return {name: values.astype(np.float32) for name, values in latents.items()}


# Global variable for worker processes to share large data objects
# This avoids pickling and sending these objects for every task in the process pool
_shared_worker_data = {}


def init_worker(region_ids, population_vector, emap_data, gene_targets, reported_cfg, wastewater_cfg, args_dict):
    """Initialize a worker process with shared data."""
    global _shared_worker_data
    _shared_worker_data["region_ids"] = region_ids
    _shared_worker_data["population_vector"] = population_vector
    _shared_worker_data["emap_data"] = emap_data
    _shared_worker_data["gene_targets"] = gene_targets
    _shared_worker_data["reported_cfg"] = reported_cfg
    _shared_worker_data["wastewater_cfg"] = wastewater_cfg
    _shared_worker_data["args_dict"] = args_dict


def process_single_run(
    run_dir_path: str,
    sparsity: float,
    seed: Optional[int] = None,
) -> Optional[dict]:
    """
    Process a single run directory and return processed data.

    This function uses shared data from the global _shared_worker_data dict
    to avoid IPC overhead.

    Args:
        run_dir_path: Path to run directory as string
        sparsity: Sparsity level for this run
        seed: Random seed for reproducibility

    Returns:
        Dict with processed data or None if failed
    """
    global _shared_worker_data
    region_ids = _shared_worker_data["region_ids"]
    population_vector = _shared_worker_data["population_vector"]
    emap_data = _shared_worker_data["emap_data"]
    gene_targets = _shared_worker_data["gene_targets"]
    reported_cfg = _shared_worker_data["reported_cfg"]
    wastewater_cfg = _shared_worker_data["wastewater_cfg"]
    args_dict = _shared_worker_data["args_dict"]

    run_dir = Path(run_dir_path)

    # Load artifacts
    artifacts = load_run_artifacts(run_dir)
    if artifacts is None:
        return None

    config, observables_path, compartments_path = artifacts

    # Load data
    infections_stratified = load_infections_stratified(observables_path)
    infections_total = infections_stratified.sum(axis=2)
    hospitalizations = load_hospitalizations(observables_path)
    deaths = load_deaths(observables_path)
    include_latents = args_dict.get("include_latents", False)
    latents = (
        load_compartment_latents(compartments_path) if include_latents else None
    )

    # Validate
    if infections_total.shape[1] != len(region_ids):
        raise ValueError(
            f"Region count mismatch for {run_dir}: "
            f"expected {len(region_ids)}, got {infections_total.shape[1]}"
        )

    # Build dates
    dates = build_date_range(config, infections_total.shape[0])

    # Create RNGs
    rng = np.random.default_rng(seed)
    case_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
    hosp_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
    deaths_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
    ww_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

    # Generate reported cases (total and age-stratified)
    reported_cases, _ = generate_reported_cases(
        infections_total, config=reported_cfg, rng=case_rng
    )
    cases_with_gaps, cases_mask = apply_missing_data_patterns(
        reported_cases.T,
        missing_rate=sparsity,
        missing_gap_length=args_dict.get("missing_gap_length", 3),
        rng=case_rng,
    )

    # Generate age-stratified reported cases
    n_time, n_regions, n_age_groups = infections_stratified.shape
    cases_age_stratified = np.zeros((n_time, n_regions, n_age_groups), dtype=int)
    for g in range(n_age_groups):
        reported_cases_g, _ = generate_reported_cases(
            infections_stratified[:, :, g],
            config=reported_cfg,
            rng=np.random.default_rng(case_rng.integers(0, 2**32 - 1)),
        )
        cases_age_stratified[:, :, g] = reported_cases_g

    # Generate wastewater
    if emap_data is not None:
        infections_edar_strat, population_edar = aggregate_infections_to_edar(
            infections_stratified, emap_data["emap"], population_vector
        )
        from synthetic_observations import _compute_monitoring_start_mask

        monitoring_mask = _compute_monitoring_start_mask(
            infections_edar_strat,
            threshold=args_dict.get("monitoring_threshold", 0.0),
            delay_days=args_dict.get("monitoring_delay_mean", 0),
            delay_std=args_dict.get("monitoring_delay_std", 0),
            rng=ww_rng,
        )
        ww_by_edar, censor_by_edar = generate_wastewater_with_censoring(
            infections_edar_strat,
            population_edar,
            gene_targets,
            wastewater_cfg,
            rng=ww_rng,
            monitoring_mask=monitoring_mask,
            missing_rate=sparsity,
        )
    else:
        from synthetic_observations import _compute_monitoring_start_mask

        monitoring_mask = _compute_monitoring_start_mask(
            infections_stratified,
            threshold=args_dict.get("monitoring_threshold", 0.0),
            delay_days=args_dict.get("monitoring_delay_mean", 0),
            delay_std=args_dict.get("monitoring_delay_std", 0),
            rng=ww_rng,
        )
        ww_by_edar, censor_by_edar = generate_wastewater_with_censoring(
            infections_stratified,
            population_vector,
            gene_targets,
            wastewater_cfg,
            rng=ww_rng,
            monitoring_mask=monitoring_mask,
            missing_rate=sparsity,
        )

    # Generate reported hospitalizations and deaths
    reported_hospitalizations = generate_reported_with_delay(
        hospitalizations,
        report_rate=args_dict.get("hosp_report_rate", 0.85),
        delay_mean=args_dict.get("hosp_delay_mean", 3),
        delay_std=args_dict.get("hosp_delay_std", 1),
        rng=hosp_rng,
    )
    reported_deaths = generate_reported_with_delay(
        deaths,
        report_rate=args_dict.get("deaths_report_rate", 0.90),
        delay_mean=args_dict.get("deaths_delay_mean", 7),
        delay_std=args_dict.get("deaths_delay_std", 2),
        rng=deaths_rng,
    )

    # Load kappa0 and mobility
    kappa0_path = resolve_kappa0_path(config, run_dir)
    kappa0_series = load_kappa0_series(kappa0_path, dates)

    # Determine mobility type
    mobility_npz_path = run_dir / "mobility" / "mobility_series.npz"
    mobility_type = "time_varying" if mobility_npz_path.exists() else "static"
    sigma_O, sigma_D = load_mobility_noise_params(run_dir)

    # Parse metadata
    scenario_type, strength = parse_run_metadata(run_dir.name)
    if np.isnan(strength):
        strength = float(np.nanmax(kappa0_series)) if len(kappa0_series) > 0 else 0.0

    # Save generated matrices directly to disk to prevent massive IPC transfers
    # back to the main process, which avoids OOM on high concurrency
    npz_path = run_dir / "processed_obs.npz"
    np.savez_compressed(
        npz_path,
        infections_true=infections_total.T.astype(int),
        hospitalizations_true=hospitalizations.T.astype(int),
        deaths_true=deaths.T.astype(int),
        hospitalizations_raw=reported_hospitalizations.T.astype(int),
        deaths_raw=reported_deaths.T.astype(int),
        cases_raw=cases_with_gaps.astype(float),
        cases_mask=cases_mask.astype(float),
        cases_age=cases_age_stratified.astype(int),
        wastewater_raw=ww_by_edar.astype(float),
        wastewater_censor=censor_by_edar.astype(np.int8),
        mobility_kappa0=kappa0_series.astype(float),
        **({name: values for name, values in latents.items()} if latents else {}),
    )

    return {
        "run_dir_path": str(run_dir),
        "run_id": sanitize_run_id(run_dir.name),
        "scenario_type": scenario_type,
        "strength": strength,
        "sparsity": sparsity,
        "dates": dates,
        "mobility_type": mobility_type,
        "mobility_sigma_O": sigma_O,
        "mobility_sigma_D": sigma_D,
        "mobility_noise": max(sigma_O, sigma_D),
        "npz_path": str(npz_path),
    }


def load_mobility_matrix(metapop_df: Path) -> np.ndarray:
    """
    Load the base mobility matrix and convert from sparse edge list to dense format.

    Args:
        metapop_df: DataFrame with metapopulation data (must have 'id' column)

    Returns:
        Dense mobility matrix of shape (n_regions, n_regions)
    """
    # Look for mobility matrix in the same directory as metapopulation data
    parent_dir = Path(metapop_df).parent
    mobility_path = parent_dir / "mobility_matrix.csv"

    if not mobility_path.exists():
        mobility_path = parent_dir / "R_mobility_matrix.csv"

    if not mobility_path.exists():
        raise ValueError(f"Mobility matrix not found in {parent_dir}")

    logger.info(f"Loading mobility matrix from {mobility_path}")

    # Read sparse edge list format
    mobility_df = pd.read_csv(mobility_path)

    # Get number of regions from metapop data
    metapop_data = pd.read_csv(metapop_df, dtype={"id": str})
    n_regions = len(metapop_data)

    # Convert to dense matrix
    # The CSV uses 1-indexed source_idx/target_idx
    mobility_dense = np.zeros((n_regions, n_regions), dtype=float)

    for _, row in mobility_df.iterrows():
        src = int(row["source_idx"]) - 1  # Convert to 0-indexed
        tgt = int(row["target_idx"]) - 1
        if 0 <= src < n_regions and 0 <= tgt < n_regions:
            mobility_dense[src, tgt] = row["ratio"]

    # Normalize rows to sum to 1 (destination probabilities)
    row_sums = mobility_dense.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    mobility_dense = mobility_dense / row_sums

    logger.info(f"Loaded mobility matrix: {mobility_dense.shape}")
    return mobility_dense


def apply_missing_data_patterns(
    data: np.ndarray,
    missing_rate: float = 0.05,
    missing_gap_length: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply realistic missing data patterns to observations.

    Simulates:
    1. Random missing values (sparse, distributed)
    2. Gap-based missing data (consecutive days, e.g., system outages)

    Args:
        data: Input data array (time, regions) or (time,)
        missing_rate: Fraction of data to make missing
        missing_gap_length: Average length of missing gaps
        rng: Random number generator

    Returns:
        data_with_gaps: Data with NaN values inserted
        mask: 1.0 where observed, 0.0 where missing
    """
    rng = rng or np.random.default_rng()

    data_with_gaps = data.copy().astype(float)
    mask = np.ones_like(data_with_gaps, dtype=float)

    if data.ndim == 1:
        data_with_gaps = data_with_gaps[:, None]
        mask = mask[:, None]
        squeeze_output = True
    else:
        squeeze_output = False

    n_time, n_regions = data_with_gaps.shape

    # Pattern 1: Random sparse missing values
    n_random_missing = int(n_time * n_regions * missing_rate * 0.5)
    if n_random_missing > 0:
        random_indices = rng.choice(n_time * n_regions, n_random_missing, replace=False)
        random_t, random_r = np.unravel_index(random_indices, (n_time, n_regions))
        data_with_gaps[random_t, random_r] = np.nan
        mask[random_t, random_r] = 0.0

    # Pattern 2: Gap-based missing (e.g., weekend gaps, system outages)
    n_gaps = int(n_time * missing_rate * 0.5 / missing_gap_length) + 1
    for _ in range(n_gaps):
        gap_start = rng.integers(0, max(1, n_time - missing_gap_length))
        gap_length = rng.integers(1, missing_gap_length * 2 + 1)
        gap_end = min(n_time, gap_start + gap_length)

        # Apply gap to a random subset of regions
        n_regions_affected = rng.integers(1, max(2, n_regions // 10))
        regions_affected = rng.choice(n_regions, n_regions_affected, replace=False)

        data_with_gaps[gap_start:gap_end, regions_affected] = np.nan
        mask[gap_start:gap_end, regions_affected] = 0.0

    if squeeze_output:
        return data_with_gaps[:, 0], mask[:, 0]
    return data_with_gaps, mask


def generate_wastewater_with_censoring(
    infections_stratified: np.ndarray,
    population: np.ndarray,
    gene_targets: dict,
    wastewater_cfg: dict,
    rng: Optional[np.random.Generator] = None,
    monitoring_mask: Optional[np.ndarray] = None,
    missing_rate: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate wastewater observations with censoring flags.

    Args:
        infections_stratified: (Time, Region, AgeGroups) array of infections
        population: (Region,) array of population values
        gene_targets: Dict of gene target configurations
        wastewater_cfg: Dict of wastewater generation parameters
        rng: Random number generator
        monitoring_mask: (Time, Region) bool array for delayed monitoring
        missing_rate: Fraction of wastewater data to make missing (default: 0.02)

    Returns:
        wastewater: (Time, Region, Target) array of observations
        censor_flags: (Time, Region, Target) array of flags
            0 = observed (above LoD)
            1 = censored (below LoD, but measurement exists)
            2 = missing (no measurement)
            3 = not yet monitoring (pre-threshold)
    """
    rng = rng or np.random.default_rng()
    n_time, n_regions = infections_stratified.shape[0], infections_stratified.shape[1]
    n_targets = len(gene_targets)

    wastewater = np.zeros((n_time, n_regions, n_targets), dtype=float)
    censor_flags = np.zeros((n_time, n_regions, n_targets), dtype=np.int8)

    for i, (_, target_config) in enumerate(gene_targets.items()):
        cfg = {**wastewater_cfg, **target_config}

        # Generate raw wastewater signal (without LoD applied)
        # Temporarily disable LoD to get the raw signal
        cfg_no_lod = cfg.copy()
        cfg_no_lod["limit_of_detection"] = 0.0
        cfg_no_lod["lod_probabilistic"] = False

        ww_signal = generate_wastewater_stratified(
            infections_stratified,
            population=population,
            config=cfg_no_lod,
            rng=rng,
            monitoring_mask=monitoring_mask,
        )

        lod = cfg["limit_of_detection"]

        # Apply censoring
        if cfg["lod_probabilistic"] and lod > 0.0:
            k = cfg.get("lod_slope", 2.0)
            # Use numerically stable sigmoid (expit) to avoid overflow
            # detection_prob = 1 / (1 + exp(-k * (ww_signal - lod)))
            # = sigmoid(k * (ww_signal - lod))
            detection_prob = expit(k * (ww_signal - lod))
            is_detected = rng.random(size=ww_signal.shape) < detection_prob
            censor_flags[:, :, i] = np.where(is_detected, 0, 1)
            ww_signal[~is_detected] = lod  # Set censored values to LoD (not NaN)
        elif lod > 0.0:
            censor_flags[:, :, i] = np.where(ww_signal >= lod, 0, 1)
            ww_signal[ww_signal < lod] = lod  # Set censored values to LoD (not NaN)
        else:
            censor_flags[:, :, i] = 0  # All observed

        # Add some missing data (use parameterizable rate)
        missing_mask = rng.random(ww_signal.shape) < missing_rate
        ww_signal[missing_mask] = np.nan
        censor_flags[missing_mask] = 2  # Missing flag

        # Apply monitoring mask to censor flags (value 3 = not yet monitoring)
        if monitoring_mask is not None:
            # monitoring_mask shape: (Time, Region)
            # censor_flags[:, :, i] shape: (Time, Region)
            # Set pre-threshold values to 3
            censor_flags[:, :, i][~monitoring_mask] = 3

        wastewater[:, :, i] = ww_signal

    return wastewater, censor_flags


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


def detect_and_load_time_varying_mobility(
    run_dir: Path, n_regions: int, n_dates: int
) -> Optional[np.ndarray]:
    """
    Detect if time-varying mobility was generated and load it as dense OD matrices.

    When mobility_sigma_O or mobility_sigma_D > 0, the synthetic generator creates
    time-varying mobility using IPFP and saves it to mobility/mobility_series.npz.
    This function detects and loads that data in dense format for zarr storage.

    Args:
        run_dir: Run directory path
        n_regions: Number of regions (M)
        n_dates: Number of dates (T)

    Returns:
        mobility_dense: (T, M, M) array of dense OD matrices, or None if static
    """
    mobility_npz_path = run_dir / "mobility" / "mobility_series.npz"

    if not mobility_npz_path.exists():
        return None  # Static mobility

    try:
        data = np.load(mobility_npz_path)
        R_series = data["R_series"]  # (T, E) sparse weights
        edgelist = data["edgelist"]  # (E, 2) with [origin, destination]

        # Convert sparse (T, E) to dense (T, M, M)
        T, _ = R_series.shape
        mobility_dense = np.zeros((T, n_regions, n_regions), dtype=np.float16)

        for t in range(T):
            for edge_idx, (src, tgt) in enumerate(edgelist):
                if src < n_regions and tgt < n_regions:
                    mobility_dense[t, src, tgt] = R_series[t, edge_idx]

        logger.info(
            f"Loaded time-varying mobility from {mobility_npz_path}: shape={mobility_dense.shape}"
        )
        return mobility_dense

    except Exception as e:
        logger.warning(f"Failed to load mobility series from {mobility_npz_path}: {e}")
        return None


def load_mobility_noise_params(run_dir: Path) -> tuple[float, float]:
    """
    Load mobility noise sigmas (origin/destination) from mobility_series.npz if present.

    Returns:
        (sigma_O, sigma_D): floats, defaults to 0.0 if static or missing.
    """
    mobility_npz_path = run_dir / "mobility" / "mobility_series.npz"
    if not mobility_npz_path.exists():
        return 0.0, 0.0

    try:
        data = np.load(mobility_npz_path)
        sigma_O = float(data.get("sigma_O", 0.0))
        sigma_D = float(data.get("sigma_D", 0.0))
        return sigma_O, sigma_D
    except Exception as e:
        logger.warning(
            f"Failed to load mobility noise params from {mobility_npz_path}: {e}"
        )
        return 0.0, 0.0


def write_mobility_time_varying_to_zarr(
    output_path: str,
    ordered_results: list[dict],
    run_start_index: int,
    region_ids: list[str],
    dates_ref: pd.DatetimeIndex,
    base_mobility: np.ndarray,
    mobility_kappa0_arr: np.ndarray,
    chunk_size: int = 256,
):
    """
    Stream-write mobility_time_varying to zarr to avoid OOM.

    Writes dense mobility as (run_id, origin, target, date), but never materializes
    the full 4D tensor in memory.
    """
    import zarr

    n_runs_new = len(ordered_results)
    n_regions = len(region_ids)
    n_dates = len(dates_ref)
    origin_chunk = min(chunk_size, n_regions)
    target_chunk = min(chunk_size, n_regions)
    date_chunk = min(chunk_size, n_dates)

    zgroup = zarr.open_group(output_path, mode="a")
    target_shape = (run_start_index + n_runs_new, n_regions, n_regions, n_dates)
    chunks = (1, origin_chunk, target_chunk, date_chunk)

    if "mobility_time_varying" in zgroup:
        mobility_arr = zgroup["mobility_time_varying"]
        expected_suffix = (n_regions, n_regions, n_dates)
        if tuple(mobility_arr.shape[1:]) != expected_suffix:
            raise ValueError(
                "Existing mobility_time_varying has incompatible shape: "
                f"{mobility_arr.shape}, expected (*, {expected_suffix[0]}, "
                f"{expected_suffix[1]}, {expected_suffix[2]})"
            )
        if mobility_arr.shape[0] < target_shape[0]:
            mobility_arr.resize(target_shape)
    else:
        mobility_arr = zgroup.create_dataset(
            "mobility_time_varying",
            shape=target_shape,
            chunks=chunks,
            dtype=np.float16,
        )

    logger.info(
        "Streaming mobility_time_varying write to %s (new_runs=%s, shape=%s, chunks=%s)",
        output_path,
        n_runs_new,
        target_shape,
        chunks,
    )

    for local_run_idx, result in enumerate(ordered_results):
        global_run_idx = run_start_index + local_run_idx
        run_id = result["run_id"]
        run_mob_type = result["mobility_type"]

        if run_mob_type == "time_varying":
            run_dir = Path(result["run_dir_path"])
            mobility_npz_path = run_dir / "mobility" / "mobility_series.npz"
            if not mobility_npz_path.exists():
                raise ValueError(
                    f"Run '{run_id}' is marked as time_varying but "
                    f"{mobility_npz_path} is missing."
                )

            with np.load(mobility_npz_path) as npz_data:
                R_series = npz_data["R_series"]  # (T, E)
                edgelist = npz_data["edgelist"]  # (E, 2)

            src = edgelist[:, 0].astype(np.int64)
            tgt = edgelist[:, 1].astype(np.int64)
            valid_edge_mask = (src < n_regions) & (tgt < n_regions)
            src_valid = src[valid_edge_mask]
            tgt_valid = tgt[valid_edge_mask]
            t_available = min(R_series.shape[0], n_dates)

            frame = np.zeros((n_regions, n_regions), dtype=np.float16)
            for t in range(n_dates):
                frame.fill(0.0)
                if t < t_available:
                    frame[src_valid, tgt_valid] = R_series[t, valid_edge_mask].astype(
                        np.float16
                    )
                mobility_arr[global_run_idx, :, :, t] = frame
        else:
            frame = np.empty((n_regions, n_regions), dtype=np.float16)
            for t in range(n_dates):
                np.multiply(
                    base_mobility,
                    np.float16(1.0 - mobility_kappa0_arr[local_run_idx, t]),
                    out=frame,
                    casting="unsafe",
                )
                mobility_arr[global_run_idx, :, :, t] = frame

        logger.info(
            "Wrote mobility_time_varying for run %s (%s)",
            run_id,
            run_mob_type,
        )


def build_date_range(config: dict, time_len: int):
    start_date = config.get("simulation", {}).get("start_date")
    if not start_date:
        raise ValueError("Missing simulation.start_date in config")
    return pd.date_range(start=start_date, periods=time_len)


def load_edar_muni_mapping(metapop_df, edar_nc_path=None):
    """Load EDAR-municipality edges and build EMAP (EDAR-municipality aggregation matrix).

    Returns:
        dict: {
            'edar_ids': list of EDAR IDs,
            'region_ids': list of region IDs in our metapopulation,
            'emap': (n_edar, n_region) contribution matrix where emap[edar_i, region_j]
                   is the fraction of region_j's wastewater that flows to edar_i
        }
        If no EDAR mapping found, returns None
    """
    if edar_nc_path is None:
        edar_nc_path = EDAR_MUNI_EDGES_PATH

    if not os.path.exists(edar_nc_path):
        raise FileNotFoundError(
            f"EDAR-municipality edges file not found: {edar_nc_path}\n"
            f"Biomarker generation requires EDAR catchment area mapping.\n"
            f"Expected location relative to project root: {EDAR_MUNI_EDGES_PATH}"
        )

    ds = xr.open_dataset(edar_nc_path)
    edar_ids = ds["edar_id"].values
    home_ids = ds["home"].values
    contribution_matrix = ds["contribution_ratio"].values
    ds.close()

    # Ensure metapop IDs are strings (preserve leading zeros like "08001")
    metapop_df["id"] = metapop_df["id"].astype(str)
    metapop_ids = metapop_df["id"].tolist()

    # Build EMAP: (n_edar, n_region) matrix
    # emap[edar_idx, region_idx] = contribution ratio
    region_id_to_idx = {region_id: i for i, region_id in enumerate(metapop_ids)}

    n_edar = len(edar_ids)
    n_region = len(metapop_ids)
    emap = np.zeros((n_edar, n_region))

    for edar_idx in range(n_edar):
        for home_idx, home_id in enumerate(home_ids):
            contrib = contribution_matrix[edar_idx, home_idx]
            if not np.isnan(contrib) and contrib > 0 and home_id in region_id_to_idx:
                region_idx = region_id_to_idx[home_id]
                emap[edar_idx, region_idx] = contrib

    # Normalize each EDAR row to sum to 1 (each EDAR receives contributions from multiple regions)
    row_sums = emap.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    emap = emap / row_sums

    logger.info(
        f"Loaded EMAP: {n_edar} EDARs, {n_region} regions, "
        f"{np.count_nonzero(emap):.0f} non-zero entries"
    )

    return {"edar_ids": list(edar_ids), "region_ids": metapop_ids, "emap": emap}


def aggregate_infections_to_edar(infections_stratified, emap_matrix, population_vector):
    """Aggregate stratified infections from regions to EDAR catchment areas using EMAP.

    Args:
        infections_stratified: (Time, Region, AgeGroup) array of infections
        emap_matrix: (EDAR, Region) contribution matrix from load_edar_muni_mapping
        population_vector: (Region,) population for normalization

    Returns:
        infections_edar: (Time, EDAR, AgeGroup) aggregated infections
        population_edar: (EDAR,) aggregated population per EDAR
    """
    # infections_stratified: (T, M, G)
    # emap_matrix: (E, M)
    # Result: (T, E, G) = einsum('em,tmg->teg', emap, infections)

    # Aggregate infections: for each age group, aggregate regions to EDARs
    infections_edar = np.einsum("em,tmg->teg", emap_matrix, infections_stratified)

    # Aggregate population: weighted by EMAP (fractional population flowing to each EDAR)
    population_edar = emap_matrix @ population_vector

    return infections_edar, population_edar


def assign_sparsity_tiers(
    n_runs: int, tiers: list[float], seed: Optional[int] = None
) -> np.ndarray:
    """
    Assign sparsity levels to runs using stratified sampling.

    Ensures each tier has roughly equal number of runs.
    Deterministic: same seed + n_runs produces same assignment.

    Args:
        n_runs: Number of runs to assign sparsity levels to
        tiers: List of sparsity levels to assign (e.g., [0.05, 0.20, 0.40, 0.60, 0.80])
        seed: Random seed for deterministic assignment (optional)

    Returns:
        sparsity_per_run: Array of shape (n_runs,) with sparsity level for each run
    """
    rng = np.random.default_rng(seed)
    sparsity_per_run = np.zeros(n_runs, dtype=float)

    # Stratified sampling: assign runs evenly across tiers
    runs_per_tier = n_runs // len(tiers)
    for i, tier_sparsity in enumerate(tiers):
        start_idx = i * runs_per_tier
        end_idx = start_idx + runs_per_tier if i < len(tiers) - 1 else n_runs
        sparsity_per_run[start_idx:end_idx] = tier_sparsity

    # Shuffle within tiers for randomness
    rng.shuffle(sparsity_per_run)
    return sparsity_per_run


def main():
    parser = argparse.ArgumentParser(
        description="Process synthetic simulation outputs into a raw observation zarr dataset."
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
        default="../runs/synthetic_test/raw_synthetic_observations.zarr",
        help="Output path for zarr store",
    )
    parser.add_argument(
        "--missing-rate",
        type=float,
        default=0.05,
        help="Fraction of case data to make missing (default: 0.05)",
    )
    parser.add_argument(
        "--missing-gap-length",
        type=int,
        default=3,
        help="Average length of missing data gaps in days (default: 3)",
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
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize zarr store if it doesn't exist (use with --append for 'create if not exists, else append' behavior)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Process only baseline scenarios (for spike detection in two-phase pipeline)",
    )
    parser.add_argument(
        "--compressor",
        default="zstd",
        choices=["zstd", "lz4", "blosc", "none"],
        help="Compressor for zarr (default: zstd)",
    )
    parser.add_argument(
        "--compressor-level",
        type=int,
        default=3,
        help="Compression level (default: 3)",
    )
    parser.add_argument(
        "--monitoring-threshold",
        type=float,
        default=0.0,
        help="Cumulative infections per EDAR to trigger wastewater monitoring (0=disabled)",
    )
    parser.add_argument(
        "--monitoring-delay-mean",
        type=int,
        default=0,
        help="Mean delay in days after threshold before monitoring starts",
    )
    parser.add_argument(
        "--monitoring-delay-std",
        type=int,
        default=0,
        help="Std dev of delay (stochastic variation per EDAR, prevents overfitting)",
    )

    # Reporting noise parameters for hospitalizations
    parser.add_argument(
        "--hosp-report-rate",
        type=float,
        default=DEFAULT_HOSP_REPORT_CONFIG["report_rate"],
        help="Hospitalization reporting rate (0.0 to 1.0, default: 0.85)",
    )
    parser.add_argument(
        "--hosp-delay-mean",
        type=int,
        default=DEFAULT_HOSP_REPORT_CONFIG["delay_mean"],
        help="Mean hospitalization reporting delay in days (default: 3)",
    )
    parser.add_argument(
        "--hosp-delay-std",
        type=int,
        default=DEFAULT_HOSP_REPORT_CONFIG["delay_std"],
        help="Std dev of hospitalization reporting delay (default: 1)",
    )

    # Reporting noise parameters for deaths
    parser.add_argument(
        "--deaths-report-rate",
        type=float,
        default=DEFAULT_DEATHS_REPORT_CONFIG["report_rate"],
        help="Deaths reporting rate (0.0 to 1.0, default: 0.90)",
    )
    parser.add_argument(
        "--deaths-delay-mean",
        type=int,
        default=DEFAULT_DEATHS_REPORT_CONFIG["delay_mean"],
        help="Mean deaths reporting delay in days (default: 7)",
    )
    parser.add_argument(
        "--deaths-delay-std",
        type=int,
        default=DEFAULT_DEATHS_REPORT_CONFIG["delay_std"],
        help="Std dev of deaths reporting delay (default: 2)",
    )

    # Sparsity tier configuration for curriculum learning
    parser.add_argument(
        "--sparsity-mode",
        choices=["uniform", "tiers"],
        default="tiers",
        help="Sparsity distribution mode: uniform (same for all runs) or tiers (vary per run for curriculum learning)",
    )
    parser.add_argument(
        "--sparsity-tiers",
        type=float,
        nargs="+",
        default=[0.05, 0.20, 0.40, 0.60, 0.80],
        help="Sparsity levels for tier assignment when --sparsity-mode=tiers (default: 0.05 0.20 0.40 0.60 0.80)",
    )
    parser.add_argument(
        "--sparsity-seed",
        type=int,
        default=None,
        help="Seed for deterministic tier assignment (default: None)",
    )

    # Wastewater gene-specific noise parameters (can override defaults)
    parser.add_argument(
        "--ww-noise-n1",
        type=float,
        default=None,
        help="Wastewater noise sigma for N1 gene target (default: 0.5)",
    )
    parser.add_argument(
        "--ww-noise-n2",
        type=float,
        default=None,
        help="Wastewater noise sigma for N2 gene target (default: 0.8)",
    )
    parser.add_argument(
        "--ww-noise-ip4",
        type=float,
        default=None,
        help="Wastewater noise sigma for IP4 gene target (default: 0.6)",
    )
    parser.add_argument(
        "--ww-transport-loss",
        type=float,
        default=None,
        help="Wastewater transport loss (signal decay in sewer, default: 0.0)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for processing runs (default: 1, use -1 for all CPUs)",
    )
    parser.add_argument(
        "--include-latents",
        action="store_true",
        help="Export latent simulator compartment targets from compartments_full.nc",
    )

    args = parser.parse_args()

    # Handle n_jobs parameter
    n_jobs = args.n_jobs
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if n_jobs > 1:
        logger.info(f"Using {n_jobs} parallel workers for processing.")

    # Define Gene Targets with biological properties
    # Allow command-line override of noise parameters
    GENE_TARGETS = {
        "N1": {
            "sensitivity_scale": 500000.0,
            "noise_sigma": args.ww_noise_n1 if args.ww_noise_n1 is not None else 0.5,
            "limit_of_detection": 375.0,
            "limit_of_detection_log1p": float(np.log1p(375.0)),  # ≈ 5.93
            "transport_loss": args.ww_transport_loss
            if args.ww_transport_loss is not None
            else 50.0,
            "lod_probabilistic": True,
            "lod_slope": 1.5,
        },
        "N2": {
            "sensitivity_scale": 400000.0,
            "noise_sigma": args.ww_noise_n2 if args.ww_noise_n2 is not None else 0.8,
            "limit_of_detection": 500.0,
            "limit_of_detection_log1p": float(np.log1p(500.0)),  # ≈ 6.22
            "transport_loss": args.ww_transport_loss
            if args.ww_transport_loss is not None
            else 100.0,
            "lod_probabilistic": True,
            "lod_slope": 1.5,
        },
        "IP4": {
            "sensitivity_scale": 250000.0,
            "noise_sigma": args.ww_noise_ip4 if args.ww_noise_ip4 is not None else 0.6,
            "limit_of_detection": 800.0,
            "limit_of_detection_log1p": float(np.log1p(800.0)),  # ≈ 6.69
            "transport_loss": args.ww_transport_loss
            if args.ww_transport_loss is not None
            else 200.0,
            "lod_probabilistic": True,
            "lod_slope": 1.5,
        },
    }
    TARGET_NAMES = list(GENE_TARGETS.keys())

    runs_dir = Path(args.runs_dir)
    run_dirs = sorted([d for d in runs_dir.glob("run_*") if d.is_dir()])

    if not run_dirs:
        raise ValueError(f"No run_* folders found in {runs_dir}")
    
        # Filter to baseline-only scenarios if requested
    if args.baseline_only:
        run_dirs = [
            d
            for d in run_dirs
            if d.name.endswith("_Baseline") or "_Baseline_" in d.name
        ]
        if not run_dirs:
            raise ValueError(f"No baseline runs found in {runs_dir}")

    # Determine sparsity level for each run
    n_runs_expected = len(run_dirs)
    if args.sparsity_mode == "uniform":
        sparsity_per_run = np.full(n_runs_expected, args.missing_rate)
    elif args.sparsity_mode == "tiers":
        sparsity_per_run = assign_sparsity_tiers(
            n_runs_expected, args.sparsity_tiers, seed=args.sparsity_seed
        )

    metapop_df = pd.read_csv(args.metapop_csv, dtype={"id": str})
    metapop_df["id"] = metapop_df["id"].astype(str)
    region_ids = metapop_df["id"].tolist()
    population_vector = metapop_df["total"].values.astype(float)

    # Load base mobility matrix
    base_mobility = load_mobility_matrix(args.metapop_csv)

    # Load EDAR-municipality mapping (EMAP)
    edar_path = args.edar_edges
    if edar_path is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        edar_path = os.path.join(project_root, EDAR_MUNI_EDGES_PATH)

    emap_data = load_edar_muni_mapping(metapop_df, edar_nc_path=edar_path)
    edar_ids = emap_data["edar_ids"]
    emap = emap_data["emap"]

    # Pre-filter EDARs to only those with mapped population
    population_edar = emap @ population_vector
    valid_edar_mask = population_edar > 0
    if np.sum(valid_edar_mask) < len(population_edar):
        emap = emap[valid_edar_mask, :]
        edar_ids = [edar_ids[i] for i in range(len(edar_ids)) if valid_edar_mask[i]]
        emap_data["edar_ids"] = edar_ids
        emap_data["emap"] = emap

    # Pre-scan first run to get dimensions and dates
    first_run_dir = run_dirs[0]
    first_artifacts = load_run_artifacts(first_run_dir)
    if first_artifacts is None:
        raise ValueError(f"Could not load artifacts from first run: {first_run_dir}")
    first_config, first_obs_path, _ = first_artifacts
    with xr.open_dataset(first_obs_path) as ds_meta:
        n_dates = ds_meta.sizes["T"]
        n_age_groups = ds_meta.sizes["G"] if "G" in ds_meta.sizes else 1
    dates_ref = build_date_range(first_config, n_dates)

    # Check for time-varying mobility
    output_path = args.output
    target_has_mobility_tv = False
    if args.append and os.path.exists(output_path):
        try:
            with xr.open_zarr(output_path, chunks=None) as ds_existing:
                target_has_mobility_tv = "mobility_time_varying" in ds_existing.data_vars
        except Exception:
            pass

    has_time_varying = target_has_mobility_tv
    mobility_types_pre = []
    for rd in run_dirs:
        is_tv = (rd / "mobility" / "mobility_series.npz").exists()
        mobility_types_pre.append("time_varying" if is_tv else "static")
        if is_tv:
            has_time_varying = True

    # Initialize Zarr skeleton if not appending
    ww_spatial_dim = "edar_id" if emap_data else "region_id"
    if not args.append:
        if os.path.exists(output_path):
            import shutil
            if os.path.isdir(output_path):
                shutil.rmtree(output_path)
            else:
                os.remove(output_path)
        
        # Build coordinate space for skeleton
        run_ids = [sanitize_run_id(d.name) for d in run_dirs]
        coords = {
            "run_id": run_ids,
            "region_id": region_ids,
            "date": dates_ref,
            "origin": region_ids,
            "target": region_ids,
            "age_group": [str(i) for i in range(n_age_groups)]
        }
        if emap_data is not None:
            coords["edar_id"] = edar_ids
    
            # Create lazy skeleton dataset
        import dask.array as da
        data_vars = {}
        
        def create_lazy(dims, dtype=np.float16):
            shape = [len(coords[d]) for d in dims]
            return (dims, da.zeros(shape, chunks=(1, *[len(coords[d]) for d in dims[1:]]), dtype=dtype))

        data_vars["infections_true"] = create_lazy(("run_id", "region_id", "date"))
        data_vars["hospitalizations_true"] = create_lazy(("run_id", "region_id", "date"))
        data_vars["deaths_true"] = create_lazy(("run_id", "region_id", "date"))
        if args.include_latents:
            for latent_var in LATENT_ZARR_VARS:
                data_vars[latent_var] = create_lazy(("run_id", "region_id", "date"))
        data_vars["cases"] = create_lazy(("run_id", "date", "region_id"))
        data_vars["cases_mask"] = create_lazy(("run_id", "date", "region_id"))
        data_vars["cases_age"] = create_lazy(("run_id", "date", "region_id", "age_group"))
        data_vars["hospitalizations"] = create_lazy(("run_id", "date", "region_id"))
        data_vars["deaths"] = create_lazy(("run_id", "date", "region_id"))
        
        data_vars["mobility_base"] = (("origin", "target"), base_mobility.astype(np.float16))
        data_vars["mobility_kappa0"] = create_lazy(("run_id", "date"))
        if has_time_varying:
            data_vars["mobility_time_varying"] = create_lazy(("run_id", "origin", "target", "date"))
        
        data_vars["synthetic_mobility_type"] = (("run_id",), da.from_array(np.array([""] * len(run_ids), dtype="U20"), chunks=1))
        
        population_tiled = np.tile(population_vector.astype(np.int32), (len(run_ids), 1))
        data_vars["population"] = (("run_id", "region_id"), population_tiled)

        for tname in TARGET_NAMES:
            data_vars[f"edar_biomarker_{tname}"] = create_lazy(("run_id", "date", ww_spatial_dim))
            data_vars[f"edar_biomarker_{tname}_censor_hints"] = create_lazy(("run_id", "date", ww_spatial_dim), dtype=np.int8)
            lod_val = GENE_TARGETS[tname]["limit_of_detection_log1p"]
            data_vars[f"edar_biomarker_{tname}_LoD"] = (("run_id", ww_spatial_dim), np.full((len(run_ids), len(coords[ww_spatial_dim])), lod_val, dtype=np.float16))

        # Add metadata variables
        for meta_var in ["synthetic_strength", "synthetic_sparsity_level", "synthetic_mobility_noise_sigma_O", 
                        "synthetic_mobility_noise_sigma_D", "synthetic_mobility_noise_factor",
                        "synthetic_cases_report_rate_min", "synthetic_cases_report_rate_max", "synthetic_cases_report_delay_mean",
                        "synthetic_hosp_report_rate", "synthetic_hosp_report_delay_mean", "synthetic_hosp_report_delay_std",
                        "synthetic_deaths_report_rate", "synthetic_deaths_report_delay_mean", "synthetic_deaths_report_delay_std",
                        "synthetic_ww_noise_sigma_N1", "synthetic_ww_noise_sigma_N2", "synthetic_ww_noise_sigma_IP4", "synthetic_ww_transport_loss"]:
            data_vars[meta_var] = create_lazy(("run_id",), dtype=float)
        
        data_vars["synthetic_scenario_type"] = (("run_id",), da.from_array(np.array([""] * len(run_ids), dtype="U20"), chunks=1))

        ds_skeleton = xr.Dataset(data_vars, coords=coords)
        
        # Apply encodings
        chunk_size = args.chunk_size
        region_chunk = min(chunk_size, len(region_ids))
        date_chunk = min(chunk_size, n_dates)
        ww_chunk = min(chunk_size, len(coords[ww_spatial_dim]))
        
        truth_vars = ["infections_true", "hospitalizations_true", "deaths_true"]
        if args.include_latents:
            truth_vars.extend(LATENT_ZARR_VARS)
        encoding = {v: {"chunksizes": (1, region_chunk, date_chunk)} for v in truth_vars}
        encoding.update({v: {"chunksizes": (1, date_chunk, region_chunk)} for v in ["cases", "cases_mask", "hospitalizations", "deaths"]})
        encoding["cases_age"] = {"chunksizes": (1, date_chunk, region_chunk, n_age_groups)}
        encoding["mobility_kappa0"] = {"chunksizes": (1, date_chunk)}
        if has_time_varying:
            encoding["mobility_time_varying"] = {"chunksizes": (1, region_chunk, region_chunk, date_chunk)}
        for tname in TARGET_NAMES:
            encoding[f"edar_biomarker_{tname}"] = {"chunksizes": (1, date_chunk, ww_chunk)}
            encoding[f"edar_biomarker_{tname}_censor_hints"] = {"chunksizes": (1, date_chunk, ww_chunk)}

        for v in encoding:
            if v in ds_skeleton.data_vars:
                ds_skeleton[v].encoding = encoding[v]

        logger.info(f"Initializing Zarr skeleton at {output_path}")
        ds_skeleton.to_zarr(output_path, compute=False, zarr_format=2)

    # Prepare work items for parallel processing
    rng = np.random.default_rng(args.seed)
    reported_cfg = {"min_rate": args.min_rate, "max_rate": args.max_rate, "inflection_day": args.inflection_day, "slope": args.slope}
    wastewater_cfg = {"gamma_shape": args.gamma_shape, "gamma_scale": args.gamma_scale, "noise_sigma": args.noise_sigma, "kernel_quantile": args.kernel_quantile}
    args_dict = {"missing_gap_length": args.missing_gap_length, "monitoring_threshold": args.monitoring_threshold, 
                 "monitoring_delay_mean": args.monitoring_delay_mean, "monitoring_delay_std": args.monitoring_delay_std,
                 "hosp_report_rate": args.hosp_report_rate, "hosp_delay_mean": args.hosp_delay_mean, "hosp_delay_std": args.hosp_delay_std,
                 "deaths_report_rate": args.deaths_report_rate, "deaths_delay_mean": args.deaths_delay_mean, "deaths_delay_std": args.deaths_delay_std,
                 "include_latents": args.include_latents}

    if args.append and args.include_latents and os.path.exists(output_path):
        with xr.open_zarr(output_path, chunks=None) as ds_existing:
            missing_latents = [
                var for var in LATENT_ZARR_VARS if var not in ds_existing.data_vars
            ]
        if missing_latents:
            raise ValueError(
                "Cannot append latent targets to an existing zarr without latent schema. "
                f"Missing variables: {missing_latents}"
            )

    work_items = []
    for run_idx, run_dir in enumerate(run_dirs):
        if load_run_artifacts(run_dir) is None: continue
        seed = rng.integers(0, 2**32 - 1) if args.seed else None
        work_items.append((str(run_dir), sparsity_per_run[run_idx], seed))

    if not work_items:
        logger.warning(f"No valid run artifacts (config/observables) found in {args.runs_dir}")
        return

    run_id_to_idx = {sanitize_run_id(d.name): i for i, d in enumerate(run_dirs)}
    
    # Store kappa0 series for final mobility pass if time-varying is enabled
    mobility_kappa0_map = {}

    def stream_result_to_zarr(result):
        run_id = result["run_id"]
        run_idx = run_id_to_idx[run_id]
        with np.load(result["npz_path"]) as data:
            dv = {}
            dv["infections_true"] = (("run_id", "region_id", "date"), data["infections_true"][None, :, :].astype(np.float16))
            dv["hospitalizations_true"] = (("run_id", "region_id", "date"), data["hospitalizations_true"][None, :, :].astype(np.float16))
            dv["deaths_true"] = (("run_id", "region_id", "date"), data["deaths_true"][None, :, :].astype(np.float16))
            if args.include_latents:
                for latent_var in LATENT_ZARR_VARS:
                    dv[latent_var] = (
                        ("run_id", "region_id", "date"),
                        data[latent_var][None, :, :].astype(np.float16),
                    )
            dv["cases"] = (("run_id", "date", "region_id"), data["cases_raw"][None, :, :].transpose(0, 2, 1).astype(np.float16))
            dv["cases_mask"] = (("run_id", "date", "region_id"), data["cases_mask"][None, :, :].transpose(0, 2, 1).astype(np.float16))
            dv["cases_age"] = (("run_id", "date", "region_id", "age_group"), data["cases_age"][None, :].astype(np.float16))
            dv["hospitalizations"] = (("run_id", "date", "region_id"), data["hospitalizations_raw"][None, :, :].transpose(0, 2, 1).astype(np.float16))
            dv["deaths"] = (("run_id", "date", "region_id"), data["deaths_raw"][None, :, :].transpose(0, 2, 1).astype(np.float16))
            dv["mobility_kappa0"] = (("run_id", "date"), data["mobility_kappa0"][None, :].astype(np.float16))
            
            # Save for second pass if mobility_time_varying is needed
            if has_time_varying:
                mobility_kappa0_map[run_id] = data["mobility_kappa0"]

            dv["population"] = (("run_id", "region_id"), population_vector[None, :].astype(np.int32))
            dv["synthetic_mobility_type"] = (("run_id",), np.array([result["mobility_type"]], dtype="U20"))
            dv["synthetic_scenario_type"] = (("run_id",), np.array([result["scenario_type"]], dtype="U20"))
            
            for tname_idx, tname in enumerate(TARGET_NAMES):
                dv[f"edar_biomarker_{tname}"] = (("run_id", "date", ww_spatial_dim), np.log1p(data["wastewater_raw"][None, :, :, tname_idx]).astype(np.float16))
                dv[f"edar_biomarker_{tname}_censor_hints"] = (("run_id", "date", ww_spatial_dim), data["wastewater_censor"][None, :, :, tname_idx])
            
            dv["synthetic_strength"] = (("run_id",), np.array([result["strength"]], dtype=float))
            dv["synthetic_sparsity_level"] = (("run_id",), np.array([result["sparsity"]], dtype=float))
            dv["synthetic_mobility_noise_sigma_O"] = (("run_id",), np.array([result["mobility_sigma_O"]], dtype=float))
            dv["synthetic_mobility_noise_sigma_D"] = (("run_id",), np.array([result["mobility_sigma_D"]], dtype=float))
            dv["synthetic_mobility_noise_factor"] = (("run_id",), np.array([result["mobility_noise"]], dtype=float))
            
            dv["synthetic_cases_report_rate_min"] = (("run_id",), np.array([args.min_rate], dtype=float))
            dv["synthetic_cases_report_rate_max"] = (("run_id",), np.array([args.max_rate], dtype=float))
            dv["synthetic_cases_report_delay_mean"] = (("run_id",), np.array([0.0], dtype=float))
            dv["synthetic_hosp_report_rate"] = (("run_id",), np.array([args.hosp_report_rate], dtype=float))
            dv["synthetic_hosp_report_delay_mean"] = (("run_id",), np.array([args.hosp_delay_mean], dtype=float))
            dv["synthetic_hosp_report_delay_std"] = (("run_id",), np.array([args.hosp_delay_std], dtype=float))
            dv["synthetic_deaths_report_rate"] = (("run_id",), np.array([args.deaths_report_rate], dtype=float))
            dv["synthetic_deaths_report_delay_mean"] = (("run_id",), np.array([args.deaths_delay_mean], dtype=float))
            dv["synthetic_deaths_report_delay_std"] = (("run_id",), np.array([args.deaths_delay_std], dtype=float))
            dv["synthetic_ww_noise_sigma_N1"] = (("run_id",), np.array([GENE_TARGETS["N1"]["noise_sigma"]], dtype=float))
            dv["synthetic_ww_noise_sigma_N2"] = (("run_id",), np.array([GENE_TARGETS["N2"]["noise_sigma"]], dtype=float))
            dv["synthetic_ww_noise_sigma_IP4"] = (("run_id",), np.array([GENE_TARGETS["IP4"]["noise_sigma"]], dtype=float))
            dv["synthetic_ww_transport_loss"] = (("run_id",), np.array([GENE_TARGETS["N1"]["transport_loss"]], dtype=float))

            ds_run = xr.Dataset(dv, coords={"run_id": [run_id], "date": dates_ref, "region_id": region_ids, "origin": region_ids, "target": region_ids, "age_group": [str(i) for i in range(n_age_groups)], ww_spatial_dim: edar_ids if emap_data else region_ids})
            
            if args.append:
                ds_run.to_zarr(output_path, mode="a", append_dim="run_id")
            else:
                # When writing to a region, we must drop coords that don't have the 'run_id' dimension
                # otherwise xarray will try to write them and fail because they don't fit the region.
                vars_to_drop = [c for c in ds_run.coords if "run_id" not in ds_run.coords[c].dims]
                ds_run.drop_vars(vars_to_drop).to_zarr(output_path, region={"run_id": slice(run_idx, run_idx+1)})
            
        if os.path.exists(result["npz_path"]): os.remove(result["npz_path"])

    # Execute processing
    processed_results = []
    if n_jobs > 1:
        logger.info(f"Processing {len(work_items)} runs with {n_jobs} workers...")
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=init_worker, 
                                 initargs=(region_ids, population_vector, emap_data, GENE_TARGETS, 
                                           reported_cfg, wastewater_cfg, args_dict)) as executor:
            futures = [executor.submit(process_single_run, *item) for item in work_items]
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res:
                        stream_result_to_zarr(res)
                        processed_results.append(res)
                        logger.info(f"Processed and streamed: {res['run_id']}")
                except Exception as e:
                    logger.error(f"Failed run: {e}")
    else:
        init_worker(region_ids, population_vector, emap_data, GENE_TARGETS, reported_cfg, wastewater_cfg, args_dict)
        logger.info(f"Processing {len(work_items)} runs sequentially...")
        for item in work_items:
            res = process_single_run(*item)
            if res:
                stream_result_to_zarr(res)
                processed_results.append(res)
                logger.info(f"Processed and streamed: {res['run_id']}")

    if not processed_results:
        logger.warning("No runs were successfully processed.")
        return

    if has_time_varying:
        # Final pass for large mobility arrays using efficient frame-by-frame streaming
        processed_results.sort(key=lambda x: x["run_id"])
        mobility_kappa0_arr = np.stack([mobility_kappa0_map[r["run_id"]] for r in processed_results], axis=0)
        
        # Determine start index for mobility write
        run_start_idx = 0
        if args.append:
            try:
                with xr.open_zarr(output_path) as ds_existing:
                    run_start_idx = int(ds_existing.sizes["run_id"]) - len(processed_results)
            except Exception:
                pass

        write_mobility_time_varying_to_zarr(
            output_path=output_path,
            ordered_results=processed_results,
            run_start_index=run_start_idx,
            region_ids=region_ids,
            dates_ref=dates_ref,
            base_mobility=base_mobility,
            mobility_kappa0_arr=mobility_kappa0_arr,
            chunk_size=args.chunk_size,
        )

    logger.info(f"Done! Output written to {output_path}")


if __name__ == "__main__":
    main()
    
