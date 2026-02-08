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
    edar_biomarker_N1, edar_biomarker_N2, edar_biomarker_IP4: (run_id, date, edar_id) - Raw wastewater
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
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit

from synthetic_observations import (
    DEFAULT_REPORTED_CASES_CONFIG,
    DEFAULT_WASTEWATER_CONFIG,
    DEFAULT_HOSP_REPORT_CONFIG,
    DEFAULT_DEATHS_REPORT_CONFIG,
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


def process_single_run(
    run_dir_path: str,
    region_ids: list,
    population_vector: np.ndarray,
    emap_data: Optional[dict],
    sparsity: float,
    reported_cfg: dict,
    wastewater_cfg: dict,
    gene_targets: dict,
    args_dict: dict,
    seed: Optional[int] = None,
) -> Optional[dict]:
    """
    Process a single run directory and return processed data.

    This function is designed to be used with multiprocessing.Pool.
    All inputs must be picklable (no logger, no complex objects).

    Args:
        run_dir_path: Path to run directory as string
        region_ids: List of region IDs
        population_vector: Population array
        emap_data: EMAP dict with 'edar_ids', 'emap', or None
        sparsity: Sparsity level for this run
        reported_cfg: Cases reporting config dict
        wastewater_cfg: Wastewater config dict
        gene_targets: Gene targets dict
        args_dict: Arguments dict with hosp_report_rate, etc.
        seed: Random seed for reproducibility

    Returns:
        Dict with processed data or None if failed
    """
    run_dir = Path(run_dir_path)

    # Load artifacts
    artifacts = load_run_artifacts(run_dir)
    if artifacts is None:
        return None

    config, observables_path = artifacts

    # Load data
    infections_stratified = load_infections_stratified(observables_path)
    infections_total = infections_stratified.sum(axis=2)
    hospitalizations = load_hospitalizations(observables_path)
    deaths = load_deaths(observables_path)

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

    # Generate reported cases
    reported_cases, _ = generate_reported_cases(
        infections_total, config=reported_cfg, rng=case_rng
    )
    cases_with_gaps, _ = apply_missing_data_patterns(
        reported_cases.T,
        missing_rate=sparsity,
        missing_gap_length=args_dict.get("missing_gap_length", 3),
        rng=case_rng,
    )

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

    mobility_dense = detect_and_load_time_varying_mobility(
        run_dir, n_regions=len(region_ids), n_dates=len(dates)
    )
    sigma_O, sigma_D = load_mobility_noise_params(run_dir)

    # Determine mobility type
    if mobility_dense is not None:
        mobility_type = "time_varying"
    else:
        mobility_type = "static"
        mobility_dense = None

    # Parse metadata
    scenario_type, strength = parse_run_metadata(run_dir.name)
    if np.isnan(strength):
        strength = float(np.nanmax(kappa0_series)) if len(kappa0_series) > 0 else 0.0

    return {
        "run_id": sanitize_run_id(run_dir.name),
        "scenario_type": scenario_type,
        "strength": strength,
        "sparsity": sparsity,
        "dates": dates,
        "infections_true": infections_total.T.astype(int),
        "hospitalizations_true": hospitalizations.T.astype(int),
        "deaths_true": deaths.T.astype(int),
        "hospitalizations_raw": reported_hospitalizations.T.astype(int),
        "deaths_raw": reported_deaths.T.astype(int),
        "cases_raw": cases_with_gaps.astype(float),
        "wastewater_raw": ww_by_edar.astype(float),
        "wastewater_censor": censor_by_edar.astype(np.int8),
        "mobility_kappa0": kappa0_series.astype(float),
        "mobility_dense": mobility_dense,
        "mobility_type": mobility_type,
        "mobility_sigma_O": sigma_O,
        "mobility_sigma_D": sigma_D,
        "mobility_noise": max(sigma_O, sigma_D),
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
        T, E = R_series.shape
        mobility_dense = np.zeros((T, n_regions, n_regions), dtype=np.float64)

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
            if not np.isnan(contrib) and contrib > 0:
                if home_id in region_id_to_idx:
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

    args = parser.parse_args()

    # Handle n_jobs parameter
    n_jobs = args.n_jobs
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if n_jobs > 1:
        logger.warning(
            f"--n-jobs={n_jobs} specified but parallel processing not yet implemented. "
            f"Processing will be sequential. This parameter is reserved for future use."
        )

    # Define Gene Targets with biological properties
    # Allow command-line override of noise parameters
    GENE_TARGETS = {
        "N1": {
            "sensitivity_scale": 500000.0,
            "noise_sigma": args.ww_noise_n1 if args.ww_noise_n1 is not None else 0.5,
            "limit_of_detection": 375.0,
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
        original_count = len(run_dirs)
        run_dirs = [
            d
            for d in run_dirs
            if d.name.endswith("_Baseline") or "_Baseline_" in d.name
        ]
        filtered_count = original_count - len(run_dirs)
        logger.info(
            f"Baseline-only mode: filtered out {filtered_count} non-baseline runs"
        )
        if not run_dirs:
            raise ValueError(f"No baseline runs found in {runs_dir}")

    # Determine sparsity level for each run
    n_runs_expected = len(run_dirs)
    if args.sparsity_mode == "uniform":
        sparsity_per_run = np.full(n_runs_expected, args.missing_rate)
        logger.info(
            f"Sparsity mode: uniform (all runs use missing_rate={args.missing_rate})"
        )
    elif args.sparsity_mode == "tiers":
        sparsity_per_run = assign_sparsity_tiers(
            n_runs_expected, args.sparsity_tiers, seed=args.sparsity_seed
        )
        unique, counts = np.unique(sparsity_per_run, return_counts=True)
        tier_summary = ", ".join([f"{u:.2f}:{c}" for u, c in zip(unique, counts)])
        logger.info(
            f"Sparsity mode: tiers with {len(args.sparsity_tiers)} levels - "
            f"distribution: {tier_summary}"
        )

    metapop_df = pd.read_csv(args.metapop_csv, dtype={"id": str})
    metapop_df["id"] = metapop_df["id"].astype(str)
    region_ids = metapop_df["id"].tolist()
    population_vector = metapop_df["total"].values.astype(float)

    # Load base mobility matrix (for outputting full OD matrices)
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
    # Compute EDAR population by aggregating region populations via EMAP
    population_vector = metapop_df["total"].values
    population_edar = emap @ population_vector
    valid_edar_mask = population_edar > 0
    n_valid_edars = np.sum(valid_edar_mask)

    if n_valid_edars < len(population_edar):
        n_filtered = len(population_edar) - n_valid_edars
        logger.info(
            f"Filtering {n_filtered} EDARs with zero population, "
            f"keeping {n_valid_edars} valid EDARs"
        )
        # Filter emap and edar_ids
        emap = emap[valid_edar_mask, :]
        edar_ids = [edar_ids[i] for i in range(len(edar_ids)) if valid_edar_mask[i]]
        # Update emap_data with filtered values
        emap_data["edar_ids"] = edar_ids
        emap_data["emap"] = emap

    logger.info(f"Using EMAP-based wastewater generation: {len(edar_ids)} EDARs")

    # Storage for all runs - stream write pattern to avoid OOM
    # NOTE: mobility is stored in factorized form (mobility_base + mobility_kappa0)
    # instead of full tensor to reduce memory from ~300GB to ~500MB for 100 runs
    # When time-varying mobility exists (mobility_sigma_O/D > 0), we store dense OD matrices
    infections_true_runs = []
    hospitalizations_true_runs = []
    deaths_true_runs = []
    hospitalizations_raw_runs = []  # Reported hospitalizations (with noise)
    deaths_raw_runs = []  # Reported deaths (with noise)
    cases_raw_runs = []
    wastewater_raw_runs = []
    wastewater_censor_runs = []
    mobility_kappa0_runs = []  # Store kappa0 series per run instead of full mobility tensor
    mobility_runs = []  # Store dense mobility if time-varying, None if static
    mobility_types = []  # Track "static" vs "time_varying" per run
    mobility_sigma_O_runs = []  # Track per-run mobility noise sigma (origin)
    mobility_sigma_D_runs = []  # Track per-run mobility noise sigma (destination)
    mobility_noise_runs = []  # Track per-run combined mobility noise factor
    scenario_types = []
    strengths = []
    run_ids = []
    sparsity_levels = []  # Track per-run sparsity levels for metadata

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

    # Prepare arguments for multiprocessing
    args_dict = {
        "missing_gap_length": args.missing_gap_length,
        "monitoring_threshold": args.monitoring_threshold,
        "monitoring_delay_mean": args.monitoring_delay_mean,
        "monitoring_delay_std": args.monitoring_delay_std,
        "hosp_report_rate": args.hosp_report_rate,
        "hosp_delay_mean": args.hosp_delay_mean,
        "hosp_delay_std": args.hosp_delay_std,
        "deaths_report_rate": args.deaths_report_rate,
        "deaths_delay_mean": args.deaths_delay_mean,
        "deaths_delay_std": args.deaths_delay_std,
    }

    # Track run index for per-run sparsity assignment
    run_idx = 0

    # Process runs in parallel using multiprocessing
    if n_jobs > 1:
        logger.info(
            f"Processing {len(run_dirs)} runs with {n_jobs} parallel workers..."
        )

        # Prepare work items
        work_items = []
        for run_idx, run_dir in enumerate(run_dirs):
            # Pre-check if run has artifacts
            artifacts = load_run_artifacts(run_dir)
            if artifacts is None:
                continue

            run_sparsity = sparsity_per_run[run_idx]
            seed = rng.integers(0, 2**32 - 1) if args.seed else None

            work_items.append(
                (
                    str(run_dir),
                    region_ids,
                    population_vector,
                    emap_data,
                    run_sparsity,
                    reported_cfg,
                    wastewater_cfg,
                    GENE_TARGETS,
                    args_dict,
                    seed,
                )
            )

        # Process in parallel
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(process_single_run, *item) for item in work_items
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        logger.info(f"Completed processing: {result['run_id']}")
                except Exception as e:
                    logger.error(f"Failed to process run: {e}")

        # Sort results by run_id to maintain order
        results.sort(key=lambda x: x["run_id"])

        # Extract dates_ref from first result
        if results:
            dates_ref = results[0]["dates"]

        # Collect results into lists
        for result in results:
            infections_true_runs.append(result["infections_true"])
            hospitalizations_true_runs.append(result["hospitalizations_true"])
            deaths_true_runs.append(result["deaths_true"])
            hospitalizations_raw_runs.append(result["hospitalizations_raw"])
            deaths_raw_runs.append(result["deaths_raw"])
            cases_raw_runs.append(result["cases_raw"])
            wastewater_raw_runs.append(result["wastewater_raw"])
            wastewater_censor_runs.append(result["wastewater_censor"])
            mobility_kappa0_runs.append(result["mobility_kappa0"])
            mobility_runs.append(result["mobility_dense"])
            mobility_types.append(result["mobility_type"])
            mobility_sigma_O_runs.append(result["mobility_sigma_O"])
            mobility_sigma_D_runs.append(result["mobility_sigma_D"])
            mobility_noise_runs.append(result["mobility_noise"])
            scenario_types.append(result["scenario_type"])
            strengths.append(result["strength"])
            run_ids.append(result["run_id"])
            sparsity_levels.append(result["sparsity"])
    else:
        # Sequential processing (original code)
        logger.info(f"Processing {len(run_dirs)} runs sequentially...")
        for run_idx, run_dir in enumerate(run_dirs):
            artifacts = load_run_artifacts(run_dir)
            if artifacts is None:
                continue

            # Get sparsity level for this run
            run_sparsity = sparsity_per_run[run_idx]

            config, observables_path = artifacts
            infections_stratified = load_infections_stratified(observables_path)
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
            hosp_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            deaths_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            ww_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

            # Generate reported cases with missing data patterns
            reported_cases, _ = generate_reported_cases(
                infections_total, config=reported_cfg, rng=case_rng
            )
            cases_with_gaps, _ = apply_missing_data_patterns(
                reported_cases.T,
                missing_rate=run_sparsity,
                missing_gap_length=args.missing_gap_length,
                rng=case_rng,
            )

            # Generate wastewater with censoring on EDAR domain
            if emap_data is not None:
                infections_edar_strat, population_edar = aggregate_infections_to_edar(
                    infections_stratified, emap_data["emap"], population_vector
                )
                from synthetic_observations import _compute_monitoring_start_mask

                monitoring_mask = _compute_monitoring_start_mask(
                    infections_edar_strat,
                    threshold=args.monitoring_threshold,
                    delay_days=args.monitoring_delay_mean,
                    delay_std=args.monitoring_delay_std,
                    rng=ww_rng,
                )
                ww_by_edar, censor_by_edar = generate_wastewater_with_censoring(
                    infections_edar_strat,
                    population_edar,
                    GENE_TARGETS,
                    wastewater_cfg,
                    rng=ww_rng,
                    monitoring_mask=monitoring_mask,
                    missing_rate=run_sparsity,
                )
            else:
                from synthetic_observations import _compute_monitoring_start_mask

                monitoring_mask = _compute_monitoring_start_mask(
                    infections_stratified,
                    threshold=args.monitoring_threshold,
                    delay_days=args.monitoring_delay_mean,
                    delay_std=args.monitoring_delay_std,
                    rng=ww_rng,
                )
                ww_by_edar, censor_by_edar = generate_wastewater_with_censoring(
                    infections_stratified,
                    population_vector,
                    GENE_TARGETS,
                    wastewater_cfg,
                    rng=ww_rng,
                    monitoring_mask=monitoring_mask,
                    missing_rate=run_sparsity,
                )

            # Generate reported hospitalizations and deaths with reporting noise
            reported_hospitalizations = generate_reported_with_delay(
                hospitalizations,
                report_rate=args.hosp_report_rate,
                delay_mean=args.hosp_delay_mean,
                delay_std=args.hosp_delay_std,
                rng=hosp_rng,
            )
            reported_deaths = generate_reported_with_delay(
                deaths,
                report_rate=args.deaths_report_rate,
                delay_mean=args.deaths_delay_mean,
                delay_std=args.deaths_delay_std,
                rng=deaths_rng,
            )

            # Load kappa0 series
            kappa0_path = resolve_kappa0_path(config, run_dir)
            kappa0_series = load_kappa0_series(kappa0_path, dates)

            # Check for time-varying mobility
            mobility_dense = detect_and_load_time_varying_mobility(
                run_dir, n_regions=len(region_ids), n_dates=len(dates)
            )
            sigma_O, sigma_D = load_mobility_noise_params(run_dir)
            mobility_sigma_O_runs.append(sigma_O)
            mobility_sigma_D_runs.append(sigma_D)
            mobility_noise_runs.append(max(sigma_O, sigma_D))

            # Store mobility data
            if mobility_dense is not None:
                mobility_runs.append(mobility_dense)
                mobility_types.append("time_varying")
            else:
                mobility_runs.append(None)
                mobility_types.append("static")

            # Store run data
            infections_true_runs.append(infections_total.T.astype(int))
            hospitalizations_true_runs.append(hospitalizations.T.astype(int))
            deaths_true_runs.append(deaths.T.astype(int))
            hospitalizations_raw_runs.append(reported_hospitalizations.T.astype(int))
            deaths_raw_runs.append(reported_deaths.T.astype(int))
            cases_raw_runs.append(cases_with_gaps.astype(float))
            wastewater_raw_runs.append(ww_by_edar.astype(float))
            wastewater_censor_runs.append(censor_by_edar.astype(np.int8))
            mobility_kappa0_runs.append(kappa0_series.astype(float))

            scenario_type, strength = parse_run_metadata(run_dir.name)
            if np.isnan(strength):
                strength = (
                    float(np.nanmax(kappa0_series)) if len(kappa0_series) > 0 else 0.0
                )
            scenario_types.append(scenario_type)
            strengths.append(strength)
            run_ids.append(sanitize_run_id(run_dir.name))
            sparsity_levels.append(run_sparsity)

    if not infections_true_runs:
        raise ValueError("No valid runs were processed")

    if dates_ref is None:
        raise ValueError("No valid dates found in run outputs")

    # Stack runs: (Run, Region, Time) for most variables
    # Note: mobility is NOT stacked here - stored factorized as mobility_base + mobility_kappa0
    infections_true_arr = np.stack(infections_true_runs, axis=0)
    hospitalizations_true_arr = np.stack(hospitalizations_true_runs, axis=0)
    deaths_true_arr = np.stack(deaths_true_runs, axis=0)
    hospitalizations_raw_arr = np.stack(hospitalizations_raw_runs, axis=0)
    deaths_raw_arr = np.stack(deaths_raw_runs, axis=0)
    cases_raw_arr = np.stack(cases_raw_runs, axis=0)
    wastewater_raw_arr = np.stack(wastewater_raw_runs, axis=0)
    wastewater_censor_arr = np.stack(wastewater_censor_runs, axis=0)
    mobility_kappa0_arr = np.stack(mobility_kappa0_runs, axis=0)

    # Determine wastewater spatial dimension
    if emap_data is not None:
        ww_spatial_dim = "edar_id"
        logger.info(f"Using EDAR-based wastewater: {len(edar_ids)} EDARs")
    else:
        ww_spatial_dim = "region_id"
        logger.info(f"Using region-based wastewater: {len(region_ids)} regions")

    # Chunk size for efficient access and compression
    chunk_size = args.chunk_size
    region_chunk = min(chunk_size, len(region_ids))
    n_dates = len(dates_ref)
    date_chunk = min(chunk_size, n_dates)

    # Build dataset matching EpiForecaster's expected input format
    # Wastewater is split into separate variables per target (N1, N2, IP4)
    data_vars = {}

    # Ground truth (for evaluation, separate from preprocessed data)
    data_vars["infections_true"] = (
        ("run_id", "region_id", "date"),
        infections_true_arr,
        {},
    )
    data_vars["hospitalizations_true"] = (
        ("run_id", "region_id", "date"),
        hospitalizations_true_arr,
        {},
    )
    data_vars["deaths_true"] = (
        ("run_id", "region_id", "date"),
        deaths_true_arr,
        {},
    )

    # Raw observations (for EpiForecaster preprocessing pipeline)
    # Variable names match EpiForecaster conventions

    # Cases: matches EpiForecaster's expected "cases" variable
    # Dimensions: (run_id, date, region_id) - note: date before region_id to match xarray convention
    cases_arr_transposed = cases_raw_arr.transpose(0, 2, 1)  # (run_id, date, region_id)
    data_vars["cases"] = (
        ("run_id", "date", "region_id"),
        cases_arr_transposed,
        {},
    )

    # Hospitalizations: raw reported hospitalizations with noise
    hosp_arr_transposed = hospitalizations_raw_arr.transpose(
        0, 2, 1
    )  # (run_id, date, region_id)
    data_vars["hospitalizations"] = (
        ("run_id", "date", "region_id"),
        hosp_arr_transposed,
        {},
    )

    # Deaths: raw reported deaths with noise
    deaths_arr_transposed = deaths_raw_arr.transpose(
        0, 2, 1
    )  # (run_id, date, region_id)
    data_vars["deaths"] = (
        ("run_id", "date", "region_id"),
        deaths_arr_transposed,
        {},
    )

    # Check if target zarr has mobility_time_varying (for append mode)
    # This ensures dimension consistency when appending runs with static mobility
    # to a zarr that already has mobility_time_varying from baseline runs
    target_has_mobility_tv = False
    if args.append and args.output:
        try:
            if os.path.exists(args.output):
                existing_ds = xr.open_zarr(args.output, chunks=None)
                target_has_mobility_tv = (
                    "mobility_time_varying" in existing_ds.data_vars
                )
                existing_ds.close()
                if target_has_mobility_tv:
                    logger.info(
                        "Existing zarr has mobility_time_varying. "
                        "Will create for new runs to maintain dimension consistency."
                    )
        except Exception as e:
            logger.warning(
                f"Could not check existing zarr for mobility_time_varying: {e}"
            )

    # Mobility: Always create factorized format (mobility_base + mobility_kappa0)
    # for backward compatibility and append support
    # Add mobility_time_varying when any run has time-varying mobility
    # OR when appending to a zarr that already has mobility_time_varying
    has_time_varying = (
        any(t == "time_varying" for t in mobility_types) or target_has_mobility_tv
    )

    # Always create factorized format variables
    data_vars["mobility_base"] = (
        ("origin", "target"),
        base_mobility,
        {},
    )
    data_vars["mobility_kappa0"] = (
        ("run_id", "date"),
        mobility_kappa0_arr,
        {},
    )

    # Optionally add dense format if any run has time-varying mobility
    if has_time_varying:
        # Convert all runs to dense format
        mobility_all = []
        for run_idx, (mobility_dense, mob_type) in enumerate(
            zip(mobility_runs, mobility_types)
        ):
            if mobility_dense is not None:
                # Already dense (T, M, M)
                mobility_all.append(mobility_dense)
            elif mob_type == "time_varying":
                # ERROR: Marked as time_varying but file not found/loaded
                run_id = run_ids[run_idx]
                raise ValueError(
                    f"Run '{run_id}' is marked as time_varying but mobility_series.npz could not be loaded. "
                    f"This indicates a data integrity issue - refusing to silently fall back to static approximation."
                )
            else:
                # Static: reconstruct from mobility_base + kappa0
                T = len(dates_ref)
                M = len(region_ids)
                mobility_reconstructed = np.zeros((T, M, M))
                for t in range(T):
                    mobility_reconstructed[t] = base_mobility * (
                        1 - mobility_kappa0_arr[len(mobility_all), t]
                    )
                mobility_all.append(mobility_reconstructed)

        # Stack: (N, T, M, M) -> (N, M, M, T)
        mobility_full_arr = np.stack(mobility_all, axis=0).transpose(0, 2, 3, 1)

        data_vars["mobility_time_varying"] = (
            ("run_id", "origin", "target", "date"),
            mobility_full_arr,
            {},
        )

    data_vars["synthetic_mobility_type"] = (
        ("run_id",),
        np.array(mobility_types, dtype=object),
        {},
    )

    # Population: matches EpiForecaster's expected "population" variable
    # Dimensions: (run_id, region_id)
    data_vars["population"] = (
        ("run_id", "region_id"),
        np.tile(population_vector, (len(run_ids), 1)),
        {},
    )

    # Wastewater: split into separate variables per target (matches EpiForecaster output format)
    # Dimensions: (run_id, date, spatial_dim) where spatial_dim is edar_id or region_id
    for target_idx, target_name in enumerate(TARGET_NAMES):
        # Extract data for this target and remove the target dimension
        # wastewater_raw_arr shape is (n_runs, n_time, n_spatial, n_targets)
        # After indexing: (n_runs, n_time, n_spatial) = (run_id, date, spatial_dim)
        ww_target = wastewater_raw_arr[:, :, :, target_idx]  # (run_id, date, spatial)

        censor_target = wastewater_censor_arr[
            :, :, :, target_idx
        ]  # (run_id, date, spatial)

        # Main biomarker variable (matches EpiForecaster naming)
        data_vars[f"edar_biomarker_{target_name}"] = (
            ("run_id", "date", ww_spatial_dim),
            ww_target,
            {},
        )

        # Censor hints (optional, for reference - EpiForecaster may recompute)
        # 0=observed, 1=censored, 2=missing
        data_vars[f"edar_biomarker_{target_name}_censor_hints"] = (
            ("run_id", "date", ww_spatial_dim),
            censor_target,
            {},
        )

    # Per-EDAR LoD values (same for all runs, but stored per-run for consistency)
    # These allow EDARProcessor to detect censored values: if value <= LoD → censored
    n_edar = len(edar_ids) if emap_data is not None else len(region_ids)
    for target_name in TARGET_NAMES:
        lod_value = GENE_TARGETS[target_name]["limit_of_detection"]
        data_vars[f"edar_biomarker_{target_name}_LoD"] = (
            ("run_id", ww_spatial_dim),
            np.full((len(run_ids), n_edar), lod_value),
            {},
        )

    # Synthetic metadata (for reference, not used by preprocessor)
    data_vars["synthetic_scenario_type"] = (
        ("run_id",),
        np.array(scenario_types, dtype=object),
        {},
    )
    data_vars["synthetic_strength"] = (
        ("run_id",),
        np.array(strengths, dtype=float),
        {},
    )
    data_vars["synthetic_sparsity_level"] = (
        ("run_id",),
        np.array(sparsity_levels, dtype=float),
        {},
    )
    data_vars["synthetic_mobility_noise_sigma_O"] = (
        ("run_id",),
        np.array(mobility_sigma_O_runs, dtype=float),
        {},
    )
    data_vars["synthetic_mobility_noise_sigma_D"] = (
        ("run_id",),
        np.array(mobility_sigma_D_runs, dtype=float),
        {},
    )
    data_vars["synthetic_mobility_noise_factor"] = (
        ("run_id",),
        np.array(mobility_noise_runs, dtype=float),
        {},
    )

    # Cases reporting noise metadata
    data_vars["synthetic_cases_report_rate_min"] = (
        ("run_id",),
        np.full(len(run_ids), args.min_rate, dtype=float),
        {},
    )
    data_vars["synthetic_cases_report_rate_max"] = (
        ("run_id",),
        np.full(len(run_ids), args.max_rate, dtype=float),
        {},
    )
    data_vars["synthetic_cases_report_delay_mean"] = (
        ("run_id",),
        np.full(len(run_ids), 0, dtype=float),  # No delay currently modeled for cases
        {},
    )

    # Hospitalizations reporting noise metadata
    data_vars["synthetic_hosp_report_rate"] = (
        ("run_id",),
        np.full(len(run_ids), args.hosp_report_rate, dtype=float),
        {},
    )
    data_vars["synthetic_hosp_report_delay_mean"] = (
        ("run_id",),
        np.full(len(run_ids), args.hosp_delay_mean, dtype=float),
        {},
    )
    data_vars["synthetic_hosp_report_delay_std"] = (
        ("run_id",),
        np.full(len(run_ids), args.hosp_delay_std, dtype=float),
        {},
    )

    # Deaths reporting noise metadata
    data_vars["synthetic_deaths_report_rate"] = (
        ("run_id",),
        np.full(len(run_ids), args.deaths_report_rate, dtype=float),
        {},
    )
    data_vars["synthetic_deaths_report_delay_mean"] = (
        ("run_id",),
        np.full(len(run_ids), args.deaths_delay_mean, dtype=float),
        {},
    )
    data_vars["synthetic_deaths_report_delay_std"] = (
        ("run_id",),
        np.full(len(run_ids), args.deaths_delay_std, dtype=float),
        {},
    )

    # Wastewater noise metadata
    data_vars["synthetic_ww_noise_sigma_N1"] = (
        ("run_id",),
        np.full(len(run_ids), GENE_TARGETS["N1"]["noise_sigma"], dtype=float),
        {},
    )
    data_vars["synthetic_ww_noise_sigma_N2"] = (
        ("run_id",),
        np.full(len(run_ids), GENE_TARGETS["N2"]["noise_sigma"], dtype=float),
        {},
    )
    data_vars["synthetic_ww_noise_sigma_IP4"] = (
        ("run_id",),
        np.full(len(run_ids), GENE_TARGETS["IP4"]["noise_sigma"], dtype=float),
        {},
    )
    data_vars["synthetic_ww_transport_loss"] = (
        ("run_id",),
        np.full(len(run_ids), GENE_TARGETS["N1"]["transport_loss"], dtype=float),
        {},
    )

    # Coordinates
    coords = {
        "run_id": run_ids,
        "region_id": region_ids,
        "date": dates_ref,
        "origin": region_ids,
        "target": region_ids,
    }

    if emap_data is not None:
        coords["edar_id"] = edar_ids

    dataset = xr.Dataset(data_vars, coords=coords)

    # Set chunk sizes for efficient storage and access
    # Ground truth variables
    for var_name in ["infections_true", "hospitalizations_true", "deaths_true"]:
        dataset[var_name].encoding = {"chunksizes": (1, region_chunk, date_chunk)}

    # Cases: (run_id, date, region_id)
    dataset["cases"].encoding = {"chunksizes": (1, date_chunk, region_chunk)}

    # Hospitalizations and Deaths: (run_id, date, region_id)
    dataset["hospitalizations"].encoding = {"chunksizes": (1, date_chunk, region_chunk)}
    dataset["deaths"].encoding = {"chunksizes": (1, date_chunk, region_chunk)}

    # Mobility: Handle both factorized and dense formats
    # Factorized format - always present
    dataset["mobility_base"].encoding = {"chunksizes": (-1, -1)}
    dataset["mobility_kappa0"].encoding = {"chunksizes": (1, date_chunk)}

    # Dense format - optionally present
    if has_time_varying:
        # Dense format: (run_id, origin, target, date)
        # Use efficient chunking: chunk by run_id, then spatial blocks, then time
        mobility_origin_chunk = min(chunk_size, len(region_ids))
        mobility_target_chunk = min(chunk_size, len(region_ids))
        dataset["mobility_time_varying"].encoding = {
            "chunksizes": (1, mobility_origin_chunk, mobility_target_chunk, date_chunk)
        }

    # Population: (run_id, region_id)
    dataset["population"].encoding = {"chunksizes": (1, region_chunk)}

    # Wastewater: (run_id, date, spatial_dim) - separate variables per target
    if emap_data is not None:
        edar_chunk = min(chunk_size, len(edar_ids))
        ww_chunk = edar_chunk
    else:
        ww_chunk = region_chunk

    for target_name in TARGET_NAMES:
        dataset[f"edar_biomarker_{target_name}"].encoding = {
            "chunksizes": (1, date_chunk, ww_chunk)
        }
        dataset[f"edar_biomarker_{target_name}_censor_hints"].encoding = {
            "chunksizes": (1, date_chunk, ww_chunk)
        }
        # LoD variables: (run_id, spatial_dim) - no date dimension
        dataset[f"edar_biomarker_{target_name}_LoD"].encoding = {
            "chunksizes": (1, ww_chunk)
        }

    # Synthetic metadata: small, no chunking needed
    dataset["synthetic_scenario_type"].encoding = {"chunksizes": (len(run_ids),)}
    dataset["synthetic_strength"].encoding = {"chunksizes": (len(run_ids),)}
    dataset["synthetic_sparsity_level"].encoding = {"chunksizes": (len(run_ids),)}
    dataset["synthetic_mobility_type"].encoding = {"chunksizes": (len(run_ids),)}
    dataset["synthetic_mobility_noise_sigma_O"].encoding = {
        "chunksizes": (len(run_ids),)
    }
    dataset["synthetic_mobility_noise_sigma_D"].encoding = {
        "chunksizes": (len(run_ids),)
    }
    dataset["synthetic_mobility_noise_factor"].encoding = {
        "chunksizes": (len(run_ids),)
    }

    # Cases reporting noise metadata
    dataset["synthetic_cases_report_rate_min"].encoding = {
        "chunksizes": (len(run_ids),)
    }
    dataset["synthetic_cases_report_rate_max"].encoding = {
        "chunksizes": (len(run_ids),)
    }
    dataset["synthetic_cases_report_delay_mean"].encoding = {
        "chunksizes": (len(run_ids),)
    }

    # Hospitalizations reporting noise metadata
    dataset["synthetic_hosp_report_rate"].encoding = {"chunksizes": (len(run_ids),)}
    dataset["synthetic_hosp_report_delay_mean"].encoding = {
        "chunksizes": (len(run_ids),)
    }
    dataset["synthetic_hosp_report_delay_std"].encoding = {
        "chunksizes": (len(run_ids),)
    }

    # Deaths reporting noise metadata
    dataset["synthetic_deaths_report_rate"].encoding = {"chunksizes": (len(run_ids),)}
    dataset["synthetic_deaths_report_delay_mean"].encoding = {
        "chunksizes": (len(run_ids),)
    }
    dataset["synthetic_deaths_report_delay_std"].encoding = {
        "chunksizes": (len(run_ids),)
    }

    # Wastewater noise metadata
    dataset["synthetic_ww_noise_sigma_N1"].encoding = {"chunksizes": (len(run_ids),)}
    dataset["synthetic_ww_noise_sigma_N2"].encoding = {"chunksizes": (len(run_ids),)}
    dataset["synthetic_ww_noise_sigma_IP4"].encoding = {"chunksizes": (len(run_ids),)}
    dataset["synthetic_ww_transport_loss"].encoding = {"chunksizes": (len(run_ids),)}

    output_path = args.output

    # Determine write mode
    if args.append:
        if os.path.exists(output_path):
            # Explicit append to existing file
            logger.info(
                "Appending to observation zarr at %s (runs=%s)",
                output_path,
                len(run_ids),
            )
            dataset.to_zarr(output_path, mode="a", append_dim="run_id", zarr_format=2)
        elif args.init:
            # Create new file with --init flag
            logger.info(
                "Initializing new zarr store at %s (runs=%s)",
                output_path,
                len(run_ids),
            )
            dataset.to_zarr(output_path, mode="w", zarr_format=2)
        else:
            # --append without --init and file doesn't exist
            logger.error("Cannot append to non-existent file: %s", output_path)
            logger.error(
                "Use --init to create a new zarr store, or omit --append to overwrite"
            )
            sys.exit(1)
    else:
        # Write mode (overwrite existing or create new)
        if os.path.exists(output_path):
            logger.info("Removing existing output at %s", output_path)
            import shutil

            if os.path.isdir(output_path):
                shutil.rmtree(output_path)
            else:
                os.remove(output_path)

        logger.info(
            "Writing raw observation zarr to %s (runs=%s, regions=%s, dates=%s, targets=%s)",
            output_path,
            len(run_ids),
            len(region_ids),
            len(dates_ref),
            len(TARGET_NAMES),
        )
        dataset.to_zarr(output_path, mode="w", zarr_format=2)

    logger.info("Done! Output written to %s", output_path)


if __name__ == "__main__":
    main()
