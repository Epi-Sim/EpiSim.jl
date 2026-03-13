#!/usr/bin/env python3
"""
One-time migration script to add cases_mask and cases_age to existing synthetic observation zarr files.

Usage:
    python add_cases_vars_to_zarr.py <zarr_path> [--runs-dir <runs_dir>] [--output <output_path>]

If --runs-dir is provided, will reprocess runs to generate accurate cases_age.
Otherwise, cases_age will be estimated from cases based on population distribution.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AddCasesVars")


def add_cases_mask(ds: xr.Dataset) -> xr.Dataset:
    """Add cases_mask derived from cases (NaN = missing)."""
    if "cases_mask" in ds.data_vars:
        logger.info("cases_mask already exists, skipping")
        return ds

    if "cases" not in ds.data_vars:
        raise ValueError("'cases' variable not found in dataset")

    logger.info("Generating cases_mask from cases...")

    # Create mask: 1.0 where observed (not NaN), 0.0 where missing (NaN)
    cases_mask = xr.where(np.isnan(ds["cases"]), 0.0, 1.0)

    # Add to dataset
    ds["cases_mask"] = cases_mask.astype(ds["cases"].dtype)
    ds["cases_mask"].attrs["description"] = (
        "Observation mask: 1.0=observed, 0.0=missing"
    )

    logger.info("cases_mask added successfully")
    return ds


def estimate_cases_age_from_population(ds: xr.Dataset) -> xr.Dataset:
    """
    Estimate cases_age from cases based on population distribution.

    This is a fallback when run directories are not available.
    Assumes uniform age distribution in population.
    """
    if "cases_age" in ds.data_vars:
        logger.info("cases_age already exists, skipping")
        return ds

    if "cases" not in ds.data_vars:
        raise ValueError("'cases' variable not found in dataset")

    logger.info("Estimating cases_age from cases and population...")

    # Default to 3 age groups if not specified
    n_age_groups = 3

    # Get cases dimensions
    cases = ds["cases"]  # (run_id, date, region_id)
    n_runs = ds.sizes["run_id"]
    n_dates = ds.sizes["date"]
    n_regions = ds.sizes["region_id"]

    # Estimate age distribution from population if available
    if "population" in ds.data_vars:
        pop = ds["population"]  # (run_id, region_id)
        # Assume uniform age distribution: divide population equally among age groups
        pop_per_age = pop / n_age_groups

        # Estimate cases per age group proportionally
        # This assumes cases are distributed like population (not accurate but a fallback)
        cases_age = np.zeros(
            (n_runs, n_dates, n_regions, n_age_groups), dtype=cases.dtype
        )

        for run_idx in range(n_runs):
            for date_idx in range(n_dates):
                total_cases = cases.isel(run_id=run_idx, date=date_idx).values
                # Simple proportional allocation
                cases_per_age = total_cases[:, None] / n_age_groups
                cases_age[run_idx, date_idx, :, :] = cases_per_age
    else:
        # No population data, just divide cases equally
        logger.warning(
            "No population data found, dividing cases equally among age groups"
        )
        cases_age = np.zeros(
            (n_runs, n_dates, n_regions, n_age_groups), dtype=cases.dtype
        )
        for i in range(n_age_groups):
            cases_age[:, :, :, i] = cases.values / n_age_groups

    # Add age_group coordinate
    ds = ds.assign_coords(age_group=[str(i) for i in range(n_age_groups)])

    # Add to dataset
    ds["cases_age"] = xr.DataArray(
        cases_age,
        dims=["run_id", "date", "region_id", "age_group"],
        coords={
            "run_id": ds.coords["run_id"],
            "date": ds.coords["date"],
            "region_id": ds.coords["region_id"],
            "age_group": ds.coords["age_group"],
        },
        attrs={
            "description": "Age-stratified reported cases (estimated from population distribution)",
            "note": "This is an estimate. For accurate values, reprocess from run directories.",
        },
    )

    logger.info("cases_age added (estimated from population)")
    return ds


def reprocess_cases_age_from_runs(ds: xr.Dataset, runs_dir: Path) -> xr.Dataset:
    """
    Reprocess cases_age from original run directories.

    This requires access to the original simulation outputs (observables.nc files).
    """
    if "cases_age" in ds.data_vars:
        logger.info("cases_age already exists, skipping")
        return ds

    logger.info(f"Reprocessing cases_age from runs in {runs_dir}...")

    # Import the necessary functions from process_synthetic_outputs
    sys.path.insert(0, str(Path(__file__).parent))
    from process_synthetic_outputs import (
        generate_reported_cases,
        load_infections_stratified,
        load_run_artifacts,
        parse_run_metadata,
        sanitize_run_id,
    )
    from synthetic_observations import DEFAULT_REPORTED_CASES_CONFIG

    n_age_groups = None
    cases_age_list = []

    for run_id in ds.coords["run_id"].values:
        # Find run directory
        run_dir_name = f"run_{run_id.strip()}"
        run_dir = runs_dir / run_dir_name

        if not run_dir.exists():
            # Try without run_ prefix
            run_dir = runs_dir / run_id.strip()

        if not run_dir.exists():
            logger.warning(f"Run directory not found for {run_id}, skipping")
            continue

        # Load artifacts
        artifacts = load_run_artifacts(run_dir)
        if artifacts is None:
            logger.warning(f"Could not load artifacts for {run_id}, skipping")
            continue

        config, observables_path = artifacts

        # Load stratified infections
        try:
            infections_stratified = load_infections_stratified(observables_path)
            n_time, n_regions, n_age = infections_stratified.shape

            if n_age_groups is None:
                n_age_groups = n_age

            # Generate reported cases per age group
            # Use default config (we don't have the original config stored)
            reported_cfg = DEFAULT_REPORTED_CASES_CONFIG.copy()

            cases_age_stratified = np.zeros(
                (n_time, n_regions, n_age_groups), dtype=int
            )
            rng = np.random.default_rng(42)  # Fixed seed for reproducibility

            for g in range(n_age_groups):
                if g < infections_stratified.shape[2]:
                    reported_cases_g, _ = generate_reported_cases(
                        infections_stratified[:, :, g],
                        config=reported_cfg,
                        rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
                    )
                    cases_age_stratified[:, :, g] = reported_cases_g

            cases_age_list.append(cases_age_stratified)
            logger.info(f"Processed {run_id}")

        except Exception as e:
            logger.error(f"Error processing {run_id}: {e}")
            continue

    if not cases_age_list:
        raise ValueError("No runs could be processed for cases_age")

    # Stack and add to dataset
    cases_age_arr = np.stack(cases_age_list, axis=0)  # (run_id, time, region, age)

    # Add age_group coordinate
    ds = ds.assign_coords(age_group=[str(i) for i in range(n_age_groups)])

    # Add to dataset
    ds["cases_age"] = xr.DataArray(
        cases_age_arr,
        dims=["run_id", "date", "region_id", "age_group"],
        coords={
            "run_id": ds.coords["run_id"],
            "date": ds.coords["date"],
            "region_id": ds.coords["region_id"],
            "age_group": ds.coords["age_group"],
        },
        attrs={
            "description": "Age-stratified reported cases (reprocessed from run directories)",
        },
    )

    logger.info(f"cases_age added (reprocessed from {len(cases_age_list)} runs)")
    return ds


def set_encodings(ds: xr.Dataset, chunk_size: int = 256) -> xr.Dataset:
    """Set chunking encodings for new variables."""
    n_dates = ds.sizes["date"]
    n_regions = ds.sizes["region_id"]
    date_chunk = min(chunk_size, n_dates)
    region_chunk = min(chunk_size, n_regions)

    if "cases_mask" in ds.data_vars:
        ds["cases_mask"].encoding = {"chunksizes": (1, date_chunk, region_chunk)}

    if "cases_age" in ds.data_vars:
        n_age_groups = ds.sizes.get("age_group", 3)
        age_chunk = min(chunk_size, n_age_groups)
        ds["cases_age"].encoding = {
            "chunksizes": (1, date_chunk, region_chunk, age_chunk)
        }

    return ds


def main():
    parser = argparse.ArgumentParser(
        description="Add cases_mask and cases_age to existing synthetic observation zarr files."
    )
    parser.add_argument("zarr_path", help="Path to existing zarr file")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help="Directory containing original run_* folders (for accurate cases_age)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path (default: modify zarr in-place)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size for new variables (default: 256)",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate cases_age from population (if runs-dir not provided)",
    )

    args = parser.parse_args()

    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr file not found: {zarr_path}")

    logger.info(f"Loading zarr file: {zarr_path}")
    ds = xr.open_zarr(zarr_path, chunks=None)

    # Track if any changes were made
    modified = False

    # Add cases_mask (always possible)
    if "cases_mask" not in ds.data_vars:
        ds = add_cases_mask(ds)
        modified = True
    else:
        logger.info("cases_mask already exists, skipping")

    # Add cases_age (requires runs-dir or estimation)
    if "cases_age" not in ds.data_vars:
        if args.runs_dir:
            ds = reprocess_cases_age_from_runs(ds, args.runs_dir)
            modified = True
        elif args.estimate:
            ds = estimate_cases_age_from_population(ds)
            modified = True
        else:
            logger.error(
                "cases_age not found and neither --runs-dir nor --estimate provided. "
                "Use --runs-dir <path> for accurate values or --estimate for approximation."
            )
            sys.exit(1)
    else:
        logger.info("cases_age already exists, skipping")

    if not modified:
        logger.info("No modifications needed, exiting")
        return

    # Set encodings
    ds = set_encodings(ds, chunk_size=args.chunk_size)

    # Save
    output_path = args.output if args.output else str(zarr_path)
    logger.info(f"Saving to: {output_path}")

    # If output is same as input, we need to use mode='a' to append/modify
    if output_path == str(zarr_path):
        ds.to_zarr(output_path, mode="a", consolidated=True)
    else:
        ds.to_zarr(output_path, mode="w", consolidated=True)

    logger.info("Done!")


if __name__ == "__main__":
    main()
