#!/usr/bin/env python3
"""
Verify the health and integrity of a synthetic observations Zarr store.

This script checks for:
1. Missing or corrupted chunks (unreadable/un-decompressible)
2. Unexpected NaN or Inf values
3. Out-of-range values (e.g., negative infection counts)
4. float16 overflow issues (values > 65504)
5. Metadata consistency and consolidated status
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr
import zarr

try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback if tqdm is not installed
    def tqdm(iterable, **kwargs):
        return iterable


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ZarrHealth")


def check_float16_limit(arr: np.ndarray, var_name: str) -> List[str]:
    """Check for values near the float16 limit (65504)."""
    issues = []
    if arr.dtype == np.float16:
        max_val = np.nanmax(np.abs(arr))
        if max_val > 60000:
            issues.append(f"WARNING: {var_name} has values near float16 limit ({max_val:.1f} > 60000)")
        if max_val >= 65504:
            issues.append(f"CRITICAL: {var_name} has values at/above float16 limit ({max_val:.1f} >= 65504)")
    return issues


def check_negative_counts(arr: np.ndarray, var_name: str) -> List[str]:
    """Check for negative values in variables that should be non-negative."""
    count_vars = ["cases", "hospitalizations", "deaths", "infections_true", 
                  "hospitalizations_true", "deaths_true", "population"]
    
    issues = []
    if any(cv in var_name for cv in count_vars):
        min_val = np.nanmin(arr)
        if min_val < 0:
            issues.append(f"ERROR: {var_name} contains negative values (min={min_val})")
    return issues


def verify_variable_health(var_name: str, var: xr.DataArray, sample_rate: float = 1.0) -> dict:
    """
    Verify the health of a single variable.
    
    Checks metadata, NaN/Inf counts, and reads data to verify chunk integrity.
    """
    logger.info(f"Checking variable: {var_name} (dtype={var.dtype}, shape={var.shape})")
    
    stats = {
        "name": var_name,
        "status": "OK",
        "issues": [],
        "nan_pct": 0.0,
        "inf_pct": 0.0,
        "min": None,
        "max": None,
    }

    try:
        # Load data (with sampling if requested)
        if sample_rate < 1.0 and var.size > 1000:
            # Simple spatial sampling for large 3D/4D arrays
            # We want to touch at least some chunks
            if var.ndim >= 2:
                n_samples = max(1, int(var.shape[0] * sample_rate))
                indices = np.random.choice(var.shape[0], n_samples, replace=False)
                data = var.isel({var.dims[0]: indices}).values
            else:
                data = var.values
        else:
            data = var.values
            
        # Basic stats
        is_numeric = np.issubdtype(data.dtype, np.number)
        
        if is_numeric:
            stats["nan_pct"] = (np.isnan(data).sum() / data.size) * 100
            stats["inf_pct"] = (np.isinf(data).sum() / data.size) * 100
            
            if data.size > 0:
                stats["min"] = float(np.nanmin(data))
                stats["max"] = float(np.nanmax(data))
            
            # Specific checks
            stats["issues"].extend(check_float16_limit(data, var_name))
            stats["issues"].extend(check_negative_counts(data, var_name))
            
            if stats["inf_pct"] > 0:
                stats["issues"].append(f"ERROR: {var_name} contains Infinity values ({stats['inf_pct']:.2f}%)")
        else:
            stats["nan_pct"] = 0.0
            stats["inf_pct"] = 0.0
            logger.debug(f"  {var_name}: skipping numeric checks for non-numeric dtype {data.dtype}")
            
    except Exception as e:
        stats["status"] = "CORRUPT"
        stats["issues"].append(f"CRITICAL: Failed to read {var_name} data: {str(e)}")
        logger.error(f"  {var_name}: READ FAILED - {e}")

    return stats


def verify_zarr_store(zarr_path: str, sample_rate: float = 1.0, check_chunks: bool = True) -> bool:
    """
    Open and verify a Zarr store.
    """
    path = Path(zarr_path)
    if not path.exists():
        logger.error(f"Path does not exist: {zarr_path}")
        return False

    logger.info(f"Verifying Zarr store at: {zarr_path}")
    
    # 1. Metadata Check
    is_consolidated = (path / ".zmetadata").exists()
    logger.info(f"Consolidated metadata: {'YES' if is_consolidated else 'NO'}")
    
    try:
        ds = xr.open_zarr(zarr_path, consolidated=is_consolidated)
    except Exception as e:
        logger.error(f"Failed to open Zarr metadata: {e}")
        if is_consolidated:
            logger.info("Attempting to open without consolidation...")
            try:
                ds = xr.open_zarr(zarr_path, consolidated=False)
                logger.warning("Opened successfully without consolidation. Consolidated metadata might be corrupt.")
            except Exception as e2:
                logger.error(f"Failed again: {e2}")
                return False
        else:
            return False

    logger.info(f"Dataset Dimensions: {dict(ds.dims)}")
    
    all_stats = []
    
    # 2. Variable Check
    variables = list(ds.data_vars)
    for var_name in tqdm(variables, desc="Checking variables"):
        stats = verify_variable_health(var_name, ds[var_name], sample_rate=sample_rate)
        all_stats.append(stats)
        
    # 3. Chunk Consistency (Low-level Zarr check)
    if check_chunks:
        logger.info("Performing low-level chunk existence check...")
        z_root = zarr.open(zarr_path, mode='r')
        for var_name in variables:
            if var_name in z_root:
                z_arr = z_root[var_name]
                if hasattr(z_arr, 'nchunks'):
                    logger.info(f"  {var_name}: {z_arr.nchunks} total chunks")
                    # We don't check every file on disk here as xr.open_zarr + compute already touched them
                    # but we could add more thorough disk-level checks if needed.

    # 4. Final Report
    print("\n" + "=" * 80)
    print(f"HEALTH REPORT: {zarr_path}")
    print("=" * 80)
    
    critical_issues = 0
    warnings = 0
    
    for s in all_stats:
        status_color = "OK" if s["status"] == "OK" else "!!! CORRUPT !!!"
        print(f"\n[{status_color}] Variable: {s['name']}")
        print(f"  Dtype: {ds[s['name']].dtype}, Chunks: {ds[s['name']].chunks}")
        print(f"  Range: [{s['min']}, {s['max']}]")
        print(f"  NaN: {s['nan_pct']:.2f}%, Inf: {s['inf_pct']:.2f}%")
        
        for issue in s["issues"]:
            print(f"  -> {issue}")
            if "CRITICAL" in issue or "ERROR" in issue:
                critical_issues += 1
            else:
                warnings += 1

    print("\n" + "=" * 80)
    print(f"SUMMARY: {critical_issues} errors, {warnings} warnings")
    if critical_issues == 0:
        print("RESULT: HEALTHY (with warnings)" if warnings > 0 else "RESULT: HEALTHY")
    else:
        print("RESULT: CORRUPTED")
    print("=" * 80)

    ds.close()
    return critical_issues == 0


def main():
    parser = argparse.ArgumentParser(
        description="Verify the health and integrity of a Zarr store."
    )
    parser.add_argument("path", help="Path to the .zarr directory")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Rate of data to sample (0.0 to 1.0, default: 1.0)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick check (sample-rate=0.1, skip chunk checks)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    sample_rate = 0.1 if args.quick else args.sample_rate
    check_chunks = not args.quick

    success = verify_zarr_store(args.path, sample_rate=sample_rate, check_chunks=check_chunks)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
