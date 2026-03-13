#!/usr/bin/env python3
"""
Downsampler script: Convert existing synthetic observation zarr from float64 to float16.

One-time use script to migrate legacy datasets to the new f16 format for memory efficiency.

Usage:
    uv run python downsample_zarr.py --input path/to/input.zarr --output path/to/output.zarr

    # Or overwrite in place (with backup):
    uv run python downsample_zarr.py --input path/to/data.zarr --in-place

    # Dry run to check what would be converted:
    uv run python downsample_zarr.py --input path/to/data.zarr --dry-run

Safety:
    - Validates no overflow would occur before conversion
    - Creates backup when using --in-place
    - Reports memory savings
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Downsampler")

# Variables to convert to float16
FLOAT_VARS_TO_CONVERT = [
    # Observations
    "cases",
    "hospitalizations",
    "deaths",
    # Ground truth
    "infections_true",
    "hospitalizations_true",
    "deaths_true",
    # Wastewater biomarkers
    "edar_biomarker_N1",
    "edar_biomarker_N2",
    "edar_biomarker_IP4",
    # Wastewater metadata
    "edar_biomarker_N1_LoD",
    "edar_biomarker_N2_LoD",
    "edar_biomarker_IP4_LoD",
    # Mobility
    "mobility_base",
    "mobility_kappa0",
    "mobility_time_varying",
    # Synthetic metadata (float arrays)
    "synthetic_strength",
    "synthetic_sparsity_level",
    "synthetic_mobility_noise_sigma_O",
    "synthetic_mobility_noise_sigma_D",
    "synthetic_mobility_noise_factor",
    "synthetic_cases_report_rate_min",
    "synthetic_cases_report_rate_max",
    "synthetic_cases_report_delay_mean",
    "synthetic_hosp_report_rate",
    "synthetic_hosp_report_delay_mean",
    "synthetic_hosp_report_delay_std",
    "synthetic_deaths_report_rate",
    "synthetic_deaths_report_delay_mean",
    "synthetic_deaths_report_delay_std",
    "synthetic_ww_noise_sigma_N1",
    "synthetic_ww_noise_sigma_N2",
    "synthetic_ww_noise_sigma_IP4",
    "synthetic_ww_transport_loss",
]

# Variables to keep as int (or convert to int32 for population)
INT_VARS = [
    "edar_biomarker_N1_censor_hints",
    "edar_biomarker_N2_censor_hints",
    "edar_biomarker_IP4_censor_hints",
    "population",
]

# String/object variables to leave unchanged
STRING_VARS = [
    "synthetic_scenario_type",
    "synthetic_mobility_type",
]


def check_f16_overflow(arr: np.ndarray, var_name: str) -> float:
    """
    Check if array values would overflow float16.

    Returns:
        max_abs_value: Maximum absolute value in array

    Raises:
        ValueError: If overflow would occur
    """
    max_val = np.finfo(np.float16).max  # 65504

    # Handle NaN values
    arr_clean = arr.copy()
    if np.issubdtype(arr.dtype, np.floating):
        arr_clean = arr_clean[~np.isnan(arr_clean)]

    if len(arr_clean) == 0:
        return 0.0

    arr_max = np.max(np.abs(arr_clean))

    if arr_max > max_val:
        raise ValueError(
            f"Cannot convert {var_name} to float16: "
            f"max absolute value {arr_max} exceeds limit {max_val}"
        )

    return float(arr_max)


def get_array_memory_bytes(arr) -> int:
    """Get memory usage of an array in bytes."""
    if hasattr(arr, "nbytes"):
        return arr.nbytes
    elif hasattr(arr, "data"):
        return arr.data.nbytes
    else:
        return 0


def downsample_dataset(
    input_path: str,
    output_path: str,
    dry_run: bool = False,
    compressor: Optional[str] = None,
    compressor_level: int = 3,
) -> dict:
    """
    Downsample a zarr dataset from float64 to float16.

    Args:
        input_path: Path to input zarr
        output_path: Path to output zarr
        dry_run: If True, only validate without writing
        compressor: Compressor to use (zstd, lz4, blosc, none)
        compressor_level: Compression level

    Returns:
        dict with statistics about the conversion
    """
    logger.info(f"Opening input dataset: {input_path}")

    try:
        ds = xr.open_zarr(input_path, chunks=None)
    except Exception as e:
        logger.error(f"Failed to open input zarr: {e}")
        raise

    stats = {
        "variables_processed": 0,
        "variables_skipped": 0,
        "variables_failed": [],
        "original_bytes": 0,
        "converted_bytes": 0,
        "overflow_checks": {},
    }

    # Process each variable
    new_data_vars = {}

    for var_name in ds.data_vars:
        var = ds[var_name]

        logger.debug(f"Processing variable: {var_name} (dtype={var.dtype})")

        # Skip if not in our conversion list and not a float
        if var_name not in FLOAT_VARS_TO_CONVERT and var_name not in INT_VARS:
            if var_name not in STRING_VARS:
                logger.warning(
                    f"Unknown variable '{var_name}', leaving as-is (dtype={var.dtype})"
                )
            new_data_vars[var_name] = var
            stats["variables_skipped"] += 1
            continue

        # Get original memory usage
        orig_bytes = get_array_memory_bytes(var)
        stats["original_bytes"] += orig_bytes

        if var_name in FLOAT_VARS_TO_CONVERT:
            # Convert float64 -> float16
            try:
                arr = var.values
                max_val = check_f16_overflow(arr, var_name)
                stats["overflow_checks"][var_name] = max_val

                converted = arr.astype(np.float16)
                new_bytes = converted.nbytes
                stats["converted_bytes"] += new_bytes

                # Preserve encoding but update dtype
                encoding = var.encoding.copy()
                encoding["dtype"] = "float16"

                new_data_vars[var_name] = xr.DataArray(
                    converted,
                    dims=var.dims,
                    coords=var.coords,
                    name=var_name,
                    attrs=var.attrs,
                )
                new_data_vars[var_name].encoding = encoding

                stats["variables_processed"] += 1
                logger.info(f"  {var_name}: float64 -> float16 (max_val={max_val:.2f})")

            except ValueError as e:
                logger.error(f"  {var_name}: FAILED - {e}")
                stats["variables_failed"].append(var_name)
                # Keep original on failure
                new_data_vars[var_name] = var
                stats["converted_bytes"] += orig_bytes

        elif var_name == "population":
            # Convert to int32
            try:
                arr = var.values
                converted = arr.astype(np.int32)
                new_bytes = converted.nbytes
                stats["converted_bytes"] += new_bytes

                encoding = var.encoding.copy()
                encoding["dtype"] = "int32"

                new_data_vars[var_name] = xr.DataArray(
                    converted,
                    dims=var.dims,
                    coords=var.coords,
                    name=var_name,
                    attrs=var.attrs,
                )
                new_data_vars[var_name].encoding = encoding

                stats["variables_processed"] += 1
                logger.info(f"  {var_name}: {var.dtype} -> int32")

            except Exception as e:
                logger.error(f"  {var_name}: FAILED - {e}")
                stats["variables_failed"].append(var_name)
                new_data_vars[var_name] = var
                stats["converted_bytes"] += orig_bytes

        elif var_name in INT_VARS:
            # Keep int arrays as-is (they're already small)
            new_data_vars[var_name] = var
            stats["converted_bytes"] += orig_bytes
            stats["variables_skipped"] += 1

    # Build new dataset
    new_ds = xr.Dataset(new_data_vars, coords=ds.coords, attrs=ds.attrs)

    # Report statistics
    orig_mb = stats["original_bytes"] / (1024 * 1024)
    new_mb = stats["converted_bytes"] / (1024 * 1024)
    savings_pct = (
        (1 - stats["converted_bytes"] / stats["original_bytes"]) * 100
        if stats["original_bytes"] > 0
        else 0
    )

    logger.info(f"\nDownsampling Summary:")
    logger.info(f"  Variables processed: {stats['variables_processed']}")
    logger.info(f"  Variables skipped: {stats['variables_skipped']}")
    if stats["variables_failed"]:
        logger.info(f"  Variables failed: {stats['variables_failed']}")
    logger.info(f"  Original size: {orig_mb:.1f} MB")
    logger.info(f"  Converted size: {new_mb:.1f} MB")
    logger.info(f"  Savings: {savings_pct:.1f}%")

    if dry_run:
        logger.info("\nDry run complete - no files written")
        ds.close()
        return stats

    # Write output
    if os.path.exists(output_path):
        logger.info(f"Removing existing output: {output_path}")
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        else:
            os.remove(output_path)

    logger.info(f"Writing output to: {output_path}")

    # Set up compressor
    if compressor:
        import zarr

        if compressor == "zstd":
            comp = zarr.Zstd(level=compressor_level)
        elif compressor == "lz4":
            comp = zarr.LZ4()
        elif compressor == "blosc":
            comp = zarr.Blosc()
        else:
            comp = None

        if comp:
            for var_name in new_ds.data_vars:
                new_ds[var_name].encoding["compressor"] = comp

    new_ds.to_zarr(output_path, mode="w", zarr_format=2)
    ds.close()

    logger.info("Downsampling complete!")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Downsample synthetic observation zarr from float64 to float16"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input zarr file (float64 format)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to output zarr file (float16 format)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify input file in-place (creates .backup first)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate conversion without writing files",
    )
    parser.add_argument(
        "--compressor",
        default="zstd",
        choices=["zstd", "lz4", "blosc", "none"],
        help="Compressor for output zarr (default: zstd)",
    )
    parser.add_argument(
        "--compressor-level",
        type=int,
        default=3,
        help="Compression level (default: 3)",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompts",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.dry_run and not args.output and not args.in_place:
        parser.error("Must specify --output, --in-place, or --dry-run")

    if args.output and args.in_place:
        parser.error("Cannot use both --output and --in-place")

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Determine output path
    if args.in_place:
        output_path = str(input_path)
        backup_path = str(input_path) + ".backup"

        if not args.yes and not args.dry_run:
            response = input(
                f"This will modify {input_path} in-place. Create backup at {backup_path}? [y/N] "
            )
            if response.lower() != "y":
                logger.info("Aborted")
                sys.exit(0)

        if not args.dry_run:
            # Create backup
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            logger.info(f"Creating backup: {backup_path}")
            shutil.copytree(input_path, backup_path)
    else:
        output_path = args.output

    # Run conversion
    try:
        stats = downsample_dataset(
            str(input_path),
            output_path,
            dry_run=args.dry_run,
            compressor=args.compressor if args.compressor != "none" else None,
            compressor_level=args.compressor_level,
        )

        if stats["variables_failed"]:
            logger.error(f"\nConversion had {len(stats['variables_failed'])} failures!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
