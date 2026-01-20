#!/usr/bin/env python3
"""
Master script to run the full synthetic data generation pipeline.

Pipeline stages:
1. Generate configs and run SIR simulations (synthetic_generator.py)
2. Process outputs into wastewater observations (process_synthetic_outputs.py)
3. Plot results (plot_synthetic_results.py)
4. Plot zarr-based epicurves with lockdown highlighting (plot_zarr_epicurves.py)

Usage:
    uv run python/run_synthetic_pipeline.py [--clean] [--skip-sim] [--skip-process] [--skip-plot] [--skip-zarr-plot]

Environment variables:
    JULIA_PROJECT: Path to EpiSim.jl project (default: ../)
    EPISIM_EXECUTABLE_PATH: Optional path to compiled episim executable
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Pipeline")

# Default paths (relative to this script)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT / "models" / "mitma"
CONFIG_PATH = DATA_FOLDER / "config_MMCACovid19.json"
OUTPUT_FOLDER = PROJECT_ROOT / "runs" / "synthetic_test"
METAPOP_CSV = DATA_FOLDER / "metapopulation_data.csv"


def run_stage(name, cmd, cwd=None):
    """Run a pipeline stage with proper error handling."""
    logger.info(f"=== Stage: {name} ===")
    logger.info(f"Command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        check=True,
        capture_output=False,
        text=True,
    )

    logger.info(f"Stage '{name}' completed successfully")
    return result


def run_simulation_batch(n_profiles, start_idx, end_idx, clean_output_dir=False):
    """Stage 1: Generate configs and run SIR simulations for a batch."""
    cmd = [
        sys.executable,
        "python/synthetic_generator.py",
        "--n-profiles",
        str(n_profiles),
        "--start-index",
        str(start_idx),
        "--end-index",
        str(end_idx),
    ]

    if clean_output_dir:
        cmd.append("--clean")

    # Set up environment for Julia
    env = os.environ.copy()
    env["JULIA_PROJECT"] = str(PROJECT_ROOT)

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=True,
        env=env,
        capture_output=False,
        text=True,
    )
    return result


def process_outputs_batch(append=False):
    """Stage 2: Process outputs into wastewater observations."""
    zarr_output = OUTPUT_FOLDER / "synthetic_observations.zarr"

    cmd = [
        sys.executable,
        "python/process_synthetic_outputs.py",
        "--runs-dir",
        str(OUTPUT_FOLDER),
        "--metapop-csv",
        str(METAPOP_CSV),
        "--output",
        str(zarr_output),
        "--preview-plot",
        "--preview-max",
        "1",
    ]

    if append:
        cmd.append("--append")

    return run_stage(f"Process Outputs (Append={append})", cmd)


def clean_run_folders():
    """Remove run folders but keep the Zarr output."""
    logger.info(f"Cleaning run folders in {OUTPUT_FOLDER}")
    for item in OUTPUT_FOLDER.iterdir():
        if item.is_dir() and item.name.startswith("run_"):
            shutil.rmtree(item)
    logger.info("Run folders cleaned.")


def clean_output():
    """Remove previous run outputs."""
    if OUTPUT_FOLDER.exists():
        logger.info(f"Cleaning output folder: {OUTPUT_FOLDER}")
        shutil.rmtree(OUTPUT_FOLDER)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created clean output folder: {OUTPUT_FOLDER}")


def plot_results():
    """Stage 3: Plot results."""
    # Use absolute path since script changes directory
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "python" / "plot_synthetic_results.py"),
    ]

    # Temporarily change to python directory for relative paths in the script
    return run_stage("Plot Results", cmd, cwd=PROJECT_ROOT / "python")


def plot_zarr_epicurves():
    """Stage 4: Plot zarr-based epicurves with lockdown highlighting."""
    zarr_path = OUTPUT_FOLDER / "synthetic_observations.zarr"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "python" / "plot_zarr_epicurves.py"),
        "--zarr",
        str(zarr_path),
        "--format",
        "faceted",
        "--output-dir",
        str(OUTPUT_FOLDER),
    ]
    return run_stage("Plot Zarr Epicurves", cmd)


def run_iterative_pipeline(n_profiles, batch_size, skip_sim=False, skip_process=False):
    """Execute the pipeline in batches to save disk space."""

    # Clean everything initially if this is a fresh start
    if not skip_sim and not skip_process:
        if OUTPUT_FOLDER.exists():
            # We want to keep the folder but remove contents to start fresh
            # BUT we must be careful not to delete if user wanted to resume?
            # For now, let's assume a fresh run implies cleaning
            pass

    total_batches = (n_profiles + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_profiles)

        logger.info(
            f"=== Processing Batch {batch_idx + 1}/{total_batches} (Profiles {start_idx}-{end_idx}) ==="
        )

        # 1. Simulate Batch
        if not skip_sim:
            # Clean run folders from previous batch before starting new one
            # BUT do NOT remove the Zarr file
            clean_run_folders()

            run_simulation_batch(
                n_profiles=n_profiles,
                start_idx=start_idx,
                end_idx=end_idx,
                clean_output_dir=False,  # We handle cleaning manually
            )

        # 2. Process Batch
        if not skip_process:
            # Append if this is not the first batch OR if we are resuming
            # Actually, logic is: Append if batch_idx > 0.
            # If batch_idx == 0, we overwrite (create new).

            is_append = batch_idx > 0
            process_outputs_batch(append=is_append)

        logger.info(f"=== Batch {batch_idx + 1} Completed ===")

    # Final cleanup of run folders
    if not skip_sim:
        clean_run_folders()


def main():
    parser = argparse.ArgumentParser(
        description="Run the full synthetic data generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean output folder before running",
    )
    parser.add_argument(
        "--skip-sim",
        action="store_true",
        help="Skip simulation stage (use existing runs)",
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="Skip processing stage (use existing zarr)",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip plotting stage",
    )
    parser.add_argument(
        "--skip-zarr-plot",
        action="store_true",
        help="Skip zarr epicurves plotting stage",
    )
    parser.add_argument(
        "--n-profiles",
        type=int,
        default=15,
        help="Number of epidemiological profiles to generate (default: 15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of profiles to process in each batch to save disk space (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )

    args = parser.parse_args()

    # Validate paths
    if not DATA_FOLDER.exists():
        logger.error(f"Data folder not found: {DATA_FOLDER}")
        sys.exit(1)
    if not CONFIG_PATH.exists():
        logger.error(f"Config not found: {CONFIG_PATH}")
        sys.exit(1)
    if not METAPOP_CSV.exists():
        logger.error(f"Metapop CSV not found: {METAPOP_CSV}")
        sys.exit(1)

    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Data Folder: {DATA_FOLDER}")
    logger.info(f"Output Folder: {OUTPUT_FOLDER}")

    # Clean if requested (Global Clean)
    if args.clean:
        clean_output()
    else:
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logger.info("DRY RUN - Iterative Pipeline:")
        logger.info(f"  Total Profiles: {args.n_profiles}")
        logger.info(f"  Batch Size: {args.batch_size}")
        logger.info(
            f"  Batches: {(args.n_profiles + args.batch_size - 1) // args.batch_size}"
        )
        return

    # Execute Iterative Pipeline (Sim + Process)
    try:
        run_iterative_pipeline(
            args.n_profiles,
            args.batch_size,
            skip_sim=args.skip_sim,
            skip_process=args.skip_process,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Pipeline failed with exit code {e.returncode}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with exception: {e}")
        sys.exit(1)

    # Run Plotting Stages (on the final consolidated Zarr)
    stages = []

    if not args.skip_plot:
        stages.append(("Plot Results", plot_results))

    if not args.skip_zarr_plot:
        stages.append(("Plot Zarr Epicurves", plot_zarr_epicurves))

    for name, stage_fn in stages:
        try:
            stage_fn()
        except subprocess.CalledProcessError as e:
            logger.error(f"Stage '{name}' failed with exit code {e.returncode}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Stage '{name}' failed with exception: {e}")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info(f"Results in: {OUTPUT_FOLDER}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
