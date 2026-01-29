#!/usr/bin/env python3
"""
Master script to run the full synthetic data generation pipeline.

Pipeline stages:
1. Generate configs and run SIR simulations (synthetic_generator.py)
2. Process outputs into raw-ish observations for EpiForecaster (process_synthetic_outputs.py)
3. Plot results (plot_synthetic_results.py)
4. Plot zarr-based epicurves with lockdown highlighting (plot_zarr_epicurves.py)

Usage:
    uv run python/run_synthetic_pipeline.py [--dataset DATASET] [--clean] [--skip-sim] [--skip-process] [--skip-plot] [--skip-zarr-plot]

Arguments:
    --dataset: Dataset to use (default: catalonia, options: catalonia, mitma)
    --clean: Clean output folder before running
    --skip-sim: Skip simulation stage (use existing runs)
    --skip-process: Skip processing stage (use existing zarr)
    --skip-plot: Skip plotting stage
    --skip-zarr-plot: Skip zarr epicurves plotting stage

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
DEFAULT_DATASET = "catalonia"  # Options: catalonia, mitma
DATA_FOLDER = PROJECT_ROOT / "models" / DEFAULT_DATASET
CONFIG_PATH = DATA_FOLDER / "config_MMCACovid19.json"
OUTPUT_FOLDER = PROJECT_ROOT / "runs" / f"synthetic_{DEFAULT_DATASET}"
METAPOP_CSV = DATA_FOLDER / "metapopulation_data.csv"

# Dataset configurations
DATASET_CONFIGS = {
    "catalonia": {
        "data_folder": PROJECT_ROOT / "models" / "catalonia",
        "config": "config_MMCACovid19.json",
        "output_suffix": "synthetic_catalonia",
    },
    "mitma": {
        "data_folder": PROJECT_ROOT / "models" / "mitma",
        "config": "config_MMCACovid19.json",
        "output_suffix": "synthetic_test",
    },
}


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


def run_simulation_batch(n_profiles, start_idx, end_idx, clean_output_dir=False, failure_tolerance=10):
    """Stage 1: Generate configs and run SIR simulations for a batch."""
    global DATA_FOLDER, OUTPUT_FOLDER, CONFIG_PATH

    cmd = [
        sys.executable,
        "python/synthetic_generator.py",
        "--n-profiles",
        str(n_profiles),
        "--start-index",
        str(start_idx),
        "--end-index",
        str(end_idx),
        "--data-folder",
        str(DATA_FOLDER),
        "--output-folder",
        str(OUTPUT_FOLDER),
        "--config",
        str(CONFIG_PATH),
        "--failure-tolerance",
        str(failure_tolerance),
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


def process_outputs_batch(append=False, edar_edges=None):
    """Stage 2: Process outputs into raw-ish observations for EpiForecaster preprocessing."""
    global OUTPUT_FOLDER, METAPOP_CSV
    zarr_output = OUTPUT_FOLDER / "raw_synthetic_observations.zarr"

    cmd = [
        sys.executable,
        "python/process_synthetic_outputs.py",
        "--runs-dir",
        str(OUTPUT_FOLDER),
        "--metapop-csv",
        str(METAPOP_CSV),
        "--output",
        str(zarr_output),
    ]

    if append:
        cmd.append("--append")

    if edar_edges:
        cmd.extend(["--edar-edges", str(edar_edges)])

    return run_stage(f"Process Outputs (Append={append})", cmd)


def clean_run_folders():
    """Remove run folders but keep the Zarr output."""
    global OUTPUT_FOLDER
    logger.info(f"Cleaning run folders in {OUTPUT_FOLDER}")
    for item in OUTPUT_FOLDER.iterdir():
        if item.is_dir() and item.name.startswith("run_"):
            shutil.rmtree(item)
    logger.info("Run folders cleaned.")


def clean_output():
    """Remove previous run outputs."""
    global OUTPUT_FOLDER
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
    global OUTPUT_FOLDER
    zarr_path = OUTPUT_FOLDER / "raw_synthetic_observations.zarr"
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


def run_iterative_pipeline(n_profiles, batch_size, skip_sim=False, skip_process=False, edar_edges=None, failure_tolerance=10):
    """Execute the pipeline in batches to save disk space."""
    global OUTPUT_FOLDER

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
                failure_tolerance=failure_tolerance,
            )

        # 2. Process Batch
        if not skip_process:
            # Append if this is not the first batch OR if we are resuming
            # Actually, logic is: Append if batch_idx > 0.
            # If batch_idx == 0, we overwrite (create new).

            is_append = batch_idx > 0
            process_outputs_batch(append=is_append, edar_edges=edar_edges)

        logger.info(f"=== Batch {batch_idx + 1} Completed ===")

    # Final cleanup of run folders
    if not skip_sim:
        clean_run_folders()


def run_two_phase_pipeline(n_profiles=15, spike_threshold=0.1, skip_sim=False, skip_process=False, edar_edges=None, failure_tolerance=10):
    """Execute the two-phase synthetic data generation pipeline.

    Phase 1: Generate and run all baseline scenarios (no interventions)
    Phase 2: Analyze baseline outputs to detect infection spikes, then generate
             intervention scenarios with realistic timing based on observed spikes

    Args:
        n_profiles: Number of epidemiological profiles to generate
        spike_threshold: Percentile threshold for spike detection (default: 0.1 = 10th percentile)
        skip_sim: Skip simulation stages (use existing runs)
        skip_process: Skip processing stages (use existing zarr)
        edar_edges: Path to EDAR-municipality edges file
        failure_tolerance: Number of consecutive failures before aborting
    """
    global OUTPUT_FOLDER, METAPOP_CSV, DATA_FOLDER, CONFIG_PATH

    output_base = PROJECT_ROOT / "runs" / "synthetic_two_phase"
    baseline_dir = output_base / "baselines"
    intervention_dir = output_base / "interventions"

    # Update global OUTPUT_FOLDER for process_outputs_batch
    # We'll update it per phase
    original_output_folder = OUTPUT_FOLDER

    # Phase 1: Generate baselines
    print("=" * 60)
    print("PHASE 1: Generating baseline scenarios...")
    print("=" * 60)

    if not skip_sim:
        # Clean previous baselines if starting fresh
        if baseline_dir.exists() and not skip_process:
            logger.info(f"Cleaning baseline directory: {baseline_dir}")
            shutil.rmtree(baseline_dir)
        baseline_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "python/synthetic_generator.py",
            "--n-profiles", str(n_profiles),
            "--output-folder", str(baseline_dir),
            "--data-folder", str(DATA_FOLDER),
            "--config", str(CONFIG_PATH),
            "--baseline-only",
            "--failure-tolerance", str(failure_tolerance),
        ]

        run_stage("Generate Baselines", cmd)

    # Process baseline outputs
    if not skip_process:
        OUTPUT_FOLDER = baseline_dir  # Update global for process_outputs_batch
        baseline_zarr = baseline_dir / "raw_synthetic_observations.zarr"

        cmd = [
            sys.executable,
            "python/process_synthetic_outputs.py",
            "--runs-dir", str(baseline_dir),
            "--metapop-csv", str(METAPOP_CSV),
            "--output", str(baseline_zarr),
            "--baseline-only",
        ]

        if edar_edges:
            cmd.extend(["--edar-edges", str(edar_edges)])

        run_stage("Process Baselines", cmd)

    # Phase 2: Generate spike-based interventions
    print("\n" + "=" * 60)
    print("PHASE 2: Generating spike-based intervention scenarios...")
    print("=" * 60)

    if not skip_sim:
        # Clean previous interventions if starting fresh
        if intervention_dir.exists() and not skip_process:
            logger.info(f"Cleaning intervention directory: {intervention_dir}")
            shutil.rmtree(intervention_dir)
        intervention_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "python/synthetic_generator.py",
            "--intervention-only", str(baseline_dir),
            "--spike-threshold", str(spike_threshold),
            "--output-folder", str(intervention_dir),
            "--data-folder", str(DATA_FOLDER),
            "--config", str(CONFIG_PATH),
            "--failure-tolerance", str(failure_tolerance),
        ]

        run_stage("Generate Spike-Based Interventions", cmd)

    # Process intervention outputs and append to baseline zarr
    if not skip_process:
        OUTPUT_FOLDER = intervention_dir  # Update global for process_outputs_batch
        baseline_zarr = baseline_dir / "raw_synthetic_observations.zarr"

        cmd = [
            sys.executable,
            "python/process_synthetic_outputs.py",
            "--runs-dir", str(intervention_dir),
            "--metapop-csv", str(METAPOP_CSV),
            "--output", str(baseline_zarr),
            "--append",
        ]

        if edar_edges:
            cmd.extend(["--edar-edges", str(edar_edges)])

        run_stage("Process Interventions (Append)", cmd)

    # Restore original OUTPUT_FOLDER
    OUTPUT_FOLDER = original_output_folder

    print("\n" + "=" * 60)
    print("Two-Phase Pipeline Complete!")
    print(f"Final output: {baseline_dir / 'raw_synthetic_observations.zarr'}")
    print("=" * 60)


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
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["catalonia", "mitma"],
        default=DEFAULT_DATASET,
        help=f"Dataset to use (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--edar-edges",
        type=str,
        default=None,
        help="Path to EDAR-municipality edges NetCDF file (default: python/data/edar_muni_edges.nc)",
    )
    parser.add_argument(
        "--failure-tolerance",
        type=int,
        default=10,
        help="Number of consecutive failures before aborting (default: 10)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy single-phase pipeline instead of default two-phase spike-based pipeline",
    )
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=0.1,
        help="Spike detection threshold percentile for two-phase pipeline (default: 0.1 = 10th percentile)",
    )

    args = parser.parse_args()
    # Default to two-phase mode unless --legacy is specified
    args.two_phase = not args.legacy

    # Update paths based on dataset selection
    global DATA_FOLDER, CONFIG_PATH, OUTPUT_FOLDER, METAPOP_CSV
    dataset_config = DATASET_CONFIGS[args.dataset]
    DATA_FOLDER = dataset_config["data_folder"]
    CONFIG_PATH = DATA_FOLDER / dataset_config["config"]
    OUTPUT_FOLDER = PROJECT_ROOT / "runs" / dataset_config["output_suffix"]
    METAPOP_CSV = DATA_FOLDER / "metapopulation_data.csv"

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
        if args.two_phase:
            logger.info("DRY RUN - Two-Phase Pipeline:")
            logger.info(f"  Total Profiles: {args.n_profiles}")
            logger.info(f"  Spike Threshold: {args.spike_threshold}")
        else:
            logger.info("DRY RUN - Iterative Pipeline:")
            logger.info(f"  Total Profiles: {args.n_profiles}")
            logger.info(f"  Batch Size: {args.batch_size}")
            logger.info(
                f"  Batches: {(args.n_profiles + args.batch_size - 1) // args.batch_size}"
            )
        return

    # Execute Two-Phase Pipeline (if enabled) or Iterative Pipeline
    if args.two_phase:
        try:
            run_two_phase_pipeline(
                n_profiles=args.n_profiles,
                spike_threshold=args.spike_threshold,
                skip_sim=args.skip_sim,
                skip_process=args.skip_process,
                edar_edges=args.edar_edges,
                failure_tolerance=args.failure_tolerance,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Two-phase pipeline failed with exit code {e.returncode}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Two-phase pipeline failed with exception: {e}")
            sys.exit(1)
    else:
        # Standard Iterative Pipeline
        try:
            run_iterative_pipeline(
                args.n_profiles,
                args.batch_size,
                skip_sim=args.skip_sim,
                skip_process=args.skip_process,
                edar_edges=args.edar_edges,
                failure_tolerance=args.failure_tolerance,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Pipeline failed with exit code {e.returncode}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Pipeline failed with exception: {e}")
            sys.exit(1)

    # Run Plotting Stages (on the final consolidated Zarr)
    # Update OUTPUT_FOLDER for two-phase mode
    if args.two_phase:
        OUTPUT_FOLDER = PROJECT_ROOT / "runs" / "synthetic_two_phase" / "baselines"

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
