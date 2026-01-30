#!/usr/bin/env python3
"""
Master script to run the full synthetic data generation pipeline.

Two-Phase Spike-Based Pipeline:
1. Phase 1: Generate and run all baseline scenarios (no interventions)
2. Phase 2: Analyze baseline outputs to detect infection spikes, then generate
             intervention scenarios with realistic timing based on observed spikes
3. Process outputs into raw-ish observations for EpiForecaster
4. Plot results (plot_synthetic_results.py)
5. Plot zarr-based epicurves with lockdown highlighting (plot_zarr_epicurves.py)

Usage:
    uv run python/run_synthetic_pipeline.py [--dataset DATASET] [--n-profiles N] [--batch-size N] [--spike-threshold X] [--clean] [--skip-sim] [--skip-process] [--skip-plot] [--skip-zarr-plot] [--sparsity-mode MODE] [--sparsity-tiers TIERS]

Arguments:
    --dataset: Dataset to use (default: catalonia, options: catalonia, mitma)
    --n-profiles: Number of epidemiological profiles to generate (default: 15)
    --batch-size: Number of profiles to process in each batch during Phase 1 (default: 5)
    --spike-threshold: Spike detection threshold percentile (default: 0.1 = 10th percentile)
    --sparsity-mode: Sparsity distribution mode (default: tiers, options: uniform, tiers)
    --sparsity-tiers: Sparsity levels for tier assignment (default: 0.05 0.20 0.40 0.60 0.80)
    --clean: Clean output folder before running
    --skip-sim: Skip simulation stages (use existing runs)
    --skip-process: Skip processing stages (use existing zarr)
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


def clean_run_folders(directory=None):
    """Remove run folders but keep the Zarr output.

    Args:
        directory: Directory to clean (defaults to global OUTPUT_FOLDER)
    """
    global OUTPUT_FOLDER
    target_dir = directory if directory is not None else OUTPUT_FOLDER
    logger.info(f"Cleaning run folders in {target_dir}")
    for item in target_dir.iterdir():
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


def run_two_phase_pipeline(n_profiles=15, batch_size=5, spike_threshold=0.1, spike_method="percentile",
                           growth_factor_threshold=1.5, min_growth_duration=3, min_cases_per_capita=1e-4,
                           skip_sim=False, skip_process=False, edar_edges=None, failure_tolerance=10,
                           sparsity_mode="tiers", sparsity_tiers=None, sparsity_seed=42):
    """Execute the two-phase synthetic data generation pipeline with batching support.

    Phase 1: Generate and run all baseline scenarios (no interventions), processed in batches
    Phase 2: Analyze baseline outputs to detect infection spikes, then generate
             intervention scenarios with realistic timing based on observed spikes

    Args:
        n_profiles: Number of epidemiological profiles to generate
        batch_size: Number of profiles to process in each batch during Phase 1
        spike_threshold: Percentile threshold for spike detection (default: 0.1 = 10th percentile)
        spike_method: Spike detection method (default: "percentile")
        growth_factor_threshold: Growth factor threshold for growth_rate method (default: 1.5)
        min_growth_duration: Minimum consecutive days of growth for growth_rate (default: 3)
        min_cases_per_capita: Minimum cases per person for growth_rate (default: 1e-4)
        skip_sim: Skip simulation stages (use existing runs)
        skip_process: Skip processing stages (use existing zarr)
        edar_edges: Path to EDAR-municipality edges file
        failure_tolerance: Number of consecutive failures before aborting
        sparsity_mode: Sparsity distribution mode for post-processing (default: "tiers")
        sparsity_tiers: Sparsity levels for tier assignment (default: [0.05, 0.20, 0.40, 0.60, 0.80])
        sparsity_seed: Seed for deterministic tier assignment (default: 42)
    """
    global OUTPUT_FOLDER, METAPOP_CSV, DATA_FOLDER, CONFIG_PATH

    # Use the OUTPUT_FOLDER that was set in main() based on dataset
    output_base = OUTPUT_FOLDER
    baseline_dir = output_base / "baselines"
    intervention_dir = output_base / "interventions"

    # Save original OUTPUT_FOLDER for restoration
    original_output_folder = OUTPUT_FOLDER

    # Calculate batches for Phase 1
    total_batches = (n_profiles + batch_size - 1) // batch_size

    # Phase 1: Generate baselines (with batching)
    print("=" * 60)
    print(f"PHASE 1: Generating baseline scenarios ({n_profiles} profiles in {total_batches} batches)...")
    print("=" * 60)

    if not skip_sim:
        # Clean previous baselines if starting fresh
        if baseline_dir.exists() and not skip_process:
            logger.info(f"Cleaning baseline directory: {baseline_dir}")
            shutil.rmtree(baseline_dir)
        baseline_dir.mkdir(parents=True, exist_ok=True)

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_profiles)

            logger.info(
                f"=== Processing Baseline Batch {batch_idx + 1}/{total_batches} (Profiles {start_idx}-{end_idx}) ==="
            )

            # Clean run folders from previous batch
            clean_run_folders(baseline_dir)

            cmd = [
                sys.executable,
                "python/synthetic_generator.py",
                "--n-profiles", str(n_profiles),
                "--start-index", str(start_idx),
                "--end-index", str(end_idx),
                "--output-folder", str(baseline_dir),
                "--data-folder", str(DATA_FOLDER),
                "--config", str(CONFIG_PATH),
                "--baseline-only",
                "--failure-tolerance", str(failure_tolerance),
            ]

            run_stage(f"Generate Baselines (Batch {batch_idx + 1}/{total_batches})", cmd)

    # Process baseline outputs
    if not skip_process:
        OUTPUT_FOLDER = baseline_dir  # Update global for clean_run_folders
        baseline_zarr = output_base / "raw_synthetic_observations.zarr"

        cmd = [
            sys.executable,
            "python/process_synthetic_outputs.py",
            "--runs-dir", str(baseline_dir),
            "--metapop-csv", str(METAPOP_CSV),
            "--output", str(baseline_zarr),
            "--baseline-only",
            "--sparsity-mode", sparsity_mode,
            "--sparsity-seed", str(sparsity_seed),
        ]

        if sparsity_tiers:
            cmd.extend(["--sparsity-tiers"] + [str(t) for t in sparsity_tiers])

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
            "--spike-method", str(spike_method),
            "--growth-factor-threshold", str(growth_factor_threshold),
            "--min-growth-duration", str(min_growth_duration),
            "--min-cases-per-capita", str(min_cases_per_capita),
            "--output-folder", str(intervention_dir),
            "--data-folder", str(DATA_FOLDER),
            "--config", str(CONFIG_PATH),
            "--failure-tolerance", str(failure_tolerance),
        ]

        run_stage("Generate Spike-Based Interventions", cmd)

    # Process intervention outputs and append to baseline zarr
    if not skip_process:
        OUTPUT_FOLDER = intervention_dir  # Update global for clean_run_folders
        baseline_zarr = output_base / "raw_synthetic_observations.zarr"

        cmd = [
            sys.executable,
            "python/process_synthetic_outputs.py",
            "--runs-dir", str(intervention_dir),
            "--metapop-csv", str(METAPOP_CSV),
            "--output", str(baseline_zarr),
            "--append",
            "--init",
            "--sparsity-mode", sparsity_mode,
            "--sparsity-seed", str(sparsity_seed),
        ]

        if sparsity_tiers:
            cmd.extend(["--sparsity-tiers"] + [str(t) for t in sparsity_tiers])

        if edar_edges:
            cmd.extend(["--edar-edges", str(edar_edges)])

        run_stage("Process Interventions (Append)", cmd)

    # Clean up run folders after Phase 2 processing
    if not skip_sim:
        clean_run_folders(intervention_dir)
        # Also clean up baseline run folders now that intervention generation is complete
        clean_run_folders(baseline_dir)

    # Restore original OUTPUT_FOLDER
    OUTPUT_FOLDER = original_output_folder

    print("\n" + "=" * 60)
    print("Two-Phase Pipeline Complete!")
    print(f"Final output: {output_base / 'raw_synthetic_observations.zarr'}")
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
        help="Number of profiles to process in each batch during Phase 1 baseline generation (default: 5)",
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
        "--spike-threshold",
        type=float,
        default=0.1,
        help="Spike detection threshold percentile for two-phase pipeline (default: 0.1 = 10th percentile)",
    )
    parser.add_argument(
        "--spike-method",
        type=str,
        default="percentile",
        choices=["percentile", "prominence", "growth_rate"],
        help="Spike detection method for two-phase pipeline (default: percentile)",
    )
    parser.add_argument(
        "--growth-factor-threshold",
        type=float,
        default=1.5,
        help="Growth factor threshold for growth_rate method (default: 1.5 = 50%% growth)",
    )
    parser.add_argument(
        "--min-growth-duration",
        type=int,
        default=3,
        help="Minimum consecutive days of growth for growth_rate method (default: 3)",
    )
    parser.add_argument(
        "--min-cases-per-capita",
        type=float,
        default=1e-4,
        help="Minimum cases per person for growth_rate method (default: 1e-4 = 1 per 10K)",
    )
    # Sparsity configuration for curriculum learning
    parser.add_argument(
        "--sparsity-mode",
        choices=["uniform", "tiers"],
        default="tiers",
        help="Sparsity distribution mode for post-processing (default: tiers)",
    )
    parser.add_argument(
        "--sparsity-tiers",
        type=float,
        nargs="+",
        default=[0.05, 0.20, 0.40, 0.60, 0.80],
        help="Sparsity levels for tier assignment (default: 0.05 0.20 0.40 0.60 0.80)",
    )
    parser.add_argument(
        "--sparsity-seed",
        type=int,
        default=42,
        help="Seed for deterministic tier assignment (default: 42)",
    )

    args = parser.parse_args()

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
        logger.info("DRY RUN - Two-Phase Pipeline:")
        logger.info(f"  Total Profiles: {args.n_profiles}")
        logger.info(f"  Batch Size: {args.batch_size}")
        logger.info(f"  Spike Threshold: {args.spike_threshold}")
        logger.info(f"  Batches: {(args.n_profiles + args.batch_size - 1) // args.batch_size}")
        return

    # Execute Two-Phase Pipeline
    try:
        run_two_phase_pipeline(
            n_profiles=args.n_profiles,
            batch_size=args.batch_size,
            spike_threshold=args.spike_threshold,
            spike_method=args.spike_method,
            growth_factor_threshold=args.growth_factor_threshold,
            min_growth_duration=args.min_growth_duration,
            min_cases_per_capita=args.min_cases_per_capita,
            skip_sim=args.skip_sim,
            skip_process=args.skip_process,
            edar_edges=args.edar_edges,
            failure_tolerance=args.failure_tolerance,
            sparsity_mode=args.sparsity_mode,
            sparsity_tiers=args.sparsity_tiers,
            sparsity_seed=args.sparsity_seed,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Two-phase pipeline failed with exit code {e.returncode}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Two-phase pipeline failed with exception: {e}")
        sys.exit(1)

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
