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

Environment variables:
    JULIA_PROJECT: Path to EpiSim.jl project (default: ../)
    EPISIM_EXECUTABLE_PATH: Optional path to compiled episim executable
"""

import argparse
import json
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


def check_baseline_success(baseline_dir, n_profiles_requested):
    """Check if enough baselines succeeded to proceed to Phase 2.

    Args:
        baseline_dir: Path to baseline directory containing BATCH_RESULTS.json
        n_profiles_requested: Number of profiles originally requested

    Returns:
        tuple: (success_count, success_rate, should_proceed, failed_profiles)
            - success_count: Number of successful runs
            - success_rate: Ratio of successful runs (0.0 to 1.0)
            - should_proceed: True if sufficient baselines succeeded
            - failed_profiles: List of failed profile IDs (empty list if none failed)
    """
    batch_results_path = Path(baseline_dir) / "BATCH_RESULTS.json"

    if not batch_results_path.exists():
        logger.error(f"BATCH_RESULTS.json not found at {batch_results_path}")
        logger.error("Cannot determine baseline success rate - aborting Phase 2")
        return 0, 0.0, False, None

    try:
        with open(batch_results_path, "r") as f:
            results = json.load(f)

        total = results.get("total", 0)
        succeeded = results.get("succeeded", 0)
        failed = results.get("failed", 0)
        skipped = results.get("skipped", 0)
        failures = results.get("failures", [])

        def extract_profile_id(failure_entry):
            """Extract profile ID from known failure entry formats."""
            candidates = []
            if isinstance(failure_entry, str):
                candidates.append(failure_entry)
            elif isinstance(failure_entry, dict):
                # Allow for potential future structured schemas.
                for key in ("run_id", "instance_folder", "name", "id"):
                    value = failure_entry.get(key)
                    if isinstance(value, str):
                        candidates.append(value)

            for candidate in candidates:
                run_name = Path(candidate).name
                if run_name.startswith("run_"):
                    run_name = run_name[4:]
                parts = run_name.split("_")
                if parts and parts[0].isdigit():
                    return int(parts[0])
            return None

        # Extract failed profile IDs from failure list
        # Formats seen:
        #   - "run_{pid}_Baseline"
        #   - "{pid}_Baseline"
        #   - "run_{pid}_{Scenario}_s{strength}"
        failed_profiles = set()
        for failure in failures:
            profile_id = extract_profile_id(failure)
            if profile_id is not None:
                failed_profiles.add(profile_id)

        # Fallback: recover profile IDs directly from run folders with ERROR.json
        # when BATCH_RESULTS has failures but an unexpected schema.
        if failed > 0 and not failed_profiles:
            logger.warning(
                "BATCH_RESULTS reported failures but no failed profile IDs were parsed. "
                "Falling back to ERROR.json scan."
            )
            baseline_dir_path = Path(baseline_dir)
            for run_dir in baseline_dir_path.glob("run_*"):
                if (run_dir / "ERROR.json").exists():
                    profile_id = extract_profile_id(run_dir.name)
                    if profile_id is not None:
                        failed_profiles.add(profile_id)

            if failed_profiles:
                logger.warning(
                    f"Recovered failed profile IDs from run folders: {sorted(failed_profiles)}"
                )
            else:
                logger.error(
                    "Could not determine failed profile IDs despite non-zero failures. "
                    "Retry cannot proceed safely."
                )
                return 0, 0.0, False, None

        # Calculate effective successes (completed runs = succeeded + skipped)
        # Skipped runs are already complete (have observables.nc), so count as successful
        effective_successes = succeeded + skipped

        # Calculate success rate based on attempted runs (excluding skipped)
        non_skipped = total - skipped
        if non_skipped > 0:
            success_rate = succeeded / non_skipped
        else:
            success_rate = 0.0

        logger.info("=" * 60)
        logger.info("Phase 1 Baseline Results Summary")
        logger.info("=" * 60)
        logger.info(f"  Total runs: {total}")
        logger.info(f"  Succeeded (new): {succeeded}")
        logger.info(f"  Skipped (existing): {skipped}")
        logger.info(f"  Effective successes: {effective_successes}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Success rate: {success_rate:.1%}")
        logger.info(f"  Requested profiles: {n_profiles_requested}")
        if failed_profiles:
            logger.info(f"  Failed profile IDs: {sorted(failed_profiles)}")

        # We need at least the requested number of effective successful baselines
        if effective_successes >= n_profiles_requested:
            logger.info(
                f"✓ Sufficient baselines completed ({effective_successes}/{n_profiles_requested})"
            )
            return effective_successes, success_rate, True, []
        else:
            logger.warning(
                f"⚠ Insufficient baselines completed ({effective_successes}/{n_profiles_requested})"
            )
            logger.warning(
                f"  Will retry {len(failed_profiles)} failed profiles with new seeds"
            )
            return effective_successes, success_rate, False, sorted(failed_profiles)

    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error parsing BATCH_RESULTS.json: {e}")
        return 0, 0.0, False, None


def clean_output():
    """Remove previous run outputs."""
    global OUTPUT_FOLDER
    if OUTPUT_FOLDER.exists():
        logger.info(f"Cleaning output folder: {OUTPUT_FOLDER}")
        shutil.rmtree(OUTPUT_FOLDER)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created clean output folder: {OUTPUT_FOLDER}")


def run_baseline_batch(
    start_idx,
    end_idx,
    n_profiles,
    baseline_dir,
    no_retry=True,
    failure_tolerance=10,
    intervention_profile_fraction=1.0,
    intervention_seed=42,
    mobility_sigma_min=0.0,
    mobility_sigma_max=0.6,
    n_jobs=-1,
):
    """Run a single baseline batch (used for initial run and retries).

    Args:
        start_idx: Start profile index (inclusive)
        end_idx: End profile index (exclusive)
        n_profiles: Total number of profiles (for LHS generation)
        baseline_dir: Output directory for baselines
        no_retry: If True, run once without retry (pipeline handles global retry)
        failure_tolerance: Number of failures before aborting
        intervention_profile_fraction: Fraction of profiles to get interventions
        intervention_seed: Seed for profile sampling
        mobility_sigma_min: Min mobility sigma
        mobility_sigma_max: Max mobility sigma
        n_jobs: Number of parallel workers

    Returns:
        bool: True if batch command succeeded, False otherwise
    """
    cmd = [
        sys.executable,
        "python/synthetic_generator.py",
        "--n-profiles",
        str(n_profiles),
        "--start-index",
        str(start_idx),
        "--end-index",
        str(end_idx),
        "--output-folder",
        str(baseline_dir),
        "--data-folder",
        str(DATA_FOLDER),
        "--config",
        str(CONFIG_PATH),
        "--baseline-only",
        "--failure-tolerance",
        str(failure_tolerance),
        "--intervention-profile-fraction",
        str(intervention_profile_fraction),
        "--intervention-seed",
        str(intervention_seed),
        "--mobility-sigma-min",
        str(mobility_sigma_min),
        "--mobility-sigma-max",
        str(mobility_sigma_max),
        "--n-jobs",
        str(n_jobs),
    ]

    if no_retry:
        cmd.append("--no-retry")

    try:
        run_stage(f"Generate Baselines (Profiles {start_idx}-{end_idx})", cmd)
        return True
    except subprocess.CalledProcessError:
        logger.warning(
            f"Batch {start_idx}-{end_idx} had failures (will retry if needed)"
        )
        return False


def plot_results():
    """Stage 3: Plot results."""
    # Use absolute path since script changes directory
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "python" / "plot_synthetic_results.py"),
        "--runs-dir",
        str(OUTPUT_FOLDER),
        "--output-dir",
        str(OUTPUT_FOLDER),
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


def run_two_phase_pipeline(
    n_profiles=15,
    batch_size=5,
    spike_threshold=0.1,
    spike_method="percentile",
    growth_factor_threshold=1.5,
    min_growth_duration=3,
    min_cases_per_capita=1e-4,
    skip_sim=False,
    skip_process=False,
    edar_edges=None,
    failure_tolerance=10,
    sparsity_mode="tiers",
    sparsity_tiers=None,
    sparsity_seed=42,
    max_intervention_duration=90,
    intervention_profile_fraction=1.0,
    intervention_seed=42,
    mobility_sigma_min=0.0,
    mobility_sigma_max=0.6,
    n_jobs=-1,
    nvme_base=None,
    include_latents=False,
):
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
        max_intervention_duration: Maximum intervention duration in days (default: 90)
        intervention_profile_fraction: Fraction of profiles receiving full intervention sweep (default: 1.0)
        intervention_seed: Random seed for profile sampling (default: 42)
        n_jobs: Number of parallel workers (-1 = auto)
        nvme_base: Base directory for NVMe staging (if provided, work happens on NVMe and results rsynced)
    """
    global OUTPUT_FOLDER, METAPOP_CSV, DATA_FOLDER, CONFIG_PATH

    # Use the OUTPUT_FOLDER that was set in main() based on dataset
    output_base = OUTPUT_FOLDER

    # Handle NVMe staging if provided
    if nvme_base:
        nvme_path = Path(nvme_base)
        baseline_dir = nvme_path / "baselines"
        intervention_dir = nvme_path / "interventions"
        logger.info(f"Using NVMe staging at: {nvme_path}")
        logger.info(f"  Baseline work dir: {baseline_dir}")
        logger.info(f"  Final output dir: {output_base / 'baselines'}")
    else:
        baseline_dir = output_base / "baselines"
        intervention_dir = output_base / "interventions"

    # Save original OUTPUT_FOLDER for restoration
    original_output_folder = OUTPUT_FOLDER

    # Calculate batches for Phase 1
    total_batches = (n_profiles + batch_size - 1) // batch_size

    # Phase 1: Generate baselines (with batching)
    print("=" * 60)
    print(
        f"PHASE 1: Generating baseline scenarios ({n_profiles} profiles in {total_batches} batches)..."
    )
    print("=" * 60)

    if not skip_sim:
        # Clean previous baselines if starting fresh
        if baseline_dir.exists() and not skip_process:
            logger.info(f"Cleaning baseline directory: {baseline_dir}")
            shutil.rmtree(baseline_dir)
        baseline_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1 with global retry: run all batches first, then retry failed profiles
        max_global_retries = 10
        global_retry_count = 0
        all_profiles_succeeded = False
        failed_profiles = []

        while not all_profiles_succeeded and global_retry_count < max_global_retries:
            global_retry_count += 1

            if global_retry_count == 1:
                # Initial run: process all profiles in batches
                logger.info(f"\n{'=' * 60}")
                logger.info(
                    f"PHASE 1 INITIAL RUN: Processing all {n_profiles} profiles"
                )
                logger.info(f"{'=' * 60}")

                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, n_profiles)

                    logger.info(
                        f"\n=== Processing Baseline Batch {batch_idx + 1}/{total_batches} "
                        f"(Profiles {start_idx}-{end_idx}) ==="
                    )

                    run_baseline_batch(
                        start_idx=start_idx,
                        end_idx=end_idx,
                        n_profiles=n_profiles,
                        baseline_dir=baseline_dir,
                        no_retry=True,  # Pipeline handles global retry
                        failure_tolerance=failure_tolerance,
                        intervention_profile_fraction=intervention_profile_fraction,
                        intervention_seed=intervention_seed,
                        mobility_sigma_min=mobility_sigma_min,
                        mobility_sigma_max=mobility_sigma_max,
                        n_jobs=n_jobs,
                    )

            else:
                # Retry run: process only failed profiles
                logger.info(f"\n{'=' * 60}")
                logger.info(
                    f"GLOBAL RETRY {global_retry_count - 1}/{max_global_retries - 1}"
                )
                logger.info(
                    f"Retrying {len(failed_profiles)} failed profiles with new seeds"
                )
                logger.info(f"{'=' * 60}")

                # Generate new seed for this retry iteration
                retry_seed = 42 + (global_retry_count - 1) * 1000

                for profile_id in failed_profiles:
                    logger.info(f"Retrying profile {profile_id} with seed {retry_seed}")
                    run_baseline_batch(
                        start_idx=profile_id,
                        end_idx=profile_id + 1,
                        n_profiles=n_profiles,
                        baseline_dir=baseline_dir,
                        no_retry=True,
                        failure_tolerance=failure_tolerance,
                        intervention_profile_fraction=intervention_profile_fraction,
                        intervention_seed=retry_seed,  # Use new seed for retry
                        mobility_sigma_min=mobility_sigma_min,
                        mobility_sigma_max=mobility_sigma_max,
                        n_jobs=n_jobs,
                    )

            # Check results after this iteration
            logger.info(f"\n{'=' * 60}")
            logger.info(f"CHECKING RESULTS (Iteration {global_retry_count})")
            logger.info(f"{'=' * 60}")

            # Always check from the working directory (NVMe if used, otherwise output)
            # BATCH_RESULTS.json is written here by Julia
            success_count, success_rate, all_profiles_succeeded, failed_profiles = (
                check_baseline_success(baseline_dir, n_profiles)
            )

            if failed_profiles is None:
                logger.error(
                    f"\n✗ Critical error: BATCH_RESULTS.json missing or invalid."
                )
                logger.error(f"  Cannot determine which profiles failed to retry.")
                break
            elif all_profiles_succeeded:
                logger.info(f"\n✓ ALL {n_profiles} PROFILES SUCCEEDED!")
                logger.info(f"  Completed in {global_retry_count} iteration(s)")
                logger.info(f"  Success rate: {success_rate:.1%}")
                break
            elif global_retry_count < max_global_retries:
                logger.warning(f"\n⚠ {len(failed_profiles)} profiles still failing")
                logger.warning(f"  Will retry in next iteration")
            else:
                logger.error(f"\n✗ MAX RETRIES ({max_global_retries}) EXHAUSTED")
                logger.error(f"  Only {success_count}/{n_profiles} profiles succeeded")
                logger.error(f"  Failed profiles: {failed_profiles}")

        if not all_profiles_succeeded:
            raise RuntimeError(
                f"Failed to generate sufficient baselines: {success_count}/{n_profiles} "
                f"({success_rate:.1%}) after {max_global_retries} retry iterations"
            )

        # Phase 1 complete - sync to GPFS if using NVMe staging
        if nvme_base:
            final_baseline_dir = output_base / "baselines"
            final_baseline_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Phase 1 Complete - Syncing from NVMe to GPFS...")
            logger.info(f"  Source: {baseline_dir}")
            logger.info(f"  Destination: {final_baseline_dir}")
            logger.info(f"{'=' * 60}")
            for run_dir in baseline_dir.glob("run_*"):
                if run_dir.is_dir():
                    run_name = run_dir.name
                    dest_dir = final_baseline_dir / run_name
                    logger.info(f"  Syncing {run_name}...")
                    rsync_cmd = [
                        "rsync",
                        "-av",
                        f"{run_dir}/",
                        f"{dest_dir}/",
                    ]
                    subprocess.run(rsync_cmd, check=False)

            # Also sync BATCH_RESULTS.json
            batch_results_src = baseline_dir / "BATCH_RESULTS.json"
            batch_results_dst = final_baseline_dir / "BATCH_RESULTS.json"
            if batch_results_src.exists():
                logger.info(f"  Syncing BATCH_RESULTS.json...")
                shutil.copy2(batch_results_src, batch_results_dst)

            logger.info(f"✓ Sync complete")

    # Process baseline outputs
    if not skip_process:
        # After Phase 1, use the appropriate directory for processing
        # If NVMe was used, we've synced to GPFS and should process from there
        # Otherwise, process from the working directory
        if nvme_base:
            process_baseline_dir = output_base / "baselines"
            logger.info(f"Processing baselines from GPFS: {process_baseline_dir}")
        else:
            process_baseline_dir = baseline_dir
            logger.info(f"Processing baselines from: {process_baseline_dir}")

        OUTPUT_FOLDER = process_baseline_dir  # Update global for clean_run_folders
        baseline_zarr = output_base / "raw_synthetic_observations.zarr"

        cmd = [
            sys.executable,
            "python/process_synthetic_outputs.py",
            "--runs-dir",
            str(process_baseline_dir),
            "--metapop-csv",
            str(METAPOP_CSV),
            "--output",
            str(baseline_zarr),
            "--baseline-only",
            "--sparsity-mode",
            sparsity_mode,
            "--sparsity-seed",
            str(sparsity_seed),
            "--n-jobs",
            str(n_jobs),
        ]

        if sparsity_tiers:
            cmd.extend(["--sparsity-tiers"] + [str(t) for t in sparsity_tiers])

        if edar_edges:
            cmd.extend(["--edar-edges", str(edar_edges)])
        if include_latents:
            cmd.append("--include-latents")

        run_stage("Process Baselines", cmd)

    # Phase 2: Generate spike-based interventions
    print("\n" + "=" * 60)
    print("PHASE 2: Generating spike-based intervention scenarios...")
    print("=" * 60)

    # Check if Phase 1 succeeded sufficiently before proceeding
    if not skip_sim:
        # Determine which baseline directory to check
        if nvme_base:
            baseline_check_dir = output_base / "baselines"
        else:
            baseline_check_dir = baseline_dir

        success_count, success_rate, should_proceed, _ = check_baseline_success(
            baseline_check_dir, n_profiles
        )

        if not should_proceed:
            logger.error("=" * 60)
            logger.error(
                "Phase 1 baseline generation failed to produce sufficient successful runs."
            )
            logger.error(f"Required: {n_profiles} successful baselines")
            logger.error(
                f"Achieved: {success_count} successful baselines ({success_rate:.1%})"
            )
            logger.error(
                "Aborting pipeline - Phase 2 cannot proceed without sufficient baseline data."
            )
            logger.error("=" * 60)
            raise RuntimeError(
                f"Insufficient baseline successes: {success_count}/{n_profiles} "
                f"({success_rate:.1%}). Pipeline aborted."
            )
        # Clean previous interventions if starting fresh
        if intervention_dir.exists() and not skip_process:
            logger.info(f"Cleaning intervention directory: {intervention_dir}")
            shutil.rmtree(intervention_dir)
        intervention_dir.mkdir(parents=True, exist_ok=True)
        # Marker is now created by synthetic_generator.py after sampling
        # This prevents false positives when intervention_profile_fraction = 0

        # When using NVMe, baseline_dir was cleared - use final GPFS path for intervention generation
        if nvme_base:
            baseline_reference_dir = output_base / "baselines"
            logger.info(
                f"Using GPFS baseline dir for intervention generation: {baseline_reference_dir}"
            )
        else:
            baseline_reference_dir = baseline_dir

        cmd = [
            sys.executable,
            "python/synthetic_generator.py",
            "--intervention-only",
            str(baseline_reference_dir),
            "--spike-threshold",
            str(spike_threshold),
            "--spike-method",
            str(spike_method),
            "--growth-factor-threshold",
            str(growth_factor_threshold),
            "--min-growth-duration",
            str(min_growth_duration),
            "--min-cases-per-capita",
            str(min_cases_per_capita),
            "--max-intervention-duration",
            str(max_intervention_duration),
            "--output-folder",
            str(intervention_dir),
            "--data-folder",
            str(DATA_FOLDER),
            "--config",
            str(CONFIG_PATH),
            "--failure-tolerance",
            str(failure_tolerance),
            "--n-profiles",
            str(n_profiles),
            "--intervention-profile-fraction",
            str(intervention_profile_fraction),
            "--intervention-seed",
            str(intervention_seed),
            "--mobility-sigma-min",
            str(mobility_sigma_min),
            "--mobility-sigma-max",
            str(mobility_sigma_max),
            "--n-jobs",
            str(n_jobs),
        ]

        run_stage("Generate Spike-Based Interventions", cmd)

    # Process intervention outputs and append to baseline zarr
    # Check if any interventions were generated
    has_interventions = False
    if intervention_dir.exists():
        # Check for actual run directories OR the pending marker
        marker_file = intervention_dir / ".interventions_pending"
        has_interventions = marker_file.exists() or any(
            d.name.startswith("run_") for d in intervention_dir.iterdir() if d.is_dir()
        )

    if not has_interventions:
        logger.info("No intervention scenarios generated. Skipping processing step.")
    elif not skip_process:
        OUTPUT_FOLDER = intervention_dir  # Update global for clean_run_folders
        baseline_zarr = output_base / "raw_synthetic_observations.zarr"

        cmd = [
            sys.executable,
            "python/process_synthetic_outputs.py",
            "--runs-dir",
            str(intervention_dir),
            "--metapop-csv",
            str(METAPOP_CSV),
            "--output",
            str(baseline_zarr),
            "--append",
            "--init",
            "--sparsity-mode",
            sparsity_mode,
            "--sparsity-seed",
            str(sparsity_seed),
            "--n-jobs",
            str(n_jobs),
        ]

        if sparsity_tiers:
            cmd.extend(["--sparsity-tiers"] + [str(t) for t in sparsity_tiers])

        if edar_edges:
            cmd.extend(["--edar-edges", str(edar_edges)])
        if include_latents:
            cmd.append("--include-latents")

        run_stage("Process Interventions (Append)", cmd)

        # Remove marker file after processing
        marker_file = intervention_dir / ".interventions_pending"
        if marker_file.exists():
            marker_file.unlink()

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
        help="Clean output folder before running (including run folders)",
    )
    parser.add_argument(
        "--preserve",
        action="store_true",
        help="Preserve existing zarr output (default: clean zarr to prevent duplicate run_ids)",
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
        default=10,
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
    parser.add_argument(
        "--max-intervention-duration",
        type=int,
        default=90,
        help="Maximum intervention duration in days for spike-based interventions (default: 90)",
    )
    parser.add_argument(
        "--intervention-profile-fraction",
        type=float,
        default=0,
        help="Fraction of profiles receiving full intervention sweep (default: 0.0 = none). "
        "All profiles generate baselines. Selected profiles get 6 intervention strengths.",
    )
    parser.add_argument(
        "--intervention-seed",
        type=int,
        default=42,
        help="Random seed for profile sampling (default: 42).",
    )
    parser.add_argument(
        "--mobility-sigma-min",
        type=float,
        default=0.0,
        help="Minimum mobility sigma for origin/destination noise (default: 0.0).",
    )
    parser.add_argument(
        "--mobility-sigma-max",
        type=float,
        default=0.6,
        help="Maximum mobility sigma for origin/destination noise (default: 0.6).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel Python workers for generation and processing. Julia simulations always run single-threaded to avoid NetCDF/HDF5 thread-safety issues. (-1 = auto, default: -1)",
    )
    parser.add_argument(
        "--nvme-base",
        type=str,
        default=None,
        help="Base directory for NVMe staging (e.g., /tmp/synthetic_pipeline). If provided, all processing happens on NVMe and results are rsynced to output folder.",
    )
    parser.add_argument(
        "--include-latents",
        action="store_true",
        help="Export latent simulator states into the synthetic zarr for hybrid supervision.",
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

    # Clean zarr output by default to prevent duplicate run_ids
    zarr_path = OUTPUT_FOLDER / "raw_synthetic_observations.zarr"
    if zarr_path.exists():
        if args.preserve:
            logger.warning(f"Preserving existing zarr: {zarr_path}")
            logger.warning(
                "Duplicate run_ids may occur if re-running with same profiles."
            )
        else:
            logger.info(f"Removing existing zarr output: {zarr_path}")
            shutil.rmtree(zarr_path)

    if args.dry_run:
        logger.info("DRY RUN - Two-Phase Pipeline:")
        logger.info(f"  Total Profiles: {args.n_profiles}")
        logger.info(f"  Batch Size: {args.batch_size}")
        logger.info(f"  Spike Threshold: {args.spike_threshold}")
        logger.info(
            f"  Mobility Sigma Range: [{args.mobility_sigma_min}, {args.mobility_sigma_max}]"
        )
        logger.info(
            f"  Batches: {(args.n_profiles + args.batch_size - 1) // args.batch_size}"
        )
        logger.info(f"  n_jobs: {args.n_jobs}")
        logger.info(f"  NVMe Base: {args.nvme_base or 'None (GPFS direct)'}")
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
            max_intervention_duration=args.max_intervention_duration,
            intervention_profile_fraction=args.intervention_profile_fraction,
            intervention_seed=args.intervention_seed,
            mobility_sigma_min=args.mobility_sigma_min,
            mobility_sigma_max=args.mobility_sigma_max,
            n_jobs=args.n_jobs,
            nvme_base=args.nvme_base,
            include_latents=args.include_latents,
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
