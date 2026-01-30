import json
import logging
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import qmc

# Ensure we can import episim_python
sys.path.append(os.path.dirname(__file__))

from episim_python.episim_utils import EpiSimConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SyntheticGenerator")


class SyntheticDataGenerator:
    def __init__(self, base_config_path, data_folder, output_folder):
        self.base_config_path = base_config_path
        self.data_folder = data_folder
        self.output_folder = output_folder

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Load base configuration
        self.base_config = EpiSimConfig.from_json(base_config_path)

        # Get data file paths from config
        metapop_filename = self.base_config.get_param("data.metapopulation_data_filename")
        mobility_filename = self.base_config.get_param("data.mobility_matrix_filename")

        # Load Metapopulation Data for seeding (ensure ID is read as string)
        self.metapop_df = pd.read_csv(
            os.path.join(data_folder, metapop_filename),
            dtype={'id': str}
        )

        # Load Mobility Matrix
        self.mobility_df = pd.read_csv(
            os.path.join(data_folder, mobility_filename)
        )

        # Load rosetta data for correct index mapping
        rosetta_filename = metapop_filename.replace("metapopulation_data", "rosetta")
        try:
            rosetta_path = os.path.join(data_folder, rosetta_filename)
            self.rosetta_df = pd.read_csv(rosetta_path, dtype={'id': str})
        except FileNotFoundError:
            # Fallback to default rosetta.csv
            self.rosetta_df = pd.read_csv(
                os.path.join(data_folder, "rosetta.csv"),
                dtype={'id': str}
            )

    def generate_parameter_grid(self, n_profiles=5, seed=42):
        """
        Generate Latin Hypercube Samples for Epidemiological Profiles:
        0: R0_scale (scale_β) [0.5, 3.0]
        1: T_inf (Infectious Period) [2.0, 10.0] -> μᵍ = 1/T (recovery rate)
        2: T_inc (Incubation Period) [2.0, 10.0] -> ηᵍ = 1/T (progression rate)
        3: Reaction Delay [0, 30] (Replaces absolute Event Start)
        4: Event Duration [7, 60]
        5: Affected Fraction [0.1, 0.6] (For Local Scenarios)
        6: Ratio Beta A (ratio_beta_a) [0.1, 1.0]
        7: Alpha Scale (alpha_scale) [0.5, 1.5]
        8: Mu Scale (mu_scale) [0.5, 1.5] (unused, kept for backward compatibility)
        9: Seed Size [10, 500] (Log-uniform sampled)

        NOTE: We enforce T_inf <= T_inc to ensure μᵍ >= ηᵍ for model stability.
        """
        sampler = qmc.LatinHypercube(d=10, seed=seed)
        sample = sampler.random(n=n_profiles)

        # Scale samples
        l_bounds = [0.5, 2.0, 2.0, 0.0, 7.0, 0.1, 0.1, 0.5, 0.5, 10.0]
        u_bounds = [3.0, 10.0, 10.0, 30.0, 60.0, 0.6, 1.0, 1.5, 1.5, 500.0]

        scaled = qmc.scale(sample, l_bounds, u_bounds)

        profiles = []
        detection_threshold = 100.0  # Assumed detected cases

        for i, row in enumerate(scaled):
            (
                r0,
                t_inf,
                t_inc,
                delay,
                duration,
                fraction,
                ratio_beta_a,
                alpha_scale,
                mu_scale,
                seed,
            ) = row

            # Enforce T_inf <= T_inc to ensure μᵍ >= ηᵍ for model stability
            # If T_inf > T_inc, swap the values
            if t_inf > t_inc:
                t_inf, t_inc = t_inc, t_inf

            # Heuristic for Event Start
            # Time to detection ~ T_inf * ln(Threshold/Seed) / (R0 - 1)
            # If R0 <= 1, growth is negative or flat. Set a default late start or based on delay.
            if r0 > 1.05:
                # Basic SIR growth approximation
                # Doubling time Td = T_inf * ln(2) / (R0 - 1)
                # Time to grow from Seed to Threshold
                if seed < detection_threshold:
                    growth_rate = (r0 - 1.0) / t_inf
                    time_to_detect = np.log(detection_threshold / seed) / growth_rate
                else:
                    time_to_detect = 0.0
            else:
                # Low R0: Epidemic won't grow fast.
                # Intervention is less critical/realistic.
                # Set a baseline delay or "late" start.
                time_to_detect = 20.0

            # event_start is detection time + reaction delay
            event_start = max(1, int(time_to_detect + delay))

            profiles.append(
                {
                    "profile_id": i,
                    "r0_scale": r0,
                    "t_inf": t_inf,
                    "t_inc": t_inc,
                    "event_start": event_start,
                    "event_duration": int(duration),
                    "affected_fraction": fraction,
                    "ratio_beta_a": ratio_beta_a,
                    "alpha_scale": alpha_scale,
                    "mu_scale": mu_scale,
                    "seed_size": int(seed),
                }
            )

        return profiles

    def validate_profile_parameters(self, profile):
        """
        Validate a profile for model stability constraints.

        Returns:
            bool: True if profile is valid, False otherwise
        """
        # Check μᵍ >= ηᵍ (i.e., T_inf <= T_inc)
        # If recovery rate is less than progression rate, the Markov chain becomes unstable
        mu = 1.0 / profile["t_inf"]
        eta = 1.0 / profile["t_inc"]

        if mu < eta:
            logger.warning(
                f"Profile {profile['profile_id']}: μ ({mu:.3f}) < η ({eta:.3f}), "
                f"T_inf={profile['t_inf']:.1f}, T_inc={profile['t_inc']:.1f}. "
                f"Skipping - this would cause DomainError in simulation."
            )
            return False

        # Check R0 > 1 for valid epidemic growth (warn only)
        if profile["r0_scale"] < 1.0:
            logger.warning(
                f"Profile {profile['profile_id']}: R0_scale={profile['r0_scale']:.2f} < 1.0 "
                f"may not produce epidemic growth."
            )

        return True

    def prepare_kappa0_file(
        self,
        run_id,
        scenario,
        strength,
        profile,
        start_date_str,
        end_date_str,
        output_dir=None,
    ):
        """Generate kappa0.csv file for ALL scenarios.

        Args:
            output_dir: If provided, write file to this directory. Otherwise use self.output_folder.
        """
        # Parse dates
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)

        # Create date range (inclusive)
        date_range = pd.date_range(start=start_date, end=end_date)
        n_days = len(date_range)

        # Initialize dataframe with default κ₀=0.0 (no confinement, full mobility)
        df = pd.DataFrame(
            {
                "date": date_range,
                "reduction": 0.0,
                "datetime": date_range,
                "time": range(n_days),
            }
        )

        kappa_val = strength

        if scenario == "Global_Timed":
            event_start = profile["event_start"]
            event_duration = profile["event_duration"]

            # Ensure indices are within bounds
            start_idx = min(event_start, n_days)
            end_idx = min(event_start + event_duration, n_days)

            # Apply reduction during event window
            if start_idx < end_idx:
                df.loc[start_idx : end_idx - 1, "reduction"] = kappa_val
        elif scenario == "Baseline":
            df["reduction"] = 0.0

        # Format dates as string for CSV
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d")

        filename = "kappa0.csv"
        output_path = output_dir or self.output_folder
        path = os.path.join(output_path, filename)
        df.to_csv(path, index=False)
        return filename, path

    def prepare_mobility_file(
        self, run_id, scenario, reduction_strength, affected_fraction, output_dir=None
    ):
        """
        Prepare mobility matrix based on scenario.
        For Global scenarios, return original matrix (unscaled).

        Args:
            output_dir: If provided, write file to this directory. Otherwise use self.output_folder.
        """
        scaled_df = self.mobility_df.copy()

        # Global scenarios use kappa0 for mobility reduction
        # Mobility matrix remains unchanged

        filename = "mobility_matrix.csv"
        output_path = output_dir or self.output_folder
        path = os.path.join(output_path, filename)
        scaled_df.to_csv(path, index=False)
        return filename, path

    def prepare_seed_file(self, run_id, seed_size=10.0):
        """Generate random seed file"""
        # Pick a random region (row index)
        random_idx = np.random.randint(0, len(self.metapop_df))
        seed_region = self.metapop_df.iloc[random_idx]
        region_id = seed_region["id"]

        # Look up the correct 1-based index from rosetta
        idx_row = self.rosetta_df[self.rosetta_df['id'] == region_id]
        if len(idx_row) == 0:
            raise ValueError(f"Region ID {region_id} not found in rosetta")
        idx = idx_row['idx'].iloc[0]

        # Create seed dataframe
        # We start with `seed_size` Exposed/Asymptomatic individuals in Middle age group (M)
        seed_data = {
            "name": ["SeedRegion"],
            "id": [region_id],
            "idx": [idx],
            "Y": [0.0],
            "M": [float(seed_size)],
            "O": [0.0],
        }

        seed_df = pd.DataFrame(seed_data)

        filename = f"seeds_{run_id}.csv"
        path = os.path.join(self.output_folder, filename)
        seed_df.to_csv(path, index=False)
        return filename, path

    def run_single_scenario(self, pid, scen_name, strength, profile, seed_path):
        """Helper to prepare files for a single configuration."""
        r0_scale = profile["r0_scale"]
        t_inf = profile["t_inf"]
        t_inc = profile["t_inc"]
        ratio_beta_a = profile.get("ratio_beta_a", 0.5)
        alpha_scale = profile.get("alpha_scale", 1.0)
        mu_scale = profile.get("mu_scale", 1.0)

        # Construct Run ID
        # Format: {pid}_{Scenario}_s{strength_int} where strength_int is percent
        str_pct = int(round(strength * 100))
        run_id = f"{pid}_{scen_name}_s{str_pct:02d}"
        if scen_name == "Baseline":
            run_id = f"{pid}_Baseline"

        logger.info(f"Preparing {run_id} (Str={strength:.2f})")

        # Prepare Directory Structure FIRST
        # Use run_id as the folder name (no UUID nesting when using name parameter)
        model_state_folder = os.path.join(self.output_folder, f"run_{run_id}")
        if not os.path.exists(model_state_folder):
            os.makedirs(model_state_folder)

        # Update Config
        import copy

        config_dict = copy.deepcopy(self.base_config.config)
        config = EpiSimConfig(config_dict)

        # Get start/end dates
        start_date = config.get_param("simulation.start_date")
        end_date = "2020-06-01"

        gamma = 1.0 / t_inf  # Recovery rate (μᵍ)
        eta = 1.0 / t_inc    # E→I/A progression rate (ηᵍ)

        # Calculate derived params
        base_beta_I = config.get_param("epidemic_params.βᴵ")
        base_alpha = config.get_param("epidemic_params.αᵍ")

        new_beta_A = base_beta_I * ratio_beta_a
        new_alpha = [min(1.0, x * alpha_scale) for x in base_alpha]

        # Prepare Mobility - Write to run directory
        # All scenarios use base mobility matrix (interventions via κ₀ only)
        _, mob_path = self.prepare_mobility_file(
            run_id, "Global", 0.0, 0.0, output_dir=model_state_folder
        )

        # NPI Params Setup - Always create CSV for ALL scenarios
        # Write to run directory
        _, kappa0_path = self.prepare_kappa0_file(
            run_id,
            scen_name,
            strength,
            profile,
            start_date,
            end_date,
            output_dir=model_state_folder,
        )

        # Inject updates
        # NOTE: μᵍ is the recovery rate (gamma), NOT γᵍ (hospitalization probability)
        updates = {
            "simulation.end_date": end_date,
            "simulation.save_full_output": False,
            "simulation.save_observables": True,
            "epidemic_params.scale_β": r0_scale,
            "epidemic_params.βᴬ": new_beta_A,
            "epidemic_params.αᵍ": new_alpha,
            "epidemic_params.μᵍ": [gamma] * 3,  # Recovery rate = 1/T_inf
            "epidemic_params.ηᵍ": [eta] * 3,    # E→I/A progression rate = 1/T_inc
            "data.mobility_matrix_filename": mob_path,
            "data.initial_condition_filename": seed_path,
            "NPI.are_there_npi": True,
        }

        config.inject(updates)

        # Always set kappa0_filename - CSV is now single source of truth for κ₀
        config.update_param("data.kappa0_filename", kappa0_path)

        # Write Config File
        config_path = os.path.join(model_state_folder, "config_auto_py.json")
        with open(config_path, "w") as f:
            json.dump(config.config, f, indent=4)

        logger.info(f"Prepared config at {config_path}")

        # Note: We do NOT remove seed_path here because batch run needs it later.
        # We rely on final cleanup or manual cleanup.

    def run_profile_sweep(self, profile, baseline_only=False):
        """Prepare files for Baseline (and optionally interventions) for a profile."""
        pid = profile["profile_id"]

        # VALIDATE before generating any configs
        if not self.validate_profile_parameters(profile):
            logger.warning(f"Skipping Profile {pid} due to validation failure")
            return

        seed_size = profile.get("seed_size", 10.0)
        logger.info(f"--- Processing Profile {pid} (Seed={seed_size}) ---")

        # 1. Generate Seed ONCE for this profile
        seed_fname, seed_path = self.prepare_seed_file(pid, seed_size=seed_size)

        # 2. Prepare Baseline
        self.run_single_scenario(pid, "Baseline", 0.0, profile, seed_path)

        # Skip intervention sweep if baseline-only mode
        if baseline_only:
            logger.info(f"Baseline-only mode: skipping intervention sweep for Profile {pid}")
            return

        # 3. Sweep - Test valid range of intervention strengths (0.05 to 0.8)
        # Note: kappa0 (κ₀) must be in range [0, 1] for model stability; we use 0.8 as practical upper bound
        # Note: 0.05 floor ensures all intervention scenarios have a meaningful (non-zero) intervention effect
        #       since 0.0 is equivalent to the Baseline scenario
        strengths = np.linspace(0.05, 0.8, 6)
        scenarios = ["Global_Timed"]

        for scen in scenarios:
            for s in strengths:
                self.run_single_scenario(pid, scen, s, profile, seed_path)

        # Note: We keep seed_path for batch run.

    def check_run_success(self, run_id):
        """Check if a run completed successfully by looking for output files."""
        run_folder = os.path.join(self.output_folder, f"run_{run_id}")
        output_folder = os.path.join(run_folder, "output")

        # Check for ERROR.json
        error_file = os.path.join(run_folder, "ERROR.json")
        if os.path.exists(error_file):
            return False, "ERROR.json found"

        # Check for output observables
        observables_path = os.path.join(output_folder, "observables.nc")
        if not os.path.exists(observables_path):
            return False, "Missing observables.nc"

        return True, None

    def _collect_results(self):
        """Collect run results from output folder."""
        succeeded = []
        failed = []

        for item in os.listdir(self.output_folder):
            if item.startswith("run_"):
                run_id = item[4:]  # Remove "run_" prefix
                success, reason = self.check_run_success(run_id)
                if success:
                    succeeded.append(run_id)
                else:
                    failed.append(run_id)
                    logger.warning(f"Run {run_id} failed: {reason}")

        return {"succeeded": succeeded, "failed": failed}

    def _cleanup_run(self, run_id):
        """Clean up a failed run directory."""
        run_folder = os.path.join(self.output_folder, f"run_{run_id}")
        if os.path.exists(run_folder):
            shutil.rmtree(run_folder)
            logger.info(f"Cleaned up {run_folder}")

    def _retry_failed_runs(self, failed_runs, seed_offset):
        """Retry failed runs with different random seeds."""
        # Extract profile IDs from failed run IDs
        # Run ID format: {pid}_{Scenario}_s{strength}
        profile_ids = set()
        for run_id in failed_runs:
            parts = run_id.split("_")
            if parts:
                try:
                    profile_ids.add(int(parts[0]))
                except ValueError:
                    logger.warning(f"Could not extract profile ID from run_id: {run_id}")
                    continue

        if not profile_ids:
            logger.warning("No valid profile IDs found in failed runs")
            return

        # Re-generate profiles with new seed
        logger.info(f"Re-generating profiles {profile_ids} with seed offset {seed_offset}")
        # Use different seed for LHS generation
        new_seed = 42 + seed_offset * 1000
        new_profiles = self.generate_parameter_grid(
            n_profiles=max(profile_ids) + 1,
            seed=new_seed
        )

        # Re-run failed profiles
        for pid in profile_ids:
            profile = new_profiles[pid]
            # Validate profile before retry
            if not self.validate_profile_parameters(profile):
                logger.warning(f"Profile {pid} failed validation, skipping retry")
                continue
            self.run_profile_sweep(profile)

    def _run_batch_single(self):
        """Execute a single batch run (invoke the Julia batch runner)."""
        logger.info("Starting Batch Execution...")

        julia_path = "julia"  # Assume in path
        script_path = os.path.join(
            self.base_config_path, "..", "..", "..", "src", "batch_run.jl"
        )
        script_path = os.path.abspath(script_path)

        project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        cmd = [
            julia_path,
            "--project=" + project_path,
            "-t",
            "1",
            script_path,
            "--batch-folder",
            self.output_folder,
            "--data-folder",
            self.data_folder,
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            logger.info("Batch execution finished successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Batch execution failed: {e}")
            return False

    def run_batch_with_retry(self, failure_tolerance=10):
        """Run batch with retry logic for failed simulations."""
        logger.info("Starting Batch Execution with Retry Logic...")

        consecutive_failures = 0
        max_attempts = 3  # Max retries per run

        for attempt in range(max_attempts):
            if consecutive_failures >= failure_tolerance:
                logger.error(
                    f"Exceeded failure tolerance ({failure_tolerance} consecutive failures). Aborting."
                )
                return False

            logger.info(f"Attempt {attempt + 1}/{max_attempts}")
            self._run_batch_single()

            # Check results
            results = self._collect_results()
            successful = results["succeeded"]
            failed_runs = results["failed"]

            logger.info(f"Results: {len(successful)} succeeded, {len(failed_runs)} failed")

            if not failed_runs:
                logger.info("All runs succeeded!")
                return True

            # Retry failed runs with new random seeds
            consecutive_failures = len(failed_runs)
            logger.warning(f"{consecutive_failures} runs failed. Retrying with new seeds...")

            # Clean up failed runs and retry
            for run_id in failed_runs:
                self._cleanup_run(run_id)

            # Generate new profiles for retry (different random seed)
            # Use attempt + 1 as seed offset
            self._retry_failed_runs(failed_runs, attempt + 1)

        return False

    def run_spike_based_interventions(self, baseline_dir, spike_threshold=0.1, min_duration=7,
                                      spike_method="percentile", growth_factor_threshold=1.5,
                                      min_growth_duration=3, min_cases_per_capita=1e-4):
        """
        Generate intervention scenarios based on detected spikes in baseline outputs.

        This method implements Phase 2 of the two-phase pipeline:
        1. Detects infection spikes in processed baseline zarr file
        2. Generates intervention scenarios with realistic timing based on spikes

        Args:
            baseline_dir: Directory containing baseline run outputs and zarr
            spike_threshold: Percentile threshold for spike detection (default: 0.1)
            min_duration: Minimum days for spike period (default: 7)
            spike_method: Spike detection method (default: "percentile")
            growth_factor_threshold: Growth factor threshold for growth_rate (default: 1.5)
            min_growth_duration: Minimum consecutive days of growth for growth_rate (default: 3)
            min_cases_per_capita: Minimum cases per person for growth_rate (default: 1e-4)

        Raises:
            FileNotFoundError: If baseline zarr or configs don't exist
            ValueError: If no baselines found or spike detection fails
        """
        from spike_detector import detect_spike_periods_from_zarr

        # Validate baseline directory exists
        # The zarr file is in the parent directory of baseline_dir (output_base)
        # because it contains both baselines AND interventions (interventions are appended)
        baseline_zarr = os.path.join(os.path.dirname(baseline_dir), "raw_synthetic_observations.zarr")
        if not os.path.exists(baseline_dir):
            raise FileNotFoundError(
                f"Baseline directory not found: {baseline_dir}\n"
                f"Please run baselines first with --baseline-only"
            )
        if not os.path.exists(baseline_zarr):
            raise FileNotFoundError(
                f"Baseline zarr not found: {baseline_zarr}\n"
                f"Please process baselines first:\n"
                f"  uv run python synthetic_generator.py --baseline-only --output-folder {os.path.dirname(baseline_dir)}\n"
                f"  uv run python process_synthetic_outputs.py --runs-dir {baseline_dir} --output {baseline_zarr}"
            )

        logger.info("=" * 60)
        logger.info("PHASE 2: Spike-Based Intervention Generation")
        logger.info("=" * 60)
        logger.info(f"Baseline directory: {baseline_dir}")
        logger.info(f"Spike method: {spike_method}")
        logger.info(f"Spike threshold: {spike_threshold} (percentile)")
        logger.info(f"Min spike duration: {min_duration} days")
        if spike_method == "growth_rate":
            logger.info(f"Growth factor threshold: {growth_factor_threshold}")
            logger.info(f"Min growth duration: {min_growth_duration}")
            logger.info(f"Min cases per capita: {min_cases_per_capita}")

        # Load baseline zarr and detect spikes per profile
        spikes = detect_spike_periods_from_zarr(
            baseline_zarr,
            threshold_pct=spike_threshold,
            min_duration=min_duration,
            method=spike_method,
            baseline_filter=True,  # Only process Baseline scenarios
            growth_factor_threshold=growth_factor_threshold,
            min_growth_duration=min_growth_duration,
            min_cases_per_capita=min_cases_per_capita,
        )

        logger.info(f"Detected spikes in {len(spikes)} baseline runs")

        # Track generated scenarios
        scenarios_generated = 0

        # Generate intervention scenarios using detected spike windows
        for run_id, spike_windows in spikes.items():
            # Strip whitespace from run_id (zarr may have trailing spaces)
            run_id = run_id.strip()
            # Extract profile ID from run_ID_Baseline format
            # run_id format: "0_Baseline", "1_Baseline", etc.
            parts = run_id.split("_")
            if len(parts) < 2 or parts[1] != "Baseline":
                logger.warning(f"Skipping non-baseline run_id: {run_id}")
                continue

            try:
                profile_id = int(parts[0])
            except ValueError:
                logger.warning(f"Could not extract profile ID from run_id: {run_id}")
                continue

            # Load original profile parameters from baseline config
            baseline_run_dir = os.path.join(baseline_dir, f"run_{run_id}")
            baseline_config_path = os.path.join(baseline_run_dir, "config_auto_py.json")

            if not os.path.exists(baseline_config_path):
                logger.warning(f"Baseline config not found: {baseline_config_path}")
                continue

            with open(baseline_config_path, "r") as f:
                baseline_config = json.load(f)

            # Generate profile dict from baseline config
            # Extract parameters from config to reconstruct profile
            profile = self._extract_profile_from_config(baseline_config, profile_id)

            # Regenerate seed for this profile (same random seed for reproducibility)
            seed_size = profile.get("seed_size", 10.0)
            _, seed_path = self.prepare_seed_file(profile_id, seed_size=seed_size)

            logger.info(f"--- Profile {profile_id}: Generating {len(spike_windows)} spike-based interventions ---")

            # For each spike window, generate intervention scenarios at different strengths
            strengths = np.linspace(0.05, 0.8, 6)
            for spike_idx, (spike_start, spike_end) in enumerate(spike_windows):
                spike_duration = spike_end - spike_start

                logger.info(f"  Spike {spike_idx + 1}: Day {spike_start} -> {spike_end} (duration={spike_duration}d)")

                for strength in strengths:
                    # Create modified profile with spike-based timing
                    spike_profile = profile.copy()
                    spike_profile["event_start"] = spike_start
                    spike_profile["event_duration"] = spike_duration

                    # Generate scenario
                    self.run_single_scenario(profile_id, "Global_Timed", strength, spike_profile, seed_path)
                    scenarios_generated += 1

        logger.info(f"Generated {scenarios_generated} intervention scenarios")
        return scenarios_generated

    def _extract_profile_from_config(self, config, profile_id):
        """Extract profile parameters from a generated config file."""
        # Extract epidemic parameters
        epidemic_params = config.get("epidemic_params", {})
        npi_params = config.get("NPI", {})

        # Calculate derived parameters
        beta_I = epidemic_params.get("βᴵ", 0.5)
        scale_beta = epidemic_params.get("scale_β", 1.0)
        r0_scale = scale_beta  # Approximation

        # Extract recovery rate (μᵍ) and progression rate (ηᵍ)
        mu_g = epidemic_params.get("μᵍ", [0.2])[0]  # Recovery rate
        eta_g = epidemic_params.get("ηᵍ", [0.2])[0]  # Progression rate

        t_inf = 1.0 / mu_g if mu_g > 0 else 5.0
        t_inc = 1.0 / eta_g if eta_g > 0 else 5.0

        # Extract other parameters
        ratio_beta_a = epidemic_params.get("βᴬ", beta_I * 0.5) / beta_I if beta_I > 0 else 0.5
        alpha_scale = epidemic_params.get("αᵍ", [0.1])[0] / 0.1 if epidemic_params.get("αᵍ") else 1.0

        # Load seed file to get seed_size
        seed_filename = config.get("data", {}).get("initial_condition_filename", "")
        seed_size = 10.0  # Default

        if seed_filename and os.path.exists(seed_filename):
            try:
                seed_df = pd.read_csv(seed_filename)
                # Seed size is sum of M (middle age group) column
                seed_size = seed_df["M"].sum()
            except Exception:
                pass

        return {
            "profile_id": profile_id,
            "r0_scale": r0_scale,
            "t_inf": t_inf,
            "t_inc": t_inc,
            "event_start": 20,  # Placeholder, will be overridden by spike timing
            "event_duration": 30,  # Placeholder
            "affected_fraction": 0.3,
            "ratio_beta_a": ratio_beta_a,
            "alpha_scale": alpha_scale,
            "mu_scale": 1.0,
            "seed_size": int(seed_size),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic data for EpiSim")
    parser.add_argument("--config", default=None, help="Path to base config json")
    parser.add_argument("--data-folder", default=None, help="Path to data folder")
    parser.add_argument("--output-folder", default=None, help="Path to output folder")
    parser.add_argument(
        "--n-profiles",
        type=int,
        default=15,
        help="Total number of profiles to generate",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index for processing (inclusive)",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End index for processing (exclusive)",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean output folder before running"
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running the simulation (generation only)",
    )
    parser.add_argument(
        "--failure-tolerance",
        type=int,
        default=10,
        help="Number of consecutive failures before aborting (default: 10)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Generate baseline scenarios only (no intervention sweep)",
    )
    parser.add_argument(
        "--intervention-only",
        type=str,
        metavar="BASELINE_DIR",
        help="Generate intervention scenarios based on spike analysis of baseline runs in BASELINE_DIR",
    )
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=0.1,
        help="Spike detection threshold percentile for --intervention-only (default: 0.1)",
    )
    parser.add_argument(
        "--min-spike-duration",
        type=int,
        default=7,
        help="Minimum days for spike period for --intervention-only (default: 7)",
    )
    parser.add_argument(
        "--spike-method",
        type=str,
        default="percentile",
        choices=["percentile", "prominence", "growth_rate"],
        help="Spike detection method for --intervention-only (default: percentile)",
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

    args = parser.parse_args()

    # Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    DATA_FOLDER = args.data_folder
    if not DATA_FOLDER:
        DATA_FOLDER = os.path.join(PROJECT_ROOT, "models", "mitma")

    CONFIG_PATH = args.config
    if not CONFIG_PATH:
        CONFIG_PATH = os.path.join(DATA_FOLDER, "config_MMCACovid19.json")

    OUTPUT_FOLDER = args.output_folder
    if not OUTPUT_FOLDER:
        OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "runs", "synthetic_test")

    logger.info(f"Project Root: {PROJECT_ROOT}")

    # Clean previous runs if requested
    if args.clean and os.path.exists(OUTPUT_FOLDER):
        logger.info(f"Cleaning output folder {OUTPUT_FOLDER}")
        shutil.rmtree(OUTPUT_FOLDER)

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    generator = SyntheticDataGenerator(CONFIG_PATH, DATA_FOLDER, OUTPUT_FOLDER)

    # Handle intervention-only mode (Phase 2 of two-phase pipeline)
    if args.intervention_only:
        baseline_dir = args.intervention_only

        logger.info("=" * 60)
        logger.info("INTERVENTION-ONLY MODE: Generating spike-based interventions")
        logger.info("=" * 60)
        logger.info(f"Baseline directory: {baseline_dir}")

        # Generate spike-based intervention scenarios
        generator.run_spike_based_interventions(
            baseline_dir=baseline_dir,
            spike_threshold=args.spike_threshold,
            min_duration=args.min_spike_duration,
            spike_method=args.spike_method,
            growth_factor_threshold=args.growth_factor_threshold,
            min_growth_duration=args.min_growth_duration,
            min_cases_per_capita=args.min_cases_per_capita,
        )

        # Execute batch if not skipped
        if not args.skip_run:
            generator.run_batch_with_retry(failure_tolerance=args.failure_tolerance)

        logger.info("Intervention-only generation complete!")
        sys.exit(0)

    # Standard mode or baseline-only mode (Phase 1 of two-phase pipeline)
    mode = "BASELINE-ONLY" if args.baseline_only else "STANDARD (Baseline + Interventions)"
    logger.info(f"Mode: {mode}")

    # Generate profiles for intervention timing analysis
    # We always generate the FULL set to maintain Latin Hypercube properties
    profiles = generator.generate_parameter_grid(n_profiles=args.n_profiles)

    # Determine subset to process
    start_idx = args.start_index
    end_idx = args.end_index if args.end_index is not None else args.n_profiles

    # Clip to bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(profiles), end_idx)

    logger.info(
        f"Processing profiles {start_idx} to {end_idx} (Total: {len(profiles)})"
    )

    # Prepare files for selected profiles
    for i in range(start_idx, end_idx):
        profile = profiles[i]
        generator.run_profile_sweep(profile, baseline_only=args.baseline_only)

    # Execute batch if not skipped
    if not args.skip_run and start_idx < end_idx:
        generator.run_batch_with_retry(failure_tolerance=args.failure_tolerance)
