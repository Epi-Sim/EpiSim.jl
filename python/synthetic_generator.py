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

        # Load Metapopulation Data for seeding
        self.metapop_df = pd.read_csv(
            os.path.join(data_folder, "metapopulation_data.csv")
        )

        # Load Mobility Matrix
        self.mobility_df = pd.read_csv(
            os.path.join(data_folder, "R_mobility_matrix.csv")
        )

    def generate_parameter_grid(self, n_profiles=5, seed=42):
        """
        Generate Latin Hypercube Samples for Epidemiological Profiles:
        0: R0_scale (scale_β) [0.5, 3.0]
        1: T_inf (Infectious Period) [2.0, 10.0] -> γ = 1/T
        2: T_inc (Incubation Period) [2.0, 10.0] -> η = 1/T
        3: Reaction Delay [0, 30] (Replaces absolute Event Start)
        4: Event Duration [7, 60]
        5: Affected Fraction [0.1, 0.6] (For Local Scenarios)
        6: Ratio Beta A (ratio_beta_a) [0.1, 1.0]
        7: Alpha Scale (alpha_scale) [0.5, 1.5]
        8: Mu Scale (mu_scale) [0.5, 1.5]
        9: Seed Size [10, 500] (Log-uniform sampled)
        """
        sampler = qmc.LatinHypercube(d=10, seed=seed)
        sample = sampler.random(n=n_profiles)

        # Scale samples
        # Note: We'll transform seed_size to log scale later if desired,
        # but for now linear [10, 500] is fine or we can do manual log transform.
        # Let's do linear for simplicity unless specified.
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

        # Julia uses 1-based indexing for patches_idx
        idx = random_idx + 1

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

        gamma = 1.0 / t_inf
        eta = 1.0 / t_inc

        # Calculate derived params
        base_beta_I = config.get_param("epidemic_params.βᴵ")
        base_alpha = config.get_param("epidemic_params.αᵍ")
        base_mu = config.get_param("epidemic_params.μᵍ")

        new_beta_A = base_beta_I * ratio_beta_a
        new_alpha = [min(1.0, x * alpha_scale) for x in base_alpha]
        new_mu = [x * mu_scale for x in base_mu]

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
        updates = {
            "simulation.end_date": end_date,
            "simulation.save_full_output": False,
            "simulation.save_observables": True,
            "epidemic_params.scale_β": r0_scale,
            "epidemic_params.βᴬ": new_beta_A,
            "epidemic_params.αᵍ": new_alpha,
            "epidemic_params.μᵍ": new_mu,
            "epidemic_params.γᵍ": [gamma] * 3,
            "epidemic_params.ηᵍ": [eta] * 3,
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

    def run_profile_sweep(self, profile):
        """Prepare files for Baseline and Sweep of Interventions for a profile."""
        pid = profile["profile_id"]
        seed_size = profile.get("seed_size", 10.0)
        logger.info(f"--- Processing Profile {pid} (Seed={seed_size}) ---")

        # 1. Generate Seed ONCE for this profile
        seed_fname, seed_path = self.prepare_seed_file(pid, seed_size=seed_size)

        # 2. Prepare Baseline
        self.run_single_scenario(pid, "Baseline", 0.0, profile, seed_path)

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

    def run_batch(self):
        """Invoke the Julia batch runner."""
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
        except subprocess.CalledProcessError as e:
            logger.error(f"Batch execution failed: {e}")


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
        generator.run_profile_sweep(profile)

    # Execute batch if not skipped
    if not args.skip_run and start_idx < end_idx:
        generator.run_batch()
