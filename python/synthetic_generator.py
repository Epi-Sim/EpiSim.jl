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

from episim_python.epi_sim import EpiSim
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

    def generate_parameter_grid(self, n_profiles=5):
        """
        Generate Latin Hypercube Samples for Epidemiological Profiles:
        0: R0_scale (scale_β) [0.5, 3.0]
        1: T_inf (Infectious Period) [2.0, 10.0] -> γ = 1/T
        2: T_inc (Incubation Period) [2.0, 10.0] -> η = 1/T
        3: Event Start [5, 40]
        4: Event Duration [7, 60]
        5: Affected Fraction [0.1, 0.6] (For Local Scenarios)
        """
        sampler = qmc.LatinHypercube(d=6)
        sample = sampler.random(n=n_profiles)

        # Scale samples
        l_bounds = [0.5, 2.0, 2.0, 5.0, 7.0, 0.1]
        u_bounds = [3.0, 10.0, 10.0, 40.0, 60.0, 0.6]

        scaled = qmc.scale(sample, l_bounds, u_bounds)

        profiles = []
        for i, row in enumerate(scaled):
            r0, t_inf, t_inc, start, duration, fraction = row

            profiles.append(
                {
                    "profile_id": i,
                    "r0_scale": r0,
                    "t_inf": t_inf,
                    "t_inc": t_inc,
                    "event_start": int(start),
                    "event_duration": int(duration),
                    "affected_fraction": fraction,
                }
            )

        return profiles

    def prepare_kappa0_file(
        self, run_id, scenario, strength, profile, start_date_str, end_date_str
    ):
        """Generate kappa0.csv file for Global scenarios."""
        # Parse dates
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)

        # Create date range (inclusive)
        date_range = pd.date_range(start=start_date, end=end_date)
        n_days = len(date_range)

        # Initialize dataframe
        df = pd.DataFrame(
            {
                "date": date_range,
                "reduction": 1.0,  # Default to no reduction (kappa=1)
                "datetime": date_range,
                "time": range(n_days),
            }
        )

        # Apply reduction
        # reduction value is kappa0 = 1 - strength
        kappa_val = max(0.0, 1.0 - strength)

        if scenario == "Global_Const":
            df["reduction"] = kappa_val
        elif scenario == "Global_Timed":
            event_start = profile["event_start"]
            event_duration = profile["event_duration"]

            # Ensure indices are within bounds
            start_idx = min(event_start, n_days)
            end_idx = min(event_start + event_duration, n_days)

            # Apply reduction during event window
            if start_idx < end_idx:
                df.loc[start_idx : end_idx - 1, "reduction"] = kappa_val

        # Format dates as string for CSV
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d")

        filename = f"kappa0_{run_id}.csv"
        path = os.path.join(self.output_folder, filename)
        df.to_csv(path, index=False)
        return filename, path

    def prepare_mobility_file(
        self, run_id, scenario, reduction_strength, affected_fraction
    ):
        """
        Prepare mobility matrix based on scenario.
        For Local scenarios, reduce connectivity for a subset of nodes.
        For Global scenarios, return original matrix (unscaled).
        """
        scaled_df = self.mobility_df.copy()

        if "Local" in scenario:
            # Localized reduction
            # Select random subset of nodes
            # Columns are source_idx, target_idx, ratio
            unique_nodes = pd.concat(
                [self.mobility_df["source_idx"], self.mobility_df["target_idx"]]
            ).unique()
            unique_nodes_list = list(unique_nodes)
            n_affected = int(len(unique_nodes_list) * affected_fraction)
            affected_nodes = np.random.choice(
                unique_nodes_list, size=n_affected, replace=False
            )
            affected_nodes_list = list(affected_nodes)

            # Reduce edges connected to these nodes
            mask = scaled_df["source_idx"].isin(affected_nodes_list) | scaled_df[
                "target_idx"
            ].isin(affected_nodes_list)

            # Apply reduction (1 - strength)
            factor = 1.0 - reduction_strength
            scaled_df.loc[mask, "ratio"] = scaled_df.loc[mask, "ratio"] * factor

        # For Global scenarios, we rely on kappa0, so we use the base matrix (or the one modified above)

        filename = f"mobility_matrix_{run_id}.csv"
        path = os.path.join(self.output_folder, filename)
        scaled_df.to_csv(path, index=False)
        return filename, path

    def prepare_seed_file(self, run_id):
        """Generate random seed file"""
        # Pick a random region (row index)
        random_idx = np.random.randint(0, len(self.metapop_df))
        seed_region = self.metapop_df.iloc[random_idx]
        region_id = seed_region["id"]

        # Julia uses 1-based indexing for patches_idx
        idx = random_idx + 1

        # Create seed dataframe
        # We start with 10 Exposed/Asymptomatic individuals in Middle age group (M)
        seed_data = {
            "name": ["SeedRegion"],
            "id": [region_id],
            "idx": [idx],
            "Y": [0.0],
            "M": [10.0],
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
        event_start = profile["event_start"]
        event_duration = profile["event_duration"]
        fraction = profile["affected_fraction"]

        # Determine if mobility modification is needed
        mob_mod = "Local" in scen_name

        # Construct Run ID
        # Format: {pid}_{Scenario}_s{strength_int} where strength_int is percent
        str_pct = int(round(strength * 100))
        run_id = f"{pid}_{scen_name}_s{str_pct:02d}"
        if scen_name == "Baseline":
            run_id = f"{pid}_Baseline"

        logger.info(f"Preparing {run_id} (Str={strength:.2f})")

        # Prepare Mobility
        if mob_mod:
            mob_fname, mob_path = self.prepare_mobility_file(
                run_id, "Local", strength, fraction
            )
        else:
            mob_fname, mob_path = self.prepare_mobility_file(run_id, "Global", 0.0, 0.0)

        # Update Config
        import copy

        config_dict = copy.deepcopy(self.base_config.config)
        config = EpiSimConfig(config_dict)

        # Get start/end dates
        start_date = config.get_param("simulation.start_date")
        end_date = "2020-06-01"

        gamma = 1.0 / t_inf
        eta = 1.0 / t_inc

        # NPI Params Setup
        kappa0_path = None

        if "Global" in scen_name:
            _, kappa0_path = self.prepare_kappa0_file(
                run_id, scen_name, strength, profile, start_date, end_date
            )

        # Inject updates
        updates = {
            "simulation.end_date": end_date,
            "simulation.save_full_output": False,
            "simulation.save_observables": True,
            "epidemic_params.scale_β": r0_scale,
            "epidemic_params.γᵍ": [gamma] * 3,
            "epidemic_params.ηᵍ": [eta] * 3,
            "data.mobility_matrix_filename": mob_path,
            "data.initial_condition_filename": seed_path,
            "NPI.are_there_npi": True,
        }

        config.inject(updates)

        if kappa0_path:
            config.update_param("data.kappa0_filename", kappa0_path)
            # Note: We keep κ₀s and tᶜs in config to avoid KeyError in engine
            # The engine will use kappa0 file values when provided
        else:
            # Ensure no kappa0 file is set for Local/Baseline
            if "kappa0_filename" in config.config["data"]:
                del config.config["data"]["kappa0_filename"]
            config.update_param("NPI.tᶜs", [0])
            config.update_param("NPI.κ₀s", [1.0])

            # Store direct mobility reduction strength for Local scenarios
            if mob_mod and strength > 0:
                config.update_param("NPI.custom.direct_mobility_reduction", strength)

        # Prepare Directory Structure
        instance_path = os.path.join(self.output_folder, f"run_{run_id}")
        if not os.path.exists(instance_path):
            os.makedirs(instance_path)

        # Create UUID subfolder manually (simulating what EpiSim wrapper did)
        import uuid

        run_uuid = str(uuid.uuid4())
        model_state_folder = os.path.join(instance_path, run_uuid)
        os.makedirs(model_state_folder)

        # Write Config File
        config_path = os.path.join(model_state_folder, "config_auto_py.json")
        with open(config_path, "w") as f:
            json.dump(config.config, f, indent=4)

        logger.info(f"Prepared config at {config_path}")

        # Note: We do NOT remove mob_path here because batch run needs it later.
        # We rely on final cleanup or manual cleanup.

    def run_profile_sweep(self, profile):
        """Prepare files for Baseline and Sweep of Interventions for a profile."""
        pid = profile["profile_id"]
        logger.info(f"--- Processing Profile {pid} ---")

        # 1. Generate Seed ONCE for this profile
        seed_fname, seed_path = self.prepare_seed_file(pid)

        # 2. Prepare Baseline
        self.run_single_scenario(pid, "Baseline", 0.0, profile, seed_path)

        # 3. Sweep
        strengths = np.linspace(0.0, 1.0, 6)
        scenarios = ["Global_Const", "Global_Timed", "Local_Static"]

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
    # Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_FOLDER = os.path.join(PROJECT_ROOT, "models", "mitma")
    CONFIG_PATH = os.path.join(DATA_FOLDER, "config_MMCACovid19.json")
    OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "runs", "synthetic_test")

    logger.info(f"Project Root: {PROJECT_ROOT}")

    # Clean previous runs
    if os.path.exists(OUTPUT_FOLDER):
        logger.info(f"Cleaning output folder {OUTPUT_FOLDER}")
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)

    generator = SyntheticDataGenerator(CONFIG_PATH, DATA_FOLDER, OUTPUT_FOLDER)

    # Generate 5 profiles
    profiles = generator.generate_parameter_grid(n_profiles=5)

    # Prepare files for all profiles
    for profile in profiles:
        generator.run_profile_sweep(profile)

    # Execute all at once
    generator.run_batch()
