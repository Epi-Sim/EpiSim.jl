import os
import shutil
import json
import logging
import numpy as np
import pandas as pd
from scipy.stats import qmc
import sys

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

    def generate_parameter_grid(self, n_samples=10):
        """
        Generate Latin Hypercube Samples for:
        0: R0_scale (scale_β) [0.5, 3.0]
        1: T_inf (Infectious Period) [2.0, 10.0] -> γ = 1/T
        2: T_inc (Incubation Period) [2.0, 10.0] -> η = 1/T
        3: Mobility Scale [0.1, 1.0]
        """
        sampler = qmc.LatinHypercube(d=4, seed=42)
        sample = sampler.random(n=n_samples)

        # Scale samples
        l_bounds = [0.5, 2.0, 2.0, 0.1]
        u_bounds = [3.0, 10.0, 10.0, 1.0]

        params = qmc.scale(sample, l_bounds, u_bounds)
        return params

    def prepare_mobility_file(self, run_id, scale_factor):
        """Scale mobility and save to temp file"""
        scaled_df = self.mobility_df.copy()
        scaled_df["ratio"] = scaled_df["ratio"] * scale_factor

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

    def run_simulation(self, run_id, params):
        r0_scale, t_inf, t_inc, mob_scale = params

        logger.info(f"Starting Run {run_id}")
        logger.info(
            f"Params: R0={r0_scale:.2f}, T_inf={t_inf:.2f}, T_inc={t_inc:.2f}, MobScale={mob_scale:.2f}"
        )

        # Prepare Inputs
        mob_fname, mob_path = self.prepare_mobility_file(run_id, mob_scale)
        seed_fname, seed_path = self.prepare_seed_file(run_id)

        # Update Config
        # We need a deep copy of the config dict
        import copy

        config_dict = copy.deepcopy(self.base_config.config)
        config = EpiSimConfig(config_dict)

        # Calculate rates
        gamma = 1.0 / t_inf
        eta = 1.0 / t_inc

        # Inject updates
        updates = {
            "simulation.end_date": "2020-03-10",  # Shorten run for testing (30 days approx). Remove this line to use full duration.
            "simulation.save_full_output": False,
            "simulation.save_observables": True,
            "epidemic_params.scale_β": r0_scale,
            "epidemic_params.γᵍ": [gamma] * 3,  # Assuming 3 age groups
            "epidemic_params.ηᵍ": [eta] * 3,
            "data.mobility_matrix_filename": mob_path,
            "data.initial_condition_filename": seed_path,
        }

        config.inject(updates)

        # Setup EpiSim
        # We use a unique instance folder for this run
        instance_path = os.path.join(self.output_folder, f"run_{run_id}")
        if not os.path.exists(instance_path):
            os.makedirs(instance_path)

        model = EpiSim(config.config, self.data_folder, instance_path)

        # Setup (assume interpreter for now)
        try:
            model.setup(executable_type="interpreter")

            uuid, output = model.run_model()
            logger.info(f"Run {run_id} complete. UUID: {uuid}")

            # Verify output
            # EpiSim wrapper creates a subfolder with the UUID
            output_file = os.path.join(instance_path, uuid, "output", "observables.nc")
            if os.path.exists(output_file):
                logger.info(f"Output generated at: {output_file}")
                return True
            else:
                logger.error(f"Output file not found at {output_file}!")
                return False
                return False

        except Exception as e:
            logger.error(f"Run {run_id} failed: {e}")
            # raise e # Optional: raise to see traceback
            return False
        finally:
            # Cleanup temp input files
            if os.path.exists(mob_path):
                os.remove(mob_path)
            if os.path.exists(seed_path):
                os.remove(seed_path)


if __name__ == "__main__":
    # Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_FOLDER = os.path.join(PROJECT_ROOT, "models", "mitma")
    CONFIG_PATH = os.path.join(DATA_FOLDER, "config_MMCACovid19.json")
    OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "runs", "synthetic_test")

    logger.info(f"Project Root: {PROJECT_ROOT}")

    generator = SyntheticDataGenerator(CONFIG_PATH, DATA_FOLDER, OUTPUT_FOLDER)

    # Generate 10 samples as requested
    params_grid = generator.generate_parameter_grid(n_samples=10)

    for i, params in enumerate(params_grid):
        generator.run_simulation(i, params)
