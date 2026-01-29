import os
import sys

import pandas as pd
import pytest

# Add python directory to path to import synthetic_generator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from episim_python.epi_sim import EpiSim
from synthetic_generator import SyntheticDataGenerator


class TestSyntheticMobility:
    @pytest.fixture
    def setup_generator(self, temp_dir):
        """Setup SyntheticDataGenerator with real data paths but temp output"""
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        data_folder = os.path.join(project_root, "models", "mitma")
        config_path = os.path.join(data_folder, "config_MMCACovid19.json")
        output_folder = os.path.join(temp_dir, "synthetic_test_output")

        generator = SyntheticDataGenerator(config_path, data_folder, output_folder)
        return generator, output_folder

    def run_simulation(self, run_folder, data_folder, days=20):
        """Run simulation using EpiSim python wrapper"""
        # Find config file
        config_path = os.path.join(run_folder, "config_auto_py.json")

        # Load config
        with open(config_path) as f:
            import json

            config_dict = json.load(f)

        # Log intervention parameters
        print(f"  Config κ₀s: {config_dict['NPI']['κ₀s']}")
        print(f"  Config tᶜs: {config_dict['NPI']['tᶜs']}")

        kappa0_file = config_dict["data"].get("kappa0_filename")
        if kappa0_file:
            print(f"  Using CSV: {kappa0_file}")
            kappa0_df = pd.read_csv(kappa0_file)
            print(f"  CSV reduction values: {kappa0_df['reduction'].unique()[:5]}")
        else:
            print("  Using JSON mode (no CSV)")

        model = EpiSim(
            config=config_dict,
            data_folder=data_folder,
            instance_folder=os.path.dirname(run_folder),
            initial_conditions=None,
        )

        model.setup(executable_type="interpreter")

        start_date = config_dict["simulation"]["start_date"]

        # Run in one go to save time (avoiding repeated Julia startup)
        final_state, next_date = model.step(start_date, length_days=days)

        # Read observables NetCDF to get total infections
        # final_state is the path to the compartments file at t=end_date
        # Observables are in the same output folder as the compartments file
        observables_path = os.path.join(os.path.dirname(final_state), "observables.nc")

        # Use xarray to read
        import xarray as xr

        ds = xr.open_dataset(observables_path)

        # new_infected dimensions: (G, M, T) - sum over all dimensions
        total_infections = ds["new_infected"].sum().values

        ds.close()

        return float(total_infections)

    def test_mobility_reduction_logic(self, setup_generator):
        generator, output_folder = setup_generator

        # Define a high R0 profile
        profile = {
            "profile_id": 999,
            "r0_scale": 2.5,
            "t_inf": 3.0,
            "t_inc": 5.0,
            "event_start": 5,
            "event_duration": 40,
            "affected_fraction": 0.5,
        }

        # 1. Prepare Baseline (Strength 0.0)
        seed_filename, seed_path = generator.prepare_seed_file("test_seed")

        generator.run_single_scenario(
            pid="test",
            scen_name="Baseline",
            strength=0.0,
            profile=profile,
            seed_path=seed_path,
        )

        baseline_run_path = os.path.join(output_folder, "run_test_Baseline")

        # 2. Prepare Intervention (κ₀=1.0, i.e., full confinement) - Global Timed
        generator.run_single_scenario(
            pid="test",
            scen_name="Global_Timed",
            strength=1.0,
            profile=profile,
            seed_path=seed_path,
        )

        intervention_run_path = os.path.join(output_folder, "run_test_Global_Timed_s100")

        # Run Simulations
        print("Running Baseline (κ₀=0.0) Simulation...")
        baseline_infections = self.run_simulation(
            baseline_run_path, generator.data_folder, days=20
        )

        print("Running Intervention (κ₀=1.0) Simulation...")
        intervention_infections = self.run_simulation(
            intervention_run_path, generator.data_folder, days=20
        )

        print(f"Baseline Total Infections: {baseline_infections}")
        print(f"Intervention Total Infections: {intervention_infections}")

        # Expectation: κ₀=1.0 (full confinement) should have FEWER infections than κ₀=0.0 (no confinement)
        assert intervention_infections < baseline_infections, (
            f"κ₀=1.0 (full confinement) should have FEWER total infections than κ₀=0.0 (no confinement). "
            f"Got Intervention (κ₀=1.0) infections={intervention_infections}, Baseline (κ₀=0.0) infections={baseline_infections}"
        )

    def test_kappa0_extremes_csv_mode(self, setup_generator):
        """Direct test: κ₀=1.0 (full confinement) vs κ₀=0.0 (no confinement) using CSV mode."""
        generator, output_folder = setup_generator

        profile = {
            "profile_id": 998,
            "r0_scale": 2.0,
            "t_inf": 4.0,
            "t_inc": 5.0,
            "event_start": 5,
            "event_duration": 20,
            "affected_fraction": 0.5,
        }

        seed_filename, seed_path = generator.prepare_seed_file("test_seed2")

        # κ₀=0.0 (no confinement) via CSV
        generator.run_single_scenario(
            pid="test_extremes",
            scen_name="Global_Timed",
            strength=0.0,
            profile=profile,
            seed_path=seed_path,
        )

        run_path_0 = os.path.join(output_folder, "run_test_extremes_Global_Timed_s00")

        # κ₀=1.0 (full confinement) via CSV
        generator.run_single_scenario(
            pid="test_extremes",
            scen_name="Global_Timed",
            strength=1.0,
            profile=profile,
            seed_path=seed_path,
        )

        run_path_1 = os.path.join(output_folder, "run_test_extremes_Global_Timed_s100")

        # Run both
        print("Running κ₀=0.0 (no confinement)...")
        infections_0 = self.run_simulation(run_path_0, generator.data_folder, days=20)

        print("Running κ₀=1.0 (full confinement)...")
        infections_1 = self.run_simulation(run_path_1, generator.data_folder, days=20)

        print(f"κ₀=0.0 infections: {infections_0}")
        print(f"κ₀=1.0 infections: {infections_1}")

        # κ₀=1.0 should have significantly fewer infections
        assert infections_1 < infections_0, (
            f"κ₀=1.0 (full confinement) must have fewer infections than κ₀=0.0 (no confinement). "
            f"Got κ₀=1.0: {infections_1}, κ₀=0.0: {infections_0}"
        )
