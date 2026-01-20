import json
import os
import sys
from unittest.mock import call, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Add python directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import run_synthetic_pipeline
from synthetic_generator import SyntheticDataGenerator


class TestIterativePipeline:
    @pytest.fixture
    def mock_paths(self, tmp_path):
        """Setup temporary paths for testing"""
        data_folder = tmp_path / "models" / "mitma"
        data_folder.mkdir(parents=True)

        output_folder = tmp_path / "runs" / "synthetic_test"
        output_folder.mkdir(parents=True)

        config_path = data_folder / "config_MMCACovid19.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "simulation": {},
                    "epidemic_params": {},
                    "population_params": {"G_labels": ["Y", "M", "O"]},
                    "NPI": {},
                },
                f,
            )

        # Create dummy metapop csv
        metapop_csv = data_folder / "metapopulation_data.csv"
        pd.DataFrame({"id": ["1", "2"], "total": [1000, 2000]}).to_csv(
            metapop_csv, index=False
        )

        # Create dummy mobility matrix
        mobility_csv = data_folder / "R_mobility_matrix.csv"
        pd.DataFrame({"from": [1, 2], "to": [2, 1], "weight": [0.1, 0.1]}).to_csv(
            mobility_csv, index=False
        )

        return {
            "root": tmp_path,
            "data": str(data_folder),
            "output": str(output_folder),
            "config": str(config_path),
        }

    def test_lhs_consistency(self, mock_paths):
        """Ensure batching preserves statistical properties (LHS consistency)."""
        gen = SyntheticDataGenerator(
            mock_paths["config"], mock_paths["data"], mock_paths["output"]
        )

        n_profiles = 10

        # 1. Full generation
        full_profiles = gen.generate_parameter_grid(n_profiles=n_profiles)

        # 2. "Partial" generation (simulating the generator's behavior)
        # The generator always calls generate_parameter_grid(n_profiles) fully,
        # then the orchestrator slices it.
        # We verify that calling it multiple times yields deterministic results
        # (assuming the seed handling in qmc or just stability of the implementation).
        # Note: synthetic_generator.py uses `sampler = qmc.LatinHypercube(d=10)`.
        # Unless seeded, it might differ. Let's check if we can seed it or if it's stable.
        # Looking at synthetic_generator.py, it doesn't set a seed for qmc explicitly.
        # However, we should verify that if we WERE to process it iteratively,
        # we would assume the LIST of profiles is constant if we generate the whole list every time.

        # Let's verify that generating the full list twice yields different results
        # (since no seed is set inside the method),
        # BUT the logic in `run_synthetic_pipeline.py` relies on `synthetic_generator.py`
        # generating the *same* profiles if we run it multiple times?
        # WAIT. `run_synthetic_pipeline.py` calls `synthetic_generator.py` via subprocess.
        # Each subprocess call instantiates `SyntheticDataGenerator` and calls `generate_parameter_grid`.
        # If `generate_parameter_grid` is not deterministic (no fixed seed),
        # then Batch 1 will generate Set A of profiles and process 0-5.
        # Batch 2 will generate Set B of profiles and process 5-10.
        # This is BAD. Set A[5-10] != Set B[5-10].

        # Let's check `synthetic_generator.py` again.
        # It imports `from scipy.stats import qmc`.
        # `sampler = qmc.LatinHypercube(d=10)`
        # `sample = sampler.random(n=n_profiles)`
        # Unless `seed` is passed to LatinHypercube, it's random.

        # Test LHS consistency

        # We need to verify if generate_parameter_grid is deterministic without an explicit seed
        # across different instances.

        gen1 = SyntheticDataGenerator(
            mock_paths["config"], mock_paths["data"], mock_paths["output"]
        )
        profiles_1 = gen1.generate_parameter_grid(n_profiles=10)

        gen2 = SyntheticDataGenerator(
            mock_paths["config"], mock_paths["data"], mock_paths["output"]
        )
        profiles_2 = gen2.generate_parameter_grid(n_profiles=10)

        # This assertion will likely FAIL if the generator is not seeded fixedly.
        # If it fails, we know we must fix the generator code.
        # For now, let's just observe.
        assert profiles_1 == profiles_2, (
            "Profiles should be deterministic across instances!"
        )

    @patch("run_synthetic_pipeline.run_simulation_batch")
    @patch("run_synthetic_pipeline.process_outputs_batch")
    @patch("run_synthetic_pipeline.clean_run_folders")
    def test_pipeline_orchestration(
        self, mock_clean, mock_process, mock_sim, mock_paths
    ):
        """Test the batching logic of the orchestration script."""

        n_profiles = 15
        batch_size = 5

        # We need to mock OUTPUT_FOLDER in the imported module to prevent real FS operations
        with patch(
            "run_synthetic_pipeline.OUTPUT_FOLDER",
            new=mock_paths["root"] / "runs" / "synthetic_test",
        ):
            run_synthetic_pipeline.run_iterative_pipeline(
                n_profiles=n_profiles, batch_size=batch_size
            )

        # Verify Simulation calls: 3 batches (0-5, 5-10, 10-15)
        assert mock_sim.call_count == 3
        mock_sim.assert_has_calls(
            [
                call(n_profiles=15, start_idx=0, end_idx=5, clean_output_dir=False),
                call(n_profiles=15, start_idx=5, end_idx=10, clean_output_dir=False),
                call(n_profiles=15, start_idx=10, end_idx=15, clean_output_dir=False),
            ]
        )

        # Verify Process calls: 3 batches
        # Batch 1 (idx=0): append=False
        # Batch 2 (idx=1): append=True
        # Batch 3 (idx=2): append=True
        assert mock_process.call_count == 3
        mock_process.assert_has_calls(
            [call(append=False), call(append=True), call(append=True)]
        )

        # Verify Cleanup: 1 (before 1st batch) + 1 (before 2nd) + 1 (before 3rd) + 1 (final) = 4
        # Wait, the logic is:
        # loop:
        #   clean()
        #   sim()
        #   process()
        # end
        # clean()
        # So yes, 4 calls.
        assert mock_clean.call_count == 4

    def test_zarr_appending(self, mock_paths):
        """Functional test for zarr appending."""
        output_zarr = mock_paths["output"]  # Actually we want a file path
        zarr_path = os.path.join(mock_paths["output"], "test.zarr")

        # We need to simulate the environment process_synthetic_outputs expects.
        # It loads 'observables.nc' and creates zarr.

        # We can't easily replicate the full logic without real NetCDF files.
        # Instead, let's assume the subprocess logic in run_synthetic_pipeline works
        # and focus on the fact that we confirmed it passes flags correctly.
        # BUT, we can test the `xarray.to_zarr(mode='a')` behavior with a small dummy dataset.

        ds1 = xr.Dataset(
            {"data": (("run", "x"), np.random.rand(2, 5))},
            coords={"run": [0, 1], "x": range(5)},
        )

        ds2 = xr.Dataset(
            {"data": (("run", "x"), np.random.rand(2, 5))},
            coords={"run": [2, 3], "x": range(5)},
        )

        # Write first batch
        ds1.to_zarr(zarr_path, mode="w")

        # Append second batch
        ds2.to_zarr(zarr_path, mode="a", append_dim="run")

        # Read back
        ds_combined = xr.open_zarr(zarr_path)

        assert len(ds_combined.run) == 4
        assert np.array_equal(ds_combined.run, [0, 1, 2, 3])
        # Check data
        np.testing.assert_allclose(ds_combined.data.isel(run=slice(0, 2)), ds1.data)
        np.testing.assert_allclose(ds_combined.data.isel(run=slice(2, 4)), ds2.data)
