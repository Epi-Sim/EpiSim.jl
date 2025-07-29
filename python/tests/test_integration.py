"""
Integration tests for episim_python package

These tests verify end-to-end functionality and require a working Julia environment.
"""

import os
import shutil
from unittest.mock import patch

import pytest

from episim_python import EpiSimConfig

from .conftest import BaseTestCase


class TestIntegration(BaseTestCase):
    """Integration tests for the episim_python package"""

    @pytest.fixture(scope="class")
    def julia_available(self):
        """Check if Julia is available for integration tests"""
        return shutil.which("julia") is not None

    def test_episim_config_full_workflow(self, test_config_json, temp_dir):
        """Test complete EpiSimConfig workflow"""
        # Load configuration
        config = EpiSimConfig.from_json(str(test_config_json))

        # Validate
        config.validate(verbose=False)

        # Modify parameters
        config.update_param("epidemic_params.βᴵ", 0.15)
        config.update_group_param("epidemic_params.ηᵍ", "M", 0.4)

        # Batch updates
        updates = {
            "simulation.start_date": "2020-02-01",
            "epidemic_params.βᴬ": 0.075,
            "epidemic_params.γᵍ": [0.005, 0.010, 0.080],
        }
        config.inject(updates)

        # Save modified configuration
        output_path = os.path.join(temp_dir, "modified_config.json")
        config.to_json(output_path)

        # Verify saved configuration
        self.assertions.assert_config_saved_correctly(
            output_path,
            {
                "epidemic_params.βᴵ": 0.15,
                "epidemic_params.ηᵍ": [0.3, 0.4, 0.3],  # M group (index 1) modified
                "simulation.start_date": "2020-02-01",
                "epidemic_params.γᵍ": [0.005, 0.010, 0.080],
            },
        )

    def test_episim_initialization_workflow(
        self, integration_config, instance_folder, temp_dir
    ):
        """Test EpiSim initialization and setup workflow"""
        # Test initialization with dict config
        model = self.helpers.setup_episim_model(
            integration_config, temp_dir, instance_folder
        )

        self.helpers.assert_episim_model_structure(model)

        # Test configuration update
        new_config = integration_config.copy()
        new_config["simulation"]["start_date"] = "2020-01-05"
        model.update_config(new_config)

        # Verify config was updated
        self.assertions.assert_config_saved_correctly(
            model.config_path, {"simulation.start_date": "2020-01-05"}
        )

        # Test backend engine setting
        model.set_backend_engine("MMCACovid19")
        assert model.backend_engine == "MMCACovid19"

    def test_episim_mock_simulation_workflow(
        self, integration_config, instance_folder, temp_dir, mock_subprocess_run
    ):
        """Test complete simulation workflow with mocked subprocess"""
        # Initialize model
        model = self.helpers.setup_episim_model(
            integration_config, temp_dir, instance_folder
        )

        # Setup with mocked Julia check
        with patch("shutil.which", return_value="/usr/bin/julia"):
            model.setup(executable_type="interpreter")

        # Run full simulation
        uuid_result, output = model.run_model()

        assert uuid_result == model.uuid
        assert "Simulation completed successfully" in output

        # Verify subprocess was called correctly
        self.assertions.assert_subprocess_called_correctly(
            mock_subprocess_run,
            ["julia", "run", "--config", "--data-folder", "--instance-folder"],
            ["run.jl"],
        )

    @pytest.mark.skip(
        reason="Step-by-step execution is experimental and untested - only single simulation runs are officially supported",
    )
    def test_episim_step_by_step_workflow(self):
        """Test step-by-step simulation workflow - EXPERIMENTAL FEATURE"""
        # NOTE: Step-by-step execution is an experimental feature for RL agents
        # and is not officially supported. Only single simulation runs are tested and supported.

    def test_config_parameter_update_workflow(self, integration_config):
        """Test comprehensive parameter update workflow"""
        config = EpiSimConfig(integration_config)

        # Test scalar parameter updates
        config.update_param("epidemic_params.βᴵ", 0.12)
        assert config.get_param("epidemic_params.βᴵ") == 0.12

        # Test group parameter updates
        new_eta = [0.35, 0.40, 0.45]
        config.update_param("epidemic_params.ηᵍ", new_eta)
        assert config.get_param("epidemic_params.ηᵍ") == new_eta

        # Test individual group updates
        config.update_group_param("epidemic_params.αᵍ", "O", 0.7)
        alpha_values = config.get_param("epidemic_params.αᵍ")
        assert alpha_values[2] == 0.7  # O group is index 2

        # Test batch updates
        batch_updates = {
            "simulation.end_date": "2020-01-10",
            "epidemic_params.μᵍ": [1.0, 0.35, 0.35],
            "NPI.are_there_npi": False,
        }
        config.inject(batch_updates)

        assert config.get_param("simulation.end_date") == "2020-01-10"
        assert config.get_param("epidemic_params.μᵍ") == [1.0, 0.35, 0.35]
        assert config.get_param("NPI.are_there_npi") is False

        # Test validation after updates
        config.validate(verbose=False)  # Should not raise exception

    @pytest.mark.skipif(not shutil.which("julia"), reason="Julia not available")
    def test_real_julia_check(self):
        """Test that Julia is actually available (only runs if Julia is installed)"""
        # This test will be skipped if Julia is not available
        # but will run if Julia is present to verify the environment
        result = shutil.which("julia")
        assert result is not None
        self.assertions.assert_file_exists(result)
        assert os.access(result, os.X_OK)

    def test_metapopulation_episim_integration(
        self,
        test_metapopulation_csv,
        integration_config,
        instance_folder,
        temp_dir,
    ):
        """Test integration between Metapopulation and EpiSim classes"""
        from episim_python import Metapopulation

        # Load metapopulation data
        metapop = Metapopulation(str(test_metapopulation_csv))

        # Get population data as array
        pop_array = metapop.as_datarray()

        # Verify structure matches config expectations
        config = EpiSimConfig(integration_config)
        expected_groups = config.group_labels

        self.assertions.assert_dataarray_structure(
            pop_array,
            ["M", "G"],
            {"G": expected_groups},
            (3, len(expected_groups)),  # 3 regions, N age groups
        )

        # Update config to use metapopulation file
        integration_config["data"]["metapopulation_data_filename"] = str(
            test_metapopulation_csv,
        )

        # Initialize EpiSim with updated config
        model = self.helpers.setup_episim_model(
            integration_config, temp_dir, instance_folder
        )

        # Verify config was saved correctly
        self.assertions.assert_config_saved_correctly(
            model.config_path,
            {"data.metapopulation_data_filename": str(test_metapopulation_csv)},
        )
