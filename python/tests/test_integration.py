"""
Integration tests for episim_python package

These tests verify end-to-end functionality and require a working Julia environment.
"""

import pytest
import os
import json
import shutil
from unittest.mock import patch

from episim_python import EpiSim, EpiSimConfig


class TestIntegration:
    """Integration tests for the episim_python package"""

    @pytest.fixture(scope="class")
    def julia_available(self):
        """Check if Julia is available for integration tests"""
        return shutil.which("julia") is not None

    @pytest.fixture
    def integration_config(self, minimal_config):
        """Configuration suitable for integration testing"""
        config = minimal_config.copy()
        # Use very short simulation period for faster tests
        config["simulation"]["start_date"] = "2020-01-01"
        config["simulation"]["end_date"] = "2020-01-03"  # Just 3 days
        return config

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
            "epidemic_params.γᵍ": [0.005, 0.010, 0.080]  # Use actual group param instead
        }
        config.inject(updates)

        # Save modified configuration
        output_path = os.path.join(temp_dir, "modified_config.json")
        config.to_json(output_path)

        # Verify saved configuration
        assert os.path.exists(output_path)
        with open(output_path, "r") as f:
            saved_config = json.load(f)

        assert saved_config["epidemic_params"]["βᴵ"] == 0.15
        assert saved_config["epidemic_params"]["ηᵍ"][1] == 0.4  # M group
        assert saved_config["simulation"]["start_date"] == "2020-02-01"
        assert saved_config["epidemic_params"]["γᵍ"] == [0.005, 0.010, 0.080]

    def test_episim_initialization_workflow(self, integration_config, temp_dir):
        """Test EpiSim initialization and setup workflow"""
        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        # Test initialization with dict config
        model = EpiSim(integration_config, temp_dir, instance_folder)

        assert model.uuid is not None
        assert os.path.exists(model.model_state_folder)
        assert os.path.exists(model.config_path)

        # Test configuration update
        new_config = integration_config.copy()
        new_config["simulation"]["start_date"] = "2020-01-05"
        model.update_config(new_config)

        # Verify config was updated
        with open(model.config_path, "r") as f:
            saved_config = json.load(f)
        assert saved_config["simulation"]["start_date"] == "2020-01-05"

        # Test backend engine setting
        model.set_backend_engine("MMCACovid19")
        assert model.backend_engine == "MMCACovid19"

    @patch("subprocess.run")
    def test_episim_mock_simulation_workflow(
        self, mock_run, integration_config, temp_dir
    ):
        """Test complete simulation workflow with mocked subprocess"""
        # Setup mock subprocess responses
        mock_result = type(
            "MockResult",
            (),
            {
                "returncode": 0,
                "stdout": "Simulation completed successfully\nTime: 100.5s\nSteps: 3",
                "stderr": "",
            },
        )
        mock_run.return_value = mock_result

        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        # Initialize model
        model = EpiSim(integration_config, temp_dir, instance_folder)

        # Setup with mocked Julia check
        with patch("shutil.which", return_value="/usr/bin/julia"):
            model.setup(executable_type="interpreter")

        # Run full simulation
        uuid_result, output = model.run_model()

        assert uuid_result == model.uuid
        assert "Simulation completed successfully" in output

        # Verify subprocess was called correctly
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "julia"
        assert "run.jl" in args[1]
        assert "run" in args
        assert "--config" in args
        assert "--data-folder" in args
        assert "--instance-folder" in args

    @pytest.mark.skip(
        reason="Step-by-step execution is experimental and untested - only single simulation runs are officially supported"
    )
    @patch("subprocess.run")
    def test_episim_step_by_step_workflow(self, mock_run, integration_config, temp_dir):
        """Test step-by-step simulation workflow - EXPERIMENTAL FEATURE"""
        # NOTE: Step-by-step execution is an experimental feature for RL agents
        # and is not officially supported. Only single simulation runs are tested and supported.
        pass

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

    def test_error_handling_workflow(self, integration_config, temp_dir):
        """Test error handling in various scenarios"""
        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        # Test initialization errors
        with pytest.raises(AssertionError):
            EpiSim(integration_config, "/nonexistent", instance_folder)

        with pytest.raises(AssertionError):
            EpiSim(integration_config, temp_dir, "/nonexistent")

        # Test setup errors
        model = EpiSim(integration_config, temp_dir, instance_folder)

        with pytest.raises(ValueError):
            model.setup(executable_type="invalid")

        # Test operation before setup
        with pytest.raises(RuntimeError):
            model.run_model()

        # Test configuration validation errors
        bad_config = integration_config.copy()
        del bad_config["simulation"]["engine"]

        config_obj = EpiSimConfig(bad_config)
        with pytest.raises(ValueError):
            config_obj.validate(verbose=False)

    @pytest.mark.skipif(not shutil.which("julia"), reason="Julia not available")
    def test_real_julia_check(self):
        """Test that Julia is actually available (only runs if Julia is installed)"""
        # This test will be skipped if Julia is not available
        # but will run if Julia is present to verify the environment
        result = shutil.which("julia")
        assert result is not None
        assert os.path.isfile(result)
        assert os.access(result, os.X_OK)

    def test_metapopulation_episim_integration(
        self, test_metapopulation_csv, integration_config, temp_dir
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

        assert list(pop_array.coords["G"]) == expected_groups
        assert pop_array.shape[1] == len(expected_groups)  # Age groups dimension

        # Update config to use metapopulation file
        integration_config["data"]["metapopulation_data_filename"] = str(
            test_metapopulation_csv
        )

        # Initialize EpiSim with updated config
        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        model = EpiSim(integration_config, temp_dir, instance_folder)

        # Verify config was saved correctly
        with open(model.config_path, "r") as f:
            saved_config = json.load(f)

        assert "metapopulation_data_filename" in saved_config["data"]
