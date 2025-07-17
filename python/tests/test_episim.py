"""
Tests for EpiSim class
"""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from episim_python import EpiSim
from .conftest import BaseTestCase


class TestEpiSim(BaseTestCase):
    """Test cases for EpiSim class"""

    def test_init_with_dict_config(self, minimal_config, instance_folder, temp_dir):
        """Test initialization with dictionary configuration"""
        model = EpiSim(minimal_config, temp_dir, instance_folder)

        self.assertions.assert_file_exists(model.model_state_folder)
        self.helpers.assert_episim_model_structure(model)
        assert model.instance_folder == instance_folder
        assert model.data_folder == temp_dir
        assert not model.setup_complete

    def test_init_with_json_config(self, test_config_json, instance_folder, temp_dir):
        """Test initialization with JSON configuration file"""
        model = EpiSim(str(test_config_json), temp_dir, instance_folder)

        self.helpers.assert_episim_model_structure(model)
        assert model.instance_folder == instance_folder
        assert model.data_folder == temp_dir
        
        # Check that config was copied to model folder
        self.assertions.assert_config_saved_correctly(
            model.config_path, 
            {"simulation.engine": "MMCACovid19"}
        )

    def test_init_with_initial_conditions(self, minimal_config, instance_folder, temp_dir, dummy_initial_conditions):
        """Test initialization with initial conditions file"""
        model = EpiSim(minimal_config, temp_dir, instance_folder, dummy_initial_conditions)

        assert model.model_state is not None
        self.assertions.assert_file_exists(model.model_state)
        assert os.path.basename(model.model_state) == "initial.nc"

    def test_setup_interpreter_mode(self, basic_episim_model, mock_julia_available):
        """Test setup with interpreter mode"""
        basic_episim_model.setup(executable_type="interpreter")

        assert basic_episim_model.setup_complete
        assert basic_episim_model.executable_type == "interpreter"
        assert basic_episim_model.executable_path[0] == "julia"
        assert basic_episim_model.executable_path[1].endswith("run.jl")

    def test_setup_compiled_mode(self, basic_episim_model, dummy_executable):
        """Test setup with compiled mode"""
        basic_episim_model.setup(executable_type="compiled", executable_path=dummy_executable)

        assert basic_episim_model.setup_complete
        assert basic_episim_model.executable_type == "compiled"
        assert basic_episim_model.executable_path == [dummy_executable]

    def test_set_backend_engine(self, basic_episim_model):
        """Test setting backend engine"""
        basic_episim_model.set_backend_engine("MMCACovid19")
        assert basic_episim_model.backend_engine == "MMCACovid19"

    def test_update_config(self, basic_episim_model, minimal_config):
        """Test updating configuration"""
        new_config = minimal_config.copy()
        new_config["simulation"]["start_date"] = "2020-02-01"

        old_config_path = basic_episim_model.config_path
        basic_episim_model.update_config(new_config)

        # Config path should change (new file created)
        assert (
            basic_episim_model.config_path != old_config_path
            or os.path.basename(basic_episim_model.config_path) == "config_auto_py.json"
        )

        # New config should be saved
        self.assertions.assert_config_saved_correctly(
            basic_episim_model.config_path,
            {"simulation.start_date": "2020-02-01"}
        )

    def test_model_state_filename(self, basic_episim_model):
        """Test model state filename generation"""
        filename = basic_episim_model.model_state_filename("2020-01-15")
        expected = os.path.join(
            basic_episim_model.model_state_folder,
            "output",
            "compartments_t_2020-01-15.nc",
        )
        assert filename == expected

    def test_update_model_state(self, basic_episim_model):
        """Test updating model state"""
        result = basic_episim_model.update_model_state("2020-01-15")
        assert result is basic_episim_model  # Should return self for chaining

        expected = os.path.join(
            basic_episim_model.model_state_folder,
            "output",
            "compartments_t_2020-01-15.nc",
        )
        assert basic_episim_model.model_state == expected

    def test_run_model_success(self, episim_model_with_interpreter, mock_subprocess_run):
        """Test successful model execution"""
        uuid_result, stdout = episim_model_with_interpreter.run_model()

        assert uuid_result == episim_model_with_interpreter.uuid
        assert stdout == "Simulation completed successfully"

        # Check that subprocess was called with correct arguments
        self.assertions.assert_subprocess_called_correctly(
            mock_subprocess_run,
            ["julia", "run", "--config", "--data-folder", "--instance-folder"],
            ["run.jl"]
        )


    def test_run_model_with_override_config(self, episim_model_with_interpreter, mock_subprocess_run):
        """Test model execution with override configuration"""
        override_config = {
            "start_date": "2020-02-01",
            "end_date": "2020-02-10",
            "save_time_step": 5,
        }

        episim_model_with_interpreter.run_model(override_config=override_config)

        # Check that override parameters were added to command
        self.assertions.assert_subprocess_called_correctly(
            mock_subprocess_run,
            ["--start-date", "2020-02-01", "--end-date", "2020-02-10", "--export-compartments-time-t", "5"]
        )

    @pytest.mark.skip(
        reason="Step-by-step execution is experimental and untested - only single simulation runs are officially supported",
    )
    def test_step_method(self):
        """Test step-by-step execution - EXPERIMENTAL FEATURE"""
        # NOTE: Step-by-step execution is an experimental feature for RL agents
        # and is not officially supported. Only single simulation runs are tested and supported.

    def test_handle_config_input_dict(self, minimal_config, temp_dir):
        """Test config input handling with dictionary"""
        config_path = EpiSim.handle_config_input(temp_dir, minimal_config)

        self.assertions.assert_file_exists(config_path)
        assert config_path.endswith("config_auto_py.json")
        
        self.assertions.assert_config_saved_correctly(
            config_path,
            {"simulation.engine": "MMCACovid19"}
        )

    def test_handle_config_input_file(self, test_config_json, temp_dir):
        """Test config input handling with file path"""
        config_path = EpiSim.handle_config_input(temp_dir, str(test_config_json))

        self.assertions.assert_file_exists(config_path)
        assert os.path.basename(config_path) == "test_config.json"

