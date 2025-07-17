"""
Tests for EpiSim class
"""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from episim_python import EpiSim


class TestEpiSim:
    """Test cases for EpiSim class"""

    def test_init_with_dict_config(self, minimal_config, temp_dir):
        """Test initialization with dictionary configuration"""
        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        model = EpiSim(minimal_config, temp_dir, instance_folder)

        assert model.instance_folder == instance_folder
        assert model.data_folder == temp_dir
        assert len(model.uuid) > 0
        assert os.path.exists(model.model_state_folder)
        assert model.backend_engine == "MMCACovid19Vac"
        assert not model.setup_complete

    def test_init_with_json_config(self, test_config_json, temp_dir):
        """Test initialization with JSON configuration file"""
        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        model = EpiSim(str(test_config_json), temp_dir, instance_folder)

        assert model.instance_folder == instance_folder
        assert model.data_folder == temp_dir
        assert os.path.exists(model.config_path)

        # Check that config was copied to model folder
        with open(model.config_path) as f:
            config = json.load(f)
        assert config["simulation"]["engine"] == "MMCACovid19"

    def test_init_with_initial_conditions(self, minimal_config, temp_dir):
        """Test initialization with initial conditions file"""
        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        # Create dummy initial conditions file
        initial_conditions = os.path.join(temp_dir, "initial.nc")
        Path(initial_conditions).write_text("dummy nc content")

        model = EpiSim(minimal_config, temp_dir, instance_folder, initial_conditions)

        assert model.model_state is not None
        assert os.path.exists(model.model_state)
        assert os.path.basename(model.model_state) == "initial.nc"

    def test_init_nonexistent_folders(self, minimal_config):
        """Test initialization with nonexistent folders"""
        with pytest.raises(AssertionError):
            EpiSim(minimal_config, "/nonexistent", "/also_nonexistent")

    def test_setup_interpreter_mode(self, minimal_config, temp_dir):
        """Test setup with interpreter mode"""
        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        model = EpiSim(minimal_config, temp_dir, instance_folder)

        with patch("shutil.which", return_value="/usr/bin/julia"):
            model.setup(executable_type="interpreter")

        assert model.setup_complete
        assert model.executable_type == "interpreter"
        assert model.executable_path[0] == "julia"
        assert model.executable_path[1].endswith("run.jl")

    def test_setup_compiled_mode(self, minimal_config, temp_dir):
        """Test setup with compiled mode"""
        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        # Create dummy executable
        executable_path = os.path.join(temp_dir, "episim")
        Path(executable_path).write_text("#!/bin/bash\necho 'compiled episim'")
        os.chmod(executable_path, 0o755)

        model = EpiSim(minimal_config, temp_dir, instance_folder)
        model.setup(executable_type="compiled", executable_path=executable_path)

        assert model.setup_complete
        assert model.executable_type == "compiled"
        assert model.executable_path == [executable_path]

    def test_setup_invalid_executable_type(self, minimal_config, temp_dir):
        """Test setup with invalid executable type"""
        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        model = EpiSim(minimal_config, temp_dir, instance_folder)

        with pytest.raises(ValueError, match="executable_type must be"):
            model.setup(executable_type="invalid")

    def test_setup_missing_julia(self, minimal_config, temp_dir):
        """Test setup when Julia interpreter is not found"""
        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        model = EpiSim(minimal_config, temp_dir, instance_folder)

        with patch("shutil.which", return_value=None), pytest.raises(
            AssertionError, match="Julia interpreter not found"
        ):
            model.setup(executable_type="interpreter")

    def test_setup_missing_compiled_executable(self, minimal_config, temp_dir):
        """Test setup with missing compiled executable"""
        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        model = EpiSim(minimal_config, temp_dir, instance_folder)

        with pytest.raises(AssertionError):
            model.setup(
                executable_type="compiled",
                executable_path="/nonexistent/episim",
            )

    def test_check_setup_not_called(self, minimal_config, temp_dir):
        """Test that methods require setup to be called"""
        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        model = EpiSim(minimal_config, temp_dir, instance_folder)

        with pytest.raises(RuntimeError, match="EpiSim not set up"):
            model.run_model()

    def test_set_backend_engine(self, minimal_config, temp_dir):
        """Test setting backend engine"""
        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        model = EpiSim(minimal_config, temp_dir, instance_folder)

        model.set_backend_engine("MMCACovid19")
        assert model.backend_engine == "MMCACovid19"

        with pytest.raises(ValueError, match="Invalid backend engine"):
            model.set_backend_engine("InvalidEngine")

    def test_update_config(self, minimal_config, temp_dir):
        """Test updating configuration"""
        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        model = EpiSim(minimal_config, temp_dir, instance_folder)

        new_config = minimal_config.copy()
        new_config["simulation"]["start_date"] = "2020-02-01"

        old_config_path = model.config_path
        model.update_config(new_config)

        # Config path should change (new file created)
        assert (
            model.config_path != old_config_path
            or os.path.basename(model.config_path) == "config_auto_py.json"
        )

        # New config should be saved
        with open(model.config_path) as f:
            saved_config = json.load(f)
        assert saved_config["simulation"]["start_date"] == "2020-02-01"

    def test_model_state_filename(self, minimal_config, temp_dir):
        """Test model state filename generation"""
        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        model = EpiSim(minimal_config, temp_dir, instance_folder)

        filename = model.model_state_filename("2020-01-15")
        expected = os.path.join(
            model.model_state_folder,
            "output",
            "compartments_t_2020-01-15.nc",
        )
        assert filename == expected

    def test_update_model_state(self, minimal_config, temp_dir):
        """Test updating model state"""
        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        model = EpiSim(minimal_config, temp_dir, instance_folder)

        result = model.update_model_state("2020-01-15")
        assert result is model  # Should return self for chaining

        expected = os.path.join(
            model.model_state_folder,
            "output",
            "compartments_t_2020-01-15.nc",
        )
        assert model.model_state == expected

    @patch("subprocess.run")
    def test_run_model_success(self, mock_run, minimal_config, temp_dir):
        """Test successful model execution"""
        # Setup mock subprocess result
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Simulation completed successfully"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        model = EpiSim(minimal_config, temp_dir, instance_folder)

        with patch("shutil.which", return_value="/usr/bin/julia"):
            model.setup(executable_type="interpreter")

        uuid_result, stdout = model.run_model()

        assert uuid_result == model.uuid
        assert stdout == "Simulation completed successfully"

        # Check that subprocess was called with correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "julia"
        assert args[-1] == model.model_state_folder  # instance folder

    @patch("subprocess.run")
    def test_run_model_failure(self, mock_run, minimal_config, temp_dir):
        """Test model execution failure"""
        # Setup mock subprocess result for failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = "Some output"
        mock_result.stderr = "Error: Julia package not found"
        mock_run.return_value = mock_result

        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        model = EpiSim(minimal_config, temp_dir, instance_folder)

        with patch("shutil.which", return_value="/usr/bin/julia"):
            model.setup(executable_type="interpreter")

        with pytest.raises(RuntimeError) as exc_info:
            model.run_model()

        error_message = str(exc_info.value)
        assert "Model execution failed with return code 1" in error_message
        assert "Error: Julia package not found" in error_message
        assert "Some output" in error_message

    @patch("subprocess.run")
    def test_run_model_with_override_config(self, mock_run, minimal_config, temp_dir):
        """Test model execution with override configuration"""
        # Setup mock subprocess result
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        instance_folder = os.path.join(temp_dir, "instances")
        os.makedirs(instance_folder)

        model = EpiSim(minimal_config, temp_dir, instance_folder)

        with patch("shutil.which", return_value="/usr/bin/julia"):
            model.setup(executable_type="interpreter")

        override_config = {
            "start_date": "2020-02-01",
            "end_date": "2020-02-10",
            "save_time_step": 5,
        }

        model.run_model(override_config=override_config)

        # Check that override parameters were added to command
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]

        assert "--start-date" in args
        assert "2020-02-01" in args
        assert "--end-date" in args
        assert "2020-02-10" in args
        assert "--export-compartments-time-t" in args
        assert "5" in args

    @pytest.mark.skip(
        reason="Step-by-step execution is experimental and untested - only single simulation runs are officially supported",
    )
    @patch("subprocess.run")
    def test_step_method(self, mock_run, minimal_config, temp_dir):
        """Test step-by-step execution - EXPERIMENTAL FEATURE"""
        # NOTE: Step-by-step execution is an experimental feature for RL agents
        # and is not officially supported. Only single simulation runs are tested and supported.

    def test_handle_config_input_dict(self, minimal_config, temp_dir):
        """Test config input handling with dictionary"""
        config_path = EpiSim.handle_config_input(temp_dir, minimal_config)

        assert os.path.exists(config_path)
        assert config_path.endswith("config_auto_py.json")

        with open(config_path) as f:
            saved_config = json.load(f)
        assert saved_config["simulation"]["engine"] == "MMCACovid19"

    def test_handle_config_input_file(self, test_config_json, temp_dir):
        """Test config input handling with file path"""
        config_path = EpiSim.handle_config_input(temp_dir, str(test_config_json))

        assert os.path.exists(config_path)
        assert os.path.basename(config_path) == "test_config.json"

    def test_handle_config_input_invalid(self, temp_dir):
        """Test config input handling with invalid input"""
        with pytest.raises(ValueError, match="Invalid config"):
            EpiSim.handle_config_input(temp_dir, 123)  # Invalid type

        with pytest.raises(ValueError, match="Invalid config"):
            EpiSim.handle_config_input(
                temp_dir,
                "/nonexistent/config.json",
            )  # Missing file
