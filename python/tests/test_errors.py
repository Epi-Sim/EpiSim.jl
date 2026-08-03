"""
Centralized error handling tests for episim_python package

This module consolidates all error handling test cases from across the test suite
to provide a single location for testing error conditions and exception handling.
"""

import os
from unittest.mock import patch

import pytest

from episim_python import EpiSim, EpiSimConfig

from .conftest import BaseTestCase

# Try to import schema validation components for schema-related error tests
try:
    from episim_python.schema_validator import EpiSimSchemaValidator, SchemaValidator

    SCHEMA_AVAILABLE = True
except ImportError:
    SCHEMA_AVAILABLE = False


class TestEpiSimInitializationErrors(BaseTestCase):
    """Test EpiSim initialization error conditions"""

    def test_init_nonexistent_data_folder(self, minimal_config, instance_folder):
        """Test initialization with nonexistent data folder"""
        with pytest.raises(AssertionError):
            EpiSim(minimal_config, "/nonexistent", instance_folder)

    def test_init_nonexistent_instance_folder(self, minimal_config, temp_dir):
        """Test initialization with nonexistent instance folder"""
        with pytest.raises(AssertionError):
            EpiSim(minimal_config, temp_dir, "/nonexistent")

    def test_invalid_config_input_string(self, temp_dir):
        """Test initialization with invalid config input (string instead of dict/path)"""
        with pytest.raises(ValueError, match="Invalid config"):
            EpiSim.handle_config_input(temp_dir, "invalid_string")

    def test_invalid_config_input_none(self, temp_dir):
        """Test initialization with None config"""
        with pytest.raises(ValueError, match="Invalid config"):
            EpiSim.handle_config_input(temp_dir, None)


class TestEpiSimSetupErrors(BaseTestCase):
    """Test EpiSim setup error conditions"""

    def test_setup_invalid_executable_type(self, basic_episim_model):
        """Test setup with invalid executable type"""
        with pytest.raises(ValueError, match="executable_type must be"):
            basic_episim_model.setup(executable_type="invalid")

    def test_setup_missing_julia(self, basic_episim_model, mock_julia_unavailable):
        """Test setup when Julia interpreter is not available"""
        with pytest.raises(AssertionError, match="Julia interpreter not found"):
            basic_episim_model.setup(executable_type="interpreter")

    @pytest.mark.skip(
        reason="Test behavior differs between CI (compiled) and local (wrapper script) environments"
    )
    def test_setup_missing_compiled_executable(self, basic_episim_model):
        """Test setup when compiled executable is missing - SKIPPED due to environment differences"""
        with pytest.raises(AssertionError):
            basic_episim_model.setup(executable_type="compiled")

    def test_compiled_executable_symlink_validation(self):
        """Test that if episim executable exists as symlink, its target is valid"""
        from episim_python.epi_sim import EpiSim

        executable_path = EpiSim.get_executable_path()

        # Only run this test if the executable exists
        if executable_path and os.path.exists(executable_path):
            # If it's a symlink (indicating compilation occurred), validate the target
            if os.path.islink(executable_path):
                target_path = os.readlink(executable_path)
                # Target should exist (either absolute or relative to executable location)
                if not os.path.isabs(target_path):
                    target_path = os.path.join(
                        os.path.dirname(executable_path), target_path
                    )
                assert os.path.exists(target_path), (
                    f"Symlink target {target_path} does not exist"
                )
                assert os.access(target_path, os.X_OK), (
                    f"Symlink target {target_path} is not executable"
                )
            else:
                # If it's not a symlink, it should still be executable (wrapper script)
                assert os.access(executable_path, os.X_OK), (
                    f"Executable {executable_path} is not executable"
                )

    def test_operation_before_setup(self, basic_episim_model):
        """Test operations before setup is called"""
        with pytest.raises(RuntimeError, match="EpiSim not set up"):
            basic_episim_model.run_model()

    def test_invalid_backend_engine(self, episim_model_with_interpreter):
        """Test setting invalid backend engine"""
        with pytest.raises(ValueError, match="Invalid backend engine"):
            episim_model_with_interpreter.set_backend_engine("InvalidEngine")


class TestEpiSimRuntimeErrors(BaseTestCase):
    """Test EpiSim runtime error conditions"""

    def test_run_model_failure(
        self, episim_model_with_interpreter, mock_subprocess_run_failure
    ):
        """Test model run failure handling"""
        with pytest.raises(RuntimeError) as exc_info:
            episim_model_with_interpreter.run_model()

        error_message = str(exc_info.value)
        assert "Model execution failed with return code 1" in error_message
        assert "Simulation failed" in error_message
        assert "Some output" in error_message


class TestConfigurationValidationErrors(BaseTestCase):
    """Test configuration validation error conditions"""

    def test_validation_missing_section(self, minimal_config):
        """Test validation with missing required section"""
        bad_config = minimal_config.copy()
        del bad_config["simulation"]
        config = EpiSimConfig(bad_config)
        with pytest.raises(ValueError, match="Configuration validation failed"):
            config.validate(verbose=False)

    def test_validation_missing_key(self, minimal_config):
        """Test validation with missing required key"""
        import copy

        bad_config = copy.deepcopy(minimal_config)
        del bad_config["simulation"]["engine"]
        config = EpiSimConfig(bad_config)
        with pytest.raises(ValueError, match="Configuration validation failed"):
            config.validate(verbose=False)

    def test_validation_group_size_mismatch(self, minimal_config):
        """Test validation with group size mismatch"""
        import copy

        bad_config = copy.deepcopy(minimal_config)
        bad_config["epidemic_params"]["ηᵍ"] = [0.3, 0.3]  # Only 2 values instead of 3
        config = EpiSimConfig(bad_config)
        with pytest.raises(ValueError, match="Configuration validation failed"):
            config.validate(verbose=False)


class TestConfigurationParameterErrors(BaseTestCase):
    """Test configuration parameter manipulation error conditions"""

    def test_update_param_invalid_group_size(self, minimal_config):
        """Test updating group parameter with wrong size"""
        config = EpiSimConfig(minimal_config)
        with pytest.raises(ValueError, match="Expected a list of length 3"):
            config.update_param("epidemic_params.ηᵍ", [0.3, 0.3])

    def test_update_param_invalid_scalar(self, minimal_config):
        """Test updating scalar parameter with list"""
        config = EpiSimConfig(minimal_config)
        with pytest.raises(ValueError, match="Expected a scalar"):
            config.update_param("epidemic_params.βᴵ", [0.1, 0.2])

    def test_update_group_param_invalid_label(self, minimal_config):
        """Test updating group parameter with invalid label"""
        config = EpiSimConfig(minimal_config)
        with pytest.raises(ValueError, match="Group label 'X' not in G_labels"):
            config.update_group_param("epidemic_params.ηᵍ", "X", 0.5)

    def test_update_group_param_non_group_param(self, minimal_config):
        """Test updating non-group parameter with group method"""
        config = EpiSimConfig(minimal_config)
        with pytest.raises(ValueError, match="not detected as group-dependent"):
            config.update_group_param("epidemic_params.βᴵ", "Y", 0.5)


@pytest.mark.skipif(not SCHEMA_AVAILABLE, reason="JSON schema validation not available")
class TestSchemaValidationErrors:
    """Test JSON schema validation error conditions"""

    def test_invalid_template_path(self):
        """Test that invalid template path raises error"""
        with pytest.raises(FileNotFoundError):
            SchemaValidator("/nonexistent/path.json")

    def test_invalid_group_size_zero(self):
        """Test that zero group size raises error"""
        validator = SchemaValidator()
        with pytest.raises(ValueError):
            validator.generate_schema(0)

    def test_invalid_group_size_negative(self):
        """Test that negative group size raises error"""
        validator = SchemaValidator()
        with pytest.raises(ValueError):
            validator.generate_schema(-1)

    def test_extract_group_size_missing_population_params(self):
        """Test group size extraction with missing population_params"""
        validator = SchemaValidator()
        with pytest.raises(ValueError, match="Missing required field"):
            validator._extract_group_size({})

    def test_extract_group_size_missing_g_labels(self):
        """Test group size extraction with missing G_labels"""
        validator = SchemaValidator()
        config = {"population_params": {}}
        with pytest.raises(ValueError, match="Missing required field"):
            validator._extract_group_size(config)

    def test_extract_group_size_empty_g_labels(self):
        """Test group size extraction with empty G_labels"""
        validator = SchemaValidator()
        config = {"population_params": {"G_labels": []}}
        with pytest.raises(ValueError, match="must be a non-empty list"):
            validator._extract_group_size(config)

    def test_extract_group_size_invalid_g_labels_type(self):
        """Test group size extraction with invalid G_labels type"""
        validator = SchemaValidator()
        config = {"population_params": {"G_labels": "invalid"}}
        with pytest.raises(ValueError, match="must be a non-empty list"):
            validator._extract_group_size(config)

    def test_validate_invalid_config_missing_section(self, minimal_config):
        """Test schema validation with invalid config (missing section)"""
        validator = EpiSimSchemaValidator()
        invalid_config = minimal_config.copy()
        del invalid_config["simulation"]["engine"]

        with pytest.raises(ValueError):
            validator.validate_config(invalid_config, verbose=False)

    def test_validate_invalid_config_wrong_group_size(self, minimal_config):
        """Test schema validation with wrong group size"""
        validator = EpiSimSchemaValidator()
        invalid_config = minimal_config.copy()
        invalid_config["epidemic_params"]["ηᵍ"] = [0.3, 0.3]  # Wrong size

        with pytest.raises(ValueError):
            validator.validate_config(invalid_config, verbose=False)

    def test_schema_additional_properties_not_allowed(self, minimal_config):
        """Test that additional properties are not allowed"""
        config_with_extra = minimal_config.copy()
        config_with_extra["custom_section"] = {"custom_param": 123}
        config_with_extra["epidemic_params"]["custom_param"] = 456

        validator = EpiSimSchemaValidator()
        with pytest.raises(ValueError, match="Additional properties are not allowed"):
            validator.validate_config(config_with_extra, verbose=False)

    def test_invalid_date_format(self, minimal_config):
        """Test validation fails with invalid date format"""
        invalid_config = minimal_config.copy()
        invalid_config["simulation"]["start_date"] = "2020/01/01"  # Wrong format

        validator = EpiSimSchemaValidator()
        is_valid, errors = validator.validate_config_safe(invalid_config, verbose=False)
        assert is_valid is False
        assert any("start_date" in err for err in errors)

    def test_negative_parameter_values(self, minimal_config):
        """Test validation fails with negative parameter values"""
        invalid_config = minimal_config.copy()
        invalid_config["epidemic_params"]["βᴵ"] = -0.1  # Negative value

        validator = EpiSimSchemaValidator()
        is_valid, errors = validator.validate_config_safe(invalid_config, verbose=False)
        assert is_valid is False


class TestMetapopulationErrors(BaseTestCase):
    """Test metapopulation-related error conditions"""

    def test_metapopulation_missing_required_columns(self, temp_dir):
        """Test metapopulation loading with missing required columns"""
        # Create CSV with id column but missing other required columns
        csv_path = os.path.join(temp_dir, "bad_metapop.csv")
        with open(csv_path, "w") as f:
            f.write("id,wrong_column\n")
            f.write("1,value1\n")

        from episim_python.episim_utils import Metapopulation

        with pytest.raises(KeyError):  # Should raise KeyError for missing columns
            Metapopulation(csv_path)

    def test_metapopulation_nonexistent_file(self):
        """Test metapopulation loading with nonexistent file"""
        from episim_python.episim_utils import Metapopulation

        with pytest.raises(FileNotFoundError):
            Metapopulation("/nonexistent/file.csv")


@pytest.mark.skipif(not SCHEMA_AVAILABLE, reason="JSON schema validation not available")
class TestSchemaWarnings(BaseTestCase):
    """Test schema validation warning conditions"""

    @patch("episim_python.episim_utils.SCHEMA_VALIDATION_AVAILABLE", False)
    def test_episim_config_schema_unavailable_warning(self, minimal_config, capsys):
        """Test warning when schema validation is requested but unavailable"""
        config = EpiSimConfig(minimal_config)

        # Should not raise exception but should print warning
        config.validate(verbose=True, use_schema=True)

        captured = capsys.readouterr()
        assert (
            "Warning: JSON schema validation requested but jsonschema package not available"
            in captured.out
        )
