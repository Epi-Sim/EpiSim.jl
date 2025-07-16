"""
Tests for dynamic JSON schema validation
"""

import json
import os
from unittest.mock import patch

import pytest

# Try to import schema validation components
try:
    from episim_python.schema_validator import (
        EpiSimSchemaValidator,
        SchemaValidator,
        validate_episim_config,
        validate_episim_config_safe,
    )

    SCHEMA_AVAILABLE = True
except ImportError:
    SCHEMA_AVAILABLE = False

from episim_python import EpiSimConfig


@pytest.mark.skipif(not SCHEMA_AVAILABLE, reason="JSON schema validation not available")
class TestSchemaValidator:
    """Test the SchemaValidator class"""

    def test_schema_validator_initialization(self):
        """Test that SchemaValidator initializes correctly"""
        validator = SchemaValidator()
        assert validator.schema_template is not None
        assert "properties" in validator.schema_template

    def test_schema_validator_with_custom_template(self, temp_dir):
        """Test SchemaValidator with custom template path"""
        # Create a minimal template
        custom_template = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "test_array": {
                    "type": "array",
                    "minItems": "{GROUP_SIZE}",
                    "maxItems": "{GROUP_SIZE}",
                },
            },
        }

        template_path = os.path.join(temp_dir, "custom_template.json")
        with open(template_path, "w") as f:
            json.dump(custom_template, f)

        validator = SchemaValidator(template_path)
        assert validator.schema_template == custom_template

    def test_invalid_template_path(self):
        """Test that invalid template path raises error"""
        with pytest.raises(FileNotFoundError):
            SchemaValidator("/nonexistent/path.json")

    def test_generate_schema_different_group_sizes(self):
        """Test schema generation with different group sizes"""
        validator = SchemaValidator()

        # Test with group size 2
        schema_2 = validator.generate_schema(2)
        assert (
            schema_2["properties"]["epidemic_params"]["properties"]["ηᵍ"]["minItems"]
            == 2
        )
        assert (
            schema_2["properties"]["epidemic_params"]["properties"]["ηᵍ"]["maxItems"]
            == 2
        )

        # Test with group size 4
        schema_4 = validator.generate_schema(4)
        assert (
            schema_4["properties"]["epidemic_params"]["properties"]["ηᵍ"]["minItems"]
            == 4
        )
        assert (
            schema_4["properties"]["epidemic_params"]["properties"]["ηᵍ"]["maxItems"]
            == 4
        )

    def test_invalid_group_size(self):
        """Test that invalid group size raises error"""
        validator = SchemaValidator()

        with pytest.raises(ValueError):
            validator.generate_schema(0)

        with pytest.raises(ValueError):
            validator.generate_schema(-1)

    def test_extract_group_size(self):
        """Test group size extraction from config"""
        validator = SchemaValidator()

        config = {"population_params": {"G_labels": ["Y", "M", "O"]}}

        group_size = validator._extract_group_size(config)
        assert group_size == 3

    def test_extract_group_size_missing_field(self):
        """Test group size extraction with missing fields"""
        validator = SchemaValidator()

        # Missing population_params
        with pytest.raises(ValueError, match="Missing required field"):
            validator._extract_group_size({})

        # Missing G_labels
        config = {"population_params": {}}
        with pytest.raises(ValueError, match="Missing required field"):
            validator._extract_group_size(config)

    def test_extract_group_size_invalid_g_labels(self):
        """Test group size extraction with invalid G_labels"""
        validator = SchemaValidator()

        # Empty G_labels
        config = {"population_params": {"G_labels": []}}
        with pytest.raises(ValueError, match="must be a non-empty list"):
            validator._extract_group_size(config)

        # Non-list G_labels
        config = {"population_params": {"G_labels": "invalid"}}
        with pytest.raises(ValueError, match="must be a non-empty list"):
            validator._extract_group_size(config)


@pytest.mark.skipif(not SCHEMA_AVAILABLE, reason="JSON schema validation not available")
class TestEpiSimSchemaValidator:
    """Test the EpiSimSchemaValidator class"""

    def test_validate_valid_config(self, minimal_config):
        """Test validation of a valid configuration"""
        validator = EpiSimSchemaValidator()

        # Should not raise exception
        result = validator.validate_config(minimal_config, verbose=False)
        assert result is True

    def test_validate_config_safe_valid(self, minimal_config):
        """Test safe validation of a valid configuration"""
        validator = EpiSimSchemaValidator()

        is_valid, errors = validator.validate_config_safe(minimal_config, verbose=False)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_invalid_config(self, minimal_config):
        """Test validation of an invalid configuration"""
        validator = EpiSimSchemaValidator()

        # Make config invalid by removing required field
        invalid_config = minimal_config.copy()
        del invalid_config["simulation"]["engine"]

        with pytest.raises(ValueError):
            validator.validate_config(invalid_config, verbose=False)

    def test_validate_config_safe_invalid(self, minimal_config):
        """Test safe validation of an invalid configuration"""
        validator = EpiSimSchemaValidator()

        # Make config invalid by wrong group size
        invalid_config = minimal_config.copy()
        invalid_config["epidemic_params"]["ηᵍ"] = [
            0.3,
            0.3,
        ]  # Only 2 values instead of 3

        is_valid, errors = validator.validate_config_safe(invalid_config, verbose=False)
        assert is_valid is False
        assert len(errors) > 0

    def test_get_schema_for_config(self, minimal_config):
        """Test schema generation for a specific config"""
        validator = EpiSimSchemaValidator()

        schema = validator.get_schema_for_config(minimal_config)

        assert "properties" in schema
        assert (
            schema["properties"]["epidemic_params"]["properties"]["ηᵍ"]["minItems"] == 3
        )
        assert (
            schema["properties"]["epidemic_params"]["properties"]["ηᵍ"]["maxItems"] == 3
        )


@pytest.mark.skipif(not SCHEMA_AVAILABLE, reason="JSON schema validation not available")
class TestConvenienceFunctions:
    """Test convenience functions for schema validation"""

    def test_validate_episim_config_valid(self, minimal_config):
        """Test validate_episim_config with valid config"""
        result = validate_episim_config(minimal_config, verbose=False)
        assert result is True

    def test_validate_episim_config_invalid(self, minimal_config):
        """Test validate_episim_config with invalid config"""
        invalid_config = minimal_config.copy()
        del invalid_config["simulation"]

        with pytest.raises(ValueError):
            validate_episim_config(invalid_config, verbose=False)

    def test_validate_episim_config_safe_valid(self, minimal_config):
        """Test validate_episim_config_safe with valid config"""
        is_valid, errors = validate_episim_config_safe(minimal_config, verbose=False)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_episim_config_safe_invalid(self, minimal_config):
        """Test validate_episim_config_safe with invalid config"""
        invalid_config = minimal_config.copy()
        invalid_config["epidemic_params"]["ηᵍ"] = [0.3]  # Wrong size

        is_valid, errors = validate_episim_config_safe(invalid_config, verbose=False)
        assert is_valid is False
        assert len(errors) > 0


@pytest.mark.skipif(not SCHEMA_AVAILABLE, reason="JSON schema validation not available")
class TestDynamicGroupSizes:
    """Test schema validation with different group sizes"""

    def test_two_group_config(self):
        """Test validation with 2 age groups"""
        config = {
            "simulation": {
                "engine": "MMCACovid19",
                "start_date": "2020-01-01",
                "end_date": "2020-01-10",
            },
            "data": {
                "initial_condition_filename": "test.nc",
                "metapopulation_data_filename": "test.csv",
                "mobility_matrix_filename": "test.csv",
            },
            "epidemic_params": {
                "βᴵ": 0.09,
                "ηᵍ": [0.3, 0.3],  # 2 groups
                "αᵍ": [0.25, 0.6],  # 2 groups
                "μᵍ": [1.0, 0.3],  # 2 groups
                "γᵍ": [0.003, 0.01],  # 2 groups
            },
            "population_params": {
                "G_labels": ["Y", "O"],  # 2 groups
                "C": [[0.6, 0.4], [0.25, 0.7]],  # 2x2 matrix
                "kᵍ": [12.0, 7.0],  # 2 groups
                "kᵍ_h": [3.0, 3.0],  # 2 groups
                "kᵍ_w": [2.0, 0.0],  # 2 groups
                "pᵍ": [0.0, 1.0],  # 2 groups
            },
            "NPI": {
                "κ₀s": [0.8],
                "ϕs": [0.2],
                "δs": [0.8],
                "tᶜs": [50],
                "are_there_npi": True,
            },
        }

        validator = EpiSimSchemaValidator()
        result = validator.validate_config(config, verbose=False)
        assert result is True

    def test_four_group_config(self):
        """Test validation with 4 age groups"""
        config = {
            "simulation": {
                "engine": "MMCACovid19",
                "start_date": "2020-01-01",
                "end_date": "2020-01-10",
            },
            "data": {
                "initial_condition_filename": "test.nc",
                "metapopulation_data_filename": "test.csv",
                "mobility_matrix_filename": "test.csv",
            },
            "epidemic_params": {
                "βᴵ": 0.09,
                "ηᵍ": [0.3, 0.3, 0.3, 0.3],  # 4 groups
                "αᵍ": [0.25, 0.6, 0.6, 0.7],  # 4 groups
                "μᵍ": [1.0, 0.3, 0.3, 0.2],  # 4 groups
                "γᵍ": [0.003, 0.01, 0.08, 0.15],  # 4 groups
            },
            "population_params": {
                "G_labels": ["C", "Y", "M", "O"],  # 4 groups
                "C": [
                    [0.6, 0.3, 0.08, 0.02],
                    [0.3, 0.5, 0.15, 0.05],
                    [0.08, 0.15, 0.6, 0.17],
                    [0.02, 0.05, 0.17, 0.76],
                ],  # 4x4 matrix
                "kᵍ": [8.0, 12.0, 13.0, 7.0],  # 4 groups
                "kᵍ_h": [2.0, 3.0, 3.0, 3.0],  # 4 groups
                "kᵍ_w": [0.0, 2.0, 5.0, 0.0],  # 4 groups
                "pᵍ": [0.0, 0.0, 1.0, 0.0],  # 4 groups
            },
            "NPI": {
                "κ₀s": [0.8],
                "ϕs": [0.2],
                "δs": [0.8],
                "tᶜs": [50],
                "are_there_npi": True,
            },
        }

        validator = EpiSimSchemaValidator()
        result = validator.validate_config(config, verbose=False)
        assert result is True

    def test_mismatched_group_sizes(self):
        """Test validation fails with mismatched group sizes"""
        config = {
            "simulation": {
                "engine": "MMCACovid19",
                "start_date": "2020-01-01",
                "end_date": "2020-01-10",
            },
            "data": {
                "initial_condition_filename": "test.nc",
                "metapopulation_data_filename": "test.csv",
                "mobility_matrix_filename": "test.csv",
            },
            "epidemic_params": {
                "βᴵ": 0.09,
                "ηᵍ": [0.3, 0.3, 0.3],  # 3 values
                "αᵍ": [0.25, 0.6],  # 2 values - MISMATCH!
            },
            "population_params": {
                "G_labels": ["Y", "M", "O"],  # 3 groups
                "C": [[0.6, 0.4, 0.02], [0.25, 0.7, 0.05], [0.2, 0.55, 0.25]],
                "kᵍ": [12.0, 13.0, 7.0],
                "kᵍ_h": [3.0, 3.0, 3.0],
                "kᵍ_w": [2.0, 5.0, 0.0],
                "pᵍ": [0.0, 1.0, 0.0],
            },
            "NPI": {
                "κ₀s": [0.8],
                "ϕs": [0.2],
                "δs": [0.8],
                "tᶜs": [50],
                "are_there_npi": True,
            },
        }

        validator = EpiSimSchemaValidator()
        is_valid, errors = validator.validate_config_safe(config, verbose=False)
        assert is_valid is False
        assert len(errors) > 0


@pytest.mark.skipif(not SCHEMA_AVAILABLE, reason="JSON schema validation not available")
class TestEpiSimConfigIntegration:
    """Test integration with EpiSimConfig class"""

    def test_episim_config_with_schema_validation(self, minimal_config):
        """Test EpiSimConfig validation with schema enabled"""
        config = EpiSimConfig(minimal_config)

        # Should not raise exception
        config.validate(verbose=False, use_schema=True)

    def test_episim_config_without_schema_validation(self, minimal_config):
        """Test EpiSimConfig validation with schema disabled"""
        config = EpiSimConfig(minimal_config)

        # Should not raise exception
        config.validate(verbose=False, use_schema=False)

    def test_episim_config_invalid_with_schema(self, minimal_config):
        """Test EpiSimConfig validation fails with invalid config and schema"""
        invalid_config = minimal_config.copy()
        invalid_config["epidemic_params"]["ηᵍ"] = [0.3, 0.3]  # Wrong size

        config = EpiSimConfig(invalid_config)

        with pytest.raises(ValueError):
            config.validate(verbose=False, use_schema=True)

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


@pytest.mark.skipif(not SCHEMA_AVAILABLE, reason="JSON schema validation not available")
class TestSchemaValidationEdgeCases:
    """Test edge cases and error conditions"""

    def test_schema_with_additional_properties(self, minimal_config):
        """Test that additional properties are not allowed"""
        config_with_extra = minimal_config.copy()
        config_with_extra["custom_section"] = {"custom_param": 123}
        config_with_extra["epidemic_params"]["custom_param"] = 456

        validator = EpiSimSchemaValidator()
        with pytest.raises(ValueError, match="Additional properties are not allowed"):
            validator.validate_config(config_with_extra, verbose=False)

    def test_missing_optional_sections(self):
        """Test configuration with only required sections"""
        minimal_required_config = {
            "simulation": {
                "engine": "MMCACovid19",
                "start_date": "2020-01-01",
                "end_date": "2020-01-10",
            },
            "data": {
                "initial_condition_filename": "test.nc",
                "metapopulation_data_filename": "test.csv",
                "mobility_matrix_filename": "test.csv",
            },
            "epidemic_params": {},
            "population_params": {"G_labels": ["Y", "M", "O"]},
        }

        validator = EpiSimSchemaValidator()
        result = validator.validate_config(minimal_required_config, verbose=False)
        assert result is True

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
