"""
Tests for EpiSimConfig class
"""

import json
import os

import pytest

from episim_python import EpiSimConfig


class TestEpiSimConfig:
    """Test cases for EpiSimConfig class"""

    def test_init_with_dict(self, minimal_config):
        """Test initialization with configuration dictionary"""
        config = EpiSimConfig(minimal_config)
        assert config.group_labels == ["Y", "M", "O"]
        assert config.group_size == 3
        assert config.config["simulation"]["engine"] == "MMCACovid19"

    def test_init_from_json(self, test_config_json):
        """Test initialization from JSON file"""
        config = EpiSimConfig.from_json(str(test_config_json))
        assert config.group_labels == ["Y", "M", "O"]
        assert config.group_size == 3
        assert config.config["simulation"]["engine"] == "MMCACovid19"

    def test_validation_success(self, minimal_config):
        """Test successful validation"""
        config = EpiSimConfig(minimal_config)
        # Should not raise exception
        config.validate(verbose=False)

    def test_validation_missing_section(self, minimal_config):
        """Test validation with missing section"""
        bad_config = minimal_config.copy()
        del bad_config["simulation"]
        config = EpiSimConfig(bad_config)
        with pytest.raises(ValueError, match="Configuration validation failed"):
            config.validate(verbose=False)

    def test_validation_missing_key(self, minimal_config):
        """Test validation with missing key"""
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

    def test_detect_group_params(self, minimal_config):
        """Test group parameter detection"""
        config = EpiSimConfig(minimal_config)
        assert config.is_group_param("epidemic_params.ηᵍ")
        assert config.is_group_param("epidemic_params.αᵍ")
        assert config.is_group_param("population_params.kᵍ")
        assert not config.is_group_param("epidemic_params.βᴵ")
        assert not config.is_group_param("simulation.engine")

    def test_update_scalar_param(self, minimal_config):
        """Test updating scalar parameter"""
        config = EpiSimConfig(minimal_config)
        config.update_param("epidemic_params.βᴵ", 0.15)
        assert config.get_param("epidemic_params.βᴵ") == 0.15

    def test_update_group_param(self, minimal_config):
        """Test updating group parameter"""
        config = EpiSimConfig(minimal_config)
        new_values = [0.4, 0.4, 0.4]
        config.update_param("epidemic_params.ηᵍ", new_values)
        assert config.get_param("epidemic_params.ηᵍ") == new_values

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

    def test_update_group_param_by_label(self, minimal_config):
        """Test updating individual group parameter by label"""
        config = EpiSimConfig(minimal_config)
        config.update_group_param("epidemic_params.ηᵍ", "M", 0.5)
        values = config.get_param("epidemic_params.ηᵍ")
        assert values[1] == 0.5  # Middle group
        assert values[0] == 0.3  # Other groups unchanged
        assert values[2] == 0.3

    def test_get_group_param_by_label(self, minimal_config):
        """Test getting individual group parameter by label"""
        config = EpiSimConfig(minimal_config)
        value = config.get_group_param("epidemic_params.ηᵍ", "M")
        assert value == 0.3

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

    def test_inject_multiple_params(self, minimal_config):
        """Test injecting multiple parameters at once"""
        config = EpiSimConfig(minimal_config)
        updates = {
            "epidemic_params.βᴵ": 0.12,
            "epidemic_params.ηᵍ": [0.4, 0.4, 0.4],
            "simulation.start_date": "2020-02-01",
        }
        config.inject(updates)
        assert config.get_param("epidemic_params.βᴵ") == 0.12
        assert config.get_param("epidemic_params.ηᵍ") == [0.4, 0.4, 0.4]
        assert config.get_param("simulation.start_date") == "2020-02-01"

    def test_inject_group_vector(self, minimal_config):
        """Test injecting group vector by labels"""
        config = EpiSimConfig(minimal_config)
        updates = {"Y": 0.35, "O": 0.45}
        config.inject_group_vector("epidemic_params.ηᵍ", updates)
        values = config.get_param("epidemic_params.ηᵍ")
        assert values[0] == 0.35  # Y
        assert values[1] == 0.3  # M unchanged
        assert values[2] == 0.45  # O

    def test_reset_config(self, minimal_config):
        """Test resetting configuration to original state"""
        config = EpiSimConfig(minimal_config)
        original_beta = config.get_param("epidemic_params.βᴵ")

        # Modify config
        config.update_param("epidemic_params.βᴵ", 0.15)
        assert config.get_param("epidemic_params.βᴵ") == 0.15

        # Reset
        config.reset()
        assert config.get_param("epidemic_params.βᴵ") == original_beta

    def test_to_json(self, minimal_config, temp_dir):
        """Test saving configuration to JSON file"""
        config = EpiSimConfig(minimal_config)
        config.update_param("epidemic_params.βᴵ", 0.15)

        output_path = os.path.join(temp_dir, "output_config.json")
        config.to_json(output_path)

        # Verify file was created and contains correct data
        assert os.path.exists(output_path)
        with open(output_path) as f:
            saved_config = json.load(f)
        assert saved_config["epidemic_params"]["βᴵ"] == 0.15

    def test_nested_parameter_access(self, minimal_config):
        """Test accessing deeply nested parameters"""
        config = EpiSimConfig(minimal_config)

        # Test matrix parameter
        matrix = config.get_param("population_params.C")
        assert len(matrix) == 3
        assert len(matrix[0]) == 3

        # Test updating nested parameter
        config.update_param("population_params.ξ", 0.02)
        assert config.get_param("population_params.ξ") == 0.02
