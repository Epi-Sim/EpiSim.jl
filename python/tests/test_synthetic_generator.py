"""Tests for synthetic_generator.py validation and parameter functions.

This module tests the core parameter generation and validation functionality
used in the synthetic data generation pipeline.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add python directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synthetic_generator import SyntheticDataGenerator


class TestGenerateParameterGrid:
    """Tests for generate_parameter_grid method."""

    @pytest.fixture
    def mock_generator(self, tmp_path):
        """Create a mock SyntheticDataGenerator."""
        data_folder = tmp_path / "data"
        data_folder.mkdir()
        output_folder = tmp_path / "output"
        output_folder.mkdir()

        config_path = data_folder / "config.json"
        config_data = {
            "simulation": {"start_date": "2020-03-01", "end_date": "2020-06-01"},
            "data": {
                "metapopulation_data_filename": "metapop.csv",
                "mobility_matrix_filename": "mobility.csv",
            },
            "epidemic_params": {
                "βᴵ": 0.5,
                "βᴬ": 0.25,
                "αᵍ": [0.1, 0.1, 0.1],
                "μᵍ": [0.2, 0.2, 0.2],
                "ηᵍ": [0.2, 0.2, 0.2],
            },
            "population_params": {"G_labels": ["Y", "M", "O"]},
            "NPI": {"κ₀s": [0.0], "tᶜs": [0]},
        }
        config_path.write_text(json.dumps(config_data))

        # Create required CSV files
        metapop_df = pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "total": [1000, 2000, 3000],
                "Y": [300, 600, 900],
                "M": [400, 800, 1200],
                "O": [300, 600, 900],
            }
        )
        metapop_df.to_csv(data_folder / "metapop.csv", index=False)

        mobility_df = pd.DataFrame(
            {
                "source_idx": [1, 1, 2, 2, 3, 3],
                "target_idx": [1, 2, 1, 2, 1, 3],
                "ratio": [0.7, 0.3, 0.4, 0.6, 0.5, 0.5],
            }
        )
        mobility_df.to_csv(data_folder / "mobility.csv", index=False)

        rosetta_df = pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "idx": [1, 2, 3],
            }
        )
        rosetta_df.to_csv(data_folder / "rosetta.csv", index=False)

        return SyntheticDataGenerator(
            str(config_path), str(data_folder), str(output_folder)
        )

    def test_generates_correct_number_of_profiles(self, mock_generator):
        """Should generate correct number of profiles."""
        profiles = mock_generator.generate_parameter_grid(n_profiles=10)
        assert len(profiles) == 10

    def test_profiles_have_required_fields(self, mock_generator):
        """Each profile should have required fields."""
        profiles = mock_generator.generate_parameter_grid(n_profiles=5)

        required_fields = [
            "profile_id",
            "r0_scale",
            "t_inf",
            "t_inc",
            "event_start",
            "event_duration",
            "affected_fraction",
            "ratio_beta_a",
            "alpha_scale",
            "mu_scale",
            "seed_size",
            "mobility_sigma_O",
            "mobility_sigma_D",
        ]

        for profile in profiles:
            for field in required_fields:
                assert field in profile, f"Missing field: {field}"

    def test_profile_ids_are_sequential(self, mock_generator):
        """Profile IDs should be sequential."""
        profiles = mock_generator.generate_parameter_grid(n_profiles=5)
        ids = [p["profile_id"] for p in profiles]
        assert ids == [0, 1, 2, 3, 4]

    def test_r0_scale_in_valid_range(self, mock_generator):
        """R0 scale should be in valid range [0.5, 3.0]."""
        profiles = mock_generator.generate_parameter_grid(n_profiles=20)

        for profile in profiles:
            assert 0.5 <= profile["r0_scale"] <= 3.0

    def test_t_inf_and_t_inc_in_valid_range(self, mock_generator):
        """T_inf and T_inc should be in valid range [2.0, 10.0]."""
        profiles = mock_generator.generate_parameter_grid(n_profiles=20)

        for profile in profiles:
            assert 2.0 <= profile["t_inf"] <= 10.0
            assert 2.0 <= profile["t_inc"] <= 10.0

    def test_t_inf_less_or_equal_t_inc(self, mock_generator):
        """T_inf should be <= T_inc for model stability."""
        profiles = mock_generator.generate_parameter_grid(n_profiles=50)

        for profile in profiles:
            assert profile["t_inf"] <= profile["t_inc"], (
                f"T_inf ({profile['t_inf']}) > T_inc ({profile['t_inc']})"
            )

    def test_event_duration_in_valid_range(self, mock_generator):
        """Event duration should be in valid range [7, 60]."""
        profiles = mock_generator.generate_parameter_grid(n_profiles=20)

        for profile in profiles:
            assert 7 <= profile["event_duration"] <= 60

    def test_seed_size_in_valid_range(self, mock_generator):
        """Seed size should be in valid range [10, 500]."""
        profiles = mock_generator.generate_parameter_grid(n_profiles=20)

        for profile in profiles:
            assert 10 <= profile["seed_size"] <= 500

    def test_reproducible_with_same_seed(self, mock_generator):
        """Same seed should produce same profiles."""
        profiles1 = mock_generator.generate_parameter_grid(n_profiles=10, seed=42)
        profiles2 = mock_generator.generate_parameter_grid(n_profiles=10, seed=42)

        for p1, p2 in zip(profiles1, profiles2):
            assert p1 == p2

    def test_different_seeds_produce_different_results(self, mock_generator):
        """Different seeds should produce different profiles."""
        profiles1 = mock_generator.generate_parameter_grid(n_profiles=10, seed=42)
        profiles2 = mock_generator.generate_parameter_grid(n_profiles=10, seed=123)

        # At least some profiles should be different
        any_different = any(p1 != p2 for p1, p2 in zip(profiles1, profiles2))
        assert any_different, "Different seeds should produce different results"

    def test_mobility_sigma_in_valid_range(self, mock_generator):
        """Mobility sigma should be in specified range."""
        mock_generator.mobility_sigma_min = 0.1
        mock_generator.mobility_sigma_max = 0.5

        profiles = mock_generator.generate_parameter_grid(n_profiles=20)

        for profile in profiles:
            assert 0.1 <= profile["mobility_sigma_O"] <= 0.5
            assert 0.1 <= profile["mobility_sigma_D"] <= 0.5


class TestValidateProfileParameters:
    """Tests for validate_profile_parameters method."""

    @pytest.fixture
    def mock_generator(self, tmp_path):
        """Create a mock SyntheticDataGenerator."""
        data_folder = tmp_path / "data"
        data_folder.mkdir()
        output_folder = tmp_path / "output"
        output_folder.mkdir()

        config_path = data_folder / "config.json"
        config_data = {
            "simulation": {},
            "data": {
                "metapopulation_data_filename": "metapop.csv",
                "mobility_matrix_filename": "mobility.csv",
            },
            "epidemic_params": {},
            "population_params": {"G_labels": ["Y", "M", "O"]},
            "NPI": {},
        }
        config_path.write_text(json.dumps(config_data))

        # Create required CSV files
        pd.DataFrame({"id": ["1"], "total": [1000]}).to_csv(
            data_folder / "metapop.csv", index=False
        )
        pd.DataFrame({"from": [1], "to": [1], "weight": [1.0]}).to_csv(
            data_folder / "mobility.csv", index=False
        )
        pd.DataFrame({"id": ["1"], "idx": [1]}).to_csv(
            data_folder / "rosetta.csv", index=False
        )

        return SyntheticDataGenerator(
            str(config_path), str(data_folder), str(output_folder)
        )

    def test_valid_profile_passes(self, mock_generator):
        """Valid profile should pass validation."""
        profile = {
            "profile_id": 0,
            "t_inf": 5.0,
            "t_inc": 7.0,
            "r0_scale": 2.0,
            "alpha_scale": 1.0,
        }
        assert mock_generator.validate_profile_parameters(profile) is True

    def test_invalid_mu_less_than_eta_fails(self, mock_generator):
        """Profile with mu < eta should fail validation."""
        profile = {
            "profile_id": 0,
            "t_inf": 8.0,  # mu = 0.125
            "t_inc": 4.0,  # eta = 0.25
            "r0_scale": 2.0,
            "alpha_scale": 1.0,
        }
        assert mock_generator.validate_profile_parameters(profile) is False

    def test_mu_greater_than_one_fails(self, mock_generator):
        """Profile with mu > 1.0 should fail validation."""
        profile = {
            "profile_id": 0,
            "t_inf": 0.5,  # mu = 2.0
            "t_inc": 1.0,
            "r0_scale": 2.0,
            "alpha_scale": 1.0,
        }
        assert mock_generator.validate_profile_parameters(profile) is False

    def test_eta_greater_than_one_fails(self, mock_generator):
        """Profile with eta > 1.0 should fail validation."""
        profile = {
            "profile_id": 0,
            "t_inf": 2.0,
            "t_inc": 0.5,  # eta = 2.0
            "r0_scale": 2.0,
            "alpha_scale": 1.0,
        }
        assert mock_generator.validate_profile_parameters(profile) is False

    def test_mu_too_small_fails(self, mock_generator):
        """Profile with mu < 0.01 should fail validation."""
        profile = {
            "profile_id": 0,
            "t_inf": 200.0,  # mu = 0.005
            "t_inc": 200.0,
            "r0_scale": 2.0,
            "alpha_scale": 1.0,
        }
        assert mock_generator.validate_profile_parameters(profile) is False

    def test_eta_too_small_fails(self, mock_generator):
        """Profile with eta < 0.01 should fail validation."""
        profile = {
            "profile_id": 0,
            "t_inf": 50.0,
            "t_inc": 200.0,  # eta = 0.005
            "r0_scale": 2.0,
            "alpha_scale": 1.0,
        }
        assert mock_generator.validate_profile_parameters(profile) is False

    def test_alpha_scale_too_large_fails(self, mock_generator):
        """Profile with alpha_scale > 1.5 should fail validation."""
        profile = {
            "profile_id": 0,
            "t_inf": 5.0,
            "t_inc": 7.0,
            "r0_scale": 2.0,
            "alpha_scale": 2.0,
        }
        assert mock_generator.validate_profile_parameters(profile) is False

    def test_r0_less_than_one_warns_but_passes(self, mock_generator, caplog):
        """Profile with R0 < 1 should warn but pass."""
        profile = {
            "profile_id": 0,
            "t_inf": 5.0,
            "t_inc": 7.0,
            "r0_scale": 0.5,
            "alpha_scale": 1.0,
        }
        result = mock_generator.validate_profile_parameters(profile)
        assert result is True


class TestPrepareKappa0File:
    """Tests for prepare_kappa0_file method."""

    @pytest.fixture
    def mock_generator(self, tmp_path):
        """Create a mock SyntheticDataGenerator."""
        data_folder = tmp_path / "data"
        data_folder.mkdir()
        output_folder = tmp_path / "output"
        output_folder.mkdir()

        config_path = data_folder / "config.json"
        config_data = {
            "simulation": {},
            "data": {
                "metapopulation_data_filename": "metapop.csv",
                "mobility_matrix_filename": "mobility.csv",
            },
            "epidemic_params": {},
            "population_params": {"G_labels": ["Y", "M", "O"]},
            "NPI": {},
        }
        config_path.write_text(json.dumps(config_data))

        pd.DataFrame({"id": ["1"], "total": [1000]}).to_csv(
            data_folder / "metapop.csv", index=False
        )
        pd.DataFrame({"from": [1], "to": [1], "weight": [1.0]}).to_csv(
            data_folder / "mobility.csv", index=False
        )
        pd.DataFrame({"id": ["1"], "idx": [1]}).to_csv(
            data_folder / "rosetta.csv", index=False
        )

        return SyntheticDataGenerator(
            str(config_path), str(data_folder), str(output_folder)
        )

    def test_creates_kappa0_csv(self, mock_generator):
        """Should create kappa0 CSV file."""
        profile = {
            "event_start": 10,
            "event_duration": 20,
        }

        filename, filepath = mock_generator.prepare_kappa0_file(
            "test_run", "Global_Timed", 0.5, profile, "2020-03-01", "2020-05-01"
        )

        assert filename == "kappa0.csv"
        assert os.path.exists(filepath)

    def test_baseline_has_zero_reduction(self, mock_generator):
        """Baseline scenario should have zero reduction."""
        profile = {}

        filename, filepath = mock_generator.prepare_kappa0_file(
            "test_run", "Baseline", 0.0, profile, "2020-03-01", "2020-03-10"
        )

        df = pd.read_csv(filepath)
        assert all(df["reduction"] == 0.0)

    def test_global_timed_applies_reduction(self, mock_generator):
        """Global_Timed should apply reduction during event window."""
        profile = {
            "event_start": 5,
            "event_duration": 3,
        }

        filename, filepath = mock_generator.prepare_kappa0_file(
            "test_run", "Global_Timed", 0.5, profile, "2020-03-01", "2020-03-15"
        )

        df = pd.read_csv(filepath)
        # Days 5-8 should have reduction of 0.5
        assert df.loc[5:7, "reduction"].iloc[0] == 0.5
        assert df.loc[5:7, "reduction"].iloc[1] == 0.5
        assert df.loc[5:7, "reduction"].iloc[2] == 0.5

    def test_respects_date_bounds(self, mock_generator):
        """Should respect start and end date bounds."""
        profile = {
            "event_start": 50,  # Beyond date range
            "event_duration": 10,
        }

        filename, filepath = mock_generator.prepare_kappa0_file(
            "test_run", "Global_Timed", 0.5, profile, "2020-03-01", "2020-03-10"
        )

        df = pd.read_csv(filepath)
        # All should be zero since event_start is beyond range
        assert all(df["reduction"] == 0.0)


class TestPrepareSeedFile:
    """Tests for prepare_seed_file method."""

    @pytest.fixture
    def mock_generator(self, tmp_path):
        """Create a mock SyntheticDataGenerator."""
        data_folder = tmp_path / "data"
        data_folder.mkdir()
        output_folder = tmp_path / "output"
        output_folder.mkdir()

        config_path = data_folder / "config.json"
        config_data = {
            "simulation": {},
            "data": {
                "metapopulation_data_filename": "metapop.csv",
                "mobility_matrix_filename": "mobility.csv",
            },
            "epidemic_params": {},
            "population_params": {"G_labels": ["Y", "M", "O"]},
            "NPI": {},
        }
        config_path.write_text(json.dumps(config_data))

        metapop_df = pd.DataFrame(
            {
                "id": ["001", "002", "003"],
                "total": [1000, 2000, 3000],
                "Y": [300, 600, 900],
                "M": [400, 800, 1200],
                "O": [300, 600, 900],
            }
        )
        metapop_df.to_csv(data_folder / "metapop.csv", index=False)

        pd.DataFrame({"from": [1], "to": [1], "weight": [1.0]}).to_csv(
            data_folder / "mobility.csv", index=False
        )

        rosetta_df = pd.DataFrame(
            {
                "id": ["001", "002", "003"],
                "idx": [1, 2, 3],
            }
        )
        rosetta_df.to_csv(data_folder / "rosetta.csv", index=False)

        return SyntheticDataGenerator(
            str(config_path), str(data_folder), str(output_folder)
        )

    def test_creates_seed_csv(self, mock_generator):
        """Should create seed CSV file."""
        filename, filepath = mock_generator.prepare_seed_file("test_run", seed_size=50)

        assert "seeds_test_run" in filename
        assert os.path.exists(filepath)

    def test_seed_file_has_correct_columns(self, mock_generator):
        """Seed file should have correct columns."""
        filename, filepath = mock_generator.prepare_seed_file("test_run", seed_size=50)

        df = pd.read_csv(filepath)
        expected_cols = ["name", "id", "idx", "Y", "M", "O"]
        assert list(df.columns) == expected_cols

    def test_seed_placed_in_middle_age_group(self, mock_generator):
        """Seed should be placed in middle age group (M)."""
        filename, filepath = mock_generator.prepare_seed_file("test_run", seed_size=50)

        df = pd.read_csv(filepath)
        assert df["M"].iloc[0] == 50
        assert df["Y"].iloc[0] == 0
        assert df["O"].iloc[0] == 0

    def test_seed_size_matches_input(self, mock_generator):
        """Seed size in file should match input."""
        filename, filepath = mock_generator.prepare_seed_file("test_run", seed_size=100)

        df = pd.read_csv(filepath)
        assert df["M"].iloc[0] == 100


class TestSampleInterventionProfiles:
    """Tests for _sample_intervention_profiles method."""

    def test_zero_fraction_returns_empty_set(self):
        """fraction=0.0 should return empty set."""
        gen = SyntheticDataGenerator.__new__(SyntheticDataGenerator)
        result = gen._sample_intervention_profiles(n_profiles=10, fraction=0.0, seed=42)
        assert result == set()

    def test_fraction_one_returns_all_profiles(self):
        """fraction=1.0 should return all profiles."""
        gen = SyntheticDataGenerator.__new__(SyntheticDataGenerator)
        result = gen._sample_intervention_profiles(n_profiles=10, fraction=1.0, seed=42)
        assert result == set(range(10))

    def test_fraction_greater_than_one_returns_all(self):
        """fraction>1.0 should return all profiles."""
        gen = SyntheticDataGenerator.__new__(SyntheticDataGenerator)
        result = gen._sample_intervention_profiles(n_profiles=10, fraction=1.5, seed=42)
        assert result == set(range(10))

    def test_partial_fraction_returns_subset(self):
        """0<fraction<1 should return subset."""
        gen = SyntheticDataGenerator.__new__(SyntheticDataGenerator)
        result = gen._sample_intervention_profiles(n_profiles=20, fraction=0.5, seed=42)

        assert 0 < len(result) < 20
        assert all(0 <= p < 20 for p in result)

    def test_reproducible_with_same_seed(self):
        """Same seed should produce same subset."""
        gen = SyntheticDataGenerator.__new__(SyntheticDataGenerator)
        result1 = gen._sample_intervention_profiles(
            n_profiles=20, fraction=0.5, seed=42
        )
        result2 = gen._sample_intervention_profiles(
            n_profiles=20, fraction=0.5, seed=42
        )

        assert result1 == result2

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different subsets."""
        gen = SyntheticDataGenerator.__new__(SyntheticDataGenerator)
        result1 = gen._sample_intervention_profiles(
            n_profiles=20, fraction=0.5, seed=42
        )
        result2 = gen._sample_intervention_profiles(
            n_profiles=20, fraction=0.5, seed=123
        )

        assert result1 != result2


class TestCheckRunSuccess:
    """Tests for check_run_success method."""

    @pytest.fixture
    def mock_generator(self, tmp_path):
        """Create a mock SyntheticDataGenerator."""
        data_folder = tmp_path / "data"
        data_folder.mkdir()
        output_folder = tmp_path / "output"
        output_folder.mkdir()

        config_path = data_folder / "config.json"
        config_data = {
            "simulation": {},
            "data": {
                "metapopulation_data_filename": "metapop.csv",
                "mobility_matrix_filename": "mobility.csv",
            },
            "epidemic_params": {},
            "population_params": {"G_labels": ["Y", "M", "O"]},
            "NPI": {},
        }
        config_path.write_text(json.dumps(config_data))

        pd.DataFrame({"id": ["1"], "total": [1000]}).to_csv(
            data_folder / "metapop.csv", index=False
        )
        pd.DataFrame({"from": [1], "to": [1], "weight": [1.0]}).to_csv(
            data_folder / "mobility.csv", index=False
        )
        pd.DataFrame({"id": ["1"], "idx": [1]}).to_csv(
            data_folder / "rosetta.csv", index=False
        )

        return SyntheticDataGenerator(
            str(config_path), str(data_folder), str(output_folder)
        )

    def test_returns_false_if_error_json_exists(self, mock_generator):
        """Should return False if ERROR.json exists."""
        run_folder = Path(mock_generator.output_folder) / "run_test"
        run_folder.mkdir()
        (run_folder / "ERROR.json").write_text('{"error": "test"}')

        success, reason = mock_generator.check_run_success("test")
        assert success is False
        assert "ERROR.json" in reason

    def test_returns_false_if_missing_observables(self, mock_generator):
        """Should return False if observables.nc missing."""
        run_folder = Path(mock_generator.output_folder) / "run_test"
        run_folder.mkdir()
        (run_folder / "output").mkdir()
        # No observables.nc

        success, reason = mock_generator.check_run_success("test")
        assert success is False
        assert "Missing observables" in reason

    def test_returns_true_if_successful(self, mock_generator):
        """Should return True if run successful."""
        run_folder = Path(mock_generator.output_folder) / "run_test"
        output_folder = run_folder / "output"
        output_folder.mkdir(parents=True)
        (output_folder / "observables.nc").write_text("")  # Mock file

        success, reason = mock_generator.check_run_success("test")
        assert success is True
        assert reason is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
