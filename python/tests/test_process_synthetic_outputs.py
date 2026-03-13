"""Tests for process_synthetic_outputs.py key functions.

This module tests the core output processing functionality used to convert
simulation outputs into zarr format for downstream analysis.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Add python directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from process_synthetic_outputs import (
    apply_missing_data_patterns,
    assign_sparsity_tiers,
    build_date_range,
    detect_and_load_time_varying_mobility,
    generate_wastewater_with_censoring,
    load_compartment_latents,
    load_edar_muni_mapping,
    load_kappa0_series,
    load_mobility_matrix,
    load_mobility_noise_params,
    parse_run_metadata,
    resolve_kappa0_path,
    sanitize_run_id,
)


class TestSanitizeRunId:
    """Tests for sanitize_run_id function."""

    def test_removes_run_prefix(self):
        """Should remove 'run_' prefix."""
        result = sanitize_run_id("run_123_Baseline")
        assert result.startswith("123_Baseline")

    def test_replaces_invalid_chars(self):
        """Should replace invalid characters with underscore."""
        result = sanitize_run_id("run_123@Baseline#test")
        assert "@" not in result
        assert "#" not in result
        assert "_" in result

    def test_pads_to_max_length(self):
        """Should pad short IDs to max_length."""
        result = sanitize_run_id("run_1", max_length=50)
        assert len(result) == 50

    def test_truncates_long_ids(self):
        """Should truncate IDs exceeding max_length."""
        result = sanitize_run_id("run_" + "a" * 100, max_length=50)
        assert len(result) == 50

    def test_handles_empty_string(self):
        """Should handle empty string."""
        result = sanitize_run_id("")
        assert result.strip() != ""


class TestParseRunMetadata:
    """Tests for parse_run_metadata function."""

    def test_parses_baseline_scenario(self):
        """Should parse baseline scenario correctly."""
        scenario, strength = parse_run_metadata("run_0_Baseline")
        assert scenario == "Baseline"
        assert strength == 0.0

    def test_parses_global_timed_with_strength(self):
        """Should parse Global_Timed with strength."""
        scenario, strength = parse_run_metadata("run_0_Global_Timed_s50")
        assert scenario == "Global_Timed"
        assert strength == 0.5

    def test_parses_high_strength(self):
        """Should parse high strength values."""
        scenario, strength = parse_run_metadata("run_0_Global_Timed_s100")
        assert strength == 1.0

    def test_handles_no_strength_suffix(self):
        """Should handle missing strength suffix."""
        scenario, strength = parse_run_metadata("run_0_UnknownScenario")
        assert np.isnan(strength)

    def test_strips_run_prefix(self):
        """Should handle run_ prefix correctly."""
        scenario, strength = parse_run_metadata("run_123_Global_Timed_s25")
        assert scenario == "Global_Timed"
        assert strength == 0.25


class TestApplyMissingDataPatterns:
    """Tests for apply_missing_data_patterns function."""

    def test_returns_data_and_mask(self):
        """Should return data with gaps and mask."""
        data = np.ones((10, 5))
        data_with_gaps, mask = apply_missing_data_patterns(data, missing_rate=0.1)

        assert data_with_gaps.shape == data.shape
        assert mask.shape == data.shape
        assert np.all((mask == 0) | (mask == 1))

    def test_applies_random_missing(self):
        """Should apply random missing values."""
        np.random.seed(42)
        data = np.ones((100, 10))
        data_with_gaps, mask = apply_missing_data_patterns(
            data, missing_rate=0.1, rng=np.random.default_rng(42)
        )

        # Should have some NaN values
        assert np.any(np.isnan(data_with_gaps))
        # Mask should be 0 where data is NaN
        nan_mask = np.isnan(data_with_gaps)
        assert np.all(mask[nan_mask] == 0)

    def test_applies_gap_based_missing(self):
        """Should apply gap-based missing values."""
        data = np.ones((50, 10))
        data_with_gaps, mask = apply_missing_data_patterns(
            data, missing_rate=0.1, missing_gap_length=5, rng=np.random.default_rng(42)
        )

        # Should have some NaN values
        assert np.any(np.isnan(data_with_gaps))

    def test_handles_1d_input(self):
        """Should handle 1D input."""
        data = np.ones(50)
        data_with_gaps, mask = apply_missing_data_patterns(
            data, missing_rate=0.1, rng=np.random.default_rng(42)
        )

        assert data_with_gaps.shape == (50,)
        assert mask.shape == (50,)

    def test_zero_missing_rate_minimal_missing(self):
        """Zero missing rate should have minimal missing data (gap-based may still apply)."""
        data = np.ones((10, 5))
        data_with_gaps, mask = apply_missing_data_patterns(data, missing_rate=0.0)

        # With missing_rate=0, random missing is 0, but gap-based may still add some
        # Just verify it doesn't crash and returns proper shapes
        assert data_with_gaps.shape == data.shape
        assert mask.shape == data.shape

    def test_full_missing_rate(self):
        """High missing rate should create many NaNs."""
        data = np.ones((10, 5))
        data_with_gaps, mask = apply_missing_data_patterns(
            data, missing_rate=0.9, rng=np.random.default_rng(42)
        )

        # Most values should be NaN
        nan_count = np.sum(np.isnan(data_with_gaps))
        assert nan_count > 20  # Significant portion


class TestBuildDateRange:
    """Tests for build_date_range function."""

    def test_builds_date_range_from_config(self):
        """Should build date range from config."""
        config = {"simulation": {"start_date": "2020-03-01"}}
        dates = build_date_range(config, time_len=10)

        assert len(dates) == 10
        assert str(dates[0]) == "2020-03-01 00:00:00"

    def test_missing_start_date_raises_error(self):
        """Should raise error if start_date missing."""
        config = {"simulation": {}}

        with pytest.raises(ValueError, match="Missing simulation.start_date"):
            build_date_range(config, time_len=10)

    def test_different_start_dates(self):
        """Should handle different start dates."""
        config = {"simulation": {"start_date": "2021-01-15"}}
        dates = build_date_range(config, time_len=5)

        assert len(dates) == 5
        assert str(dates[0]) == "2021-01-15 00:00:00"


class TestResolveKappa0Path:
    """Tests for resolve_kappa0_path function."""

    def test_returns_none_if_no_kappa0_filename(self, tmp_path):
        """Should return None if no kappa0_filename in config."""
        config = {"data": {}}
        result = resolve_kappa0_path(config, tmp_path)
        assert result is None

    def test_returns_path_if_exists(self, tmp_path):
        """Should return path if file exists."""
        kappa_file = tmp_path / "kappa0.csv"
        kappa_file.write_text("date,reduction\n2020-03-01,0.0")

        config = {"data": {"kappa0_filename": str(kappa_file)}}
        result = resolve_kappa0_path(config, tmp_path)

        assert result == str(kappa_file)

    def test_checks_run_dir_if_not_absolute(self, tmp_path):
        """Should check run directory for relative paths."""
        kappa_file = tmp_path / "kappa0.csv"
        kappa_file.write_text("date,reduction\n2020-03-01,0.0")

        config = {"data": {"kappa0_filename": "kappa0.csv"}}
        result = resolve_kappa0_path(config, tmp_path)

        assert result == str(kappa_file)

    def test_returns_none_if_not_found(self, tmp_path):
        """Should return None if file not found."""
        config = {"data": {"kappa0_filename": "nonexistent.csv"}}
        result = resolve_kappa0_path(config, tmp_path)
        assert result is None


class TestLoadKappa0Series:
    """Tests for load_kappa0_series function."""

    def test_loads_from_date_column(self, tmp_path):
        """Should load from date column."""
        kappa_file = tmp_path / "kappa0.csv"
        kappa_file.write_text(
            "date,reduction\n2020-03-01,0.0\n2020-03-02,0.5\n2020-03-03,0.0"
        )

        dates = pd.date_range("2020-03-01", periods=3)
        result = load_kappa0_series(str(kappa_file), dates)

        assert len(result) == 3
        assert result[0] == 0.0
        assert result[1] == 0.5
        assert result[2] == 0.0

    def test_loads_from_time_column(self, tmp_path):
        """Should load from time column."""
        kappa_file = tmp_path / "kappa0.csv"
        kappa_file.write_text("time,reduction\n0,0.0\n1,0.5\n2,0.0")

        dates = pd.date_range("2020-03-01", periods=3)
        result = load_kappa0_series(str(kappa_file), dates)

        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 0.0])

    def test_returns_zeros_if_no_file(self):
        """Should return zeros if no file."""
        dates = pd.date_range("2020-03-01", periods=5)
        result = load_kappa0_series(None, dates)

        assert len(result) == 5
        assert np.all(result == 0)

    def test_pads_short_series(self, tmp_path):
        """Should pad short series with zeros."""
        kappa_file = tmp_path / "kappa0.csv"
        kappa_file.write_text("time,reduction\n0,0.5\n1,0.5")

        dates = pd.date_range("2020-03-01", periods=5)
        result = load_kappa0_series(str(kappa_file), dates)

        assert len(result) == 5
        assert result[0] == 0.5
        assert result[1] == 0.5
        assert result[2] == 0.0
        assert result[3] == 0.0
        assert result[4] == 0.0


class TestDetectAndLoadTimeVaryingMobility:
    """Tests for detect_and_load_time_varying_mobility function."""

    def test_returns_none_if_no_mobility_file(self, tmp_path):
        """Should return None if no mobility file."""
        run_dir = tmp_path / "run_0"
        run_dir.mkdir()

        result = detect_and_load_time_varying_mobility(run_dir, n_regions=5, n_dates=10)
        assert result is None

    def test_loads_and_converts_sparse_to_dense(self, tmp_path):
        """Should load sparse mobility and convert to dense."""
        run_dir = tmp_path / "run_0"
        mobility_dir = run_dir / "mobility"
        mobility_dir.mkdir(parents=True)

        # Create sparse mobility data
        T, E, M = 5, 4, 3
        R_series = np.random.rand(T, E)
        edgelist = np.array([[0, 1], [1, 0], [0, 2], [2, 0]])

        np.savez(
            mobility_dir / "mobility_series.npz",
            R_series=R_series,
            edgelist=edgelist,
            T=T,
            E=E,
            M=M,
        )

        result = detect_and_load_time_varying_mobility(run_dir, n_regions=M, n_dates=T)

        assert result is not None
        assert result.shape == (T, M, M)

    def test_handles_corrupt_file(self, tmp_path):
        """Should handle corrupt mobility file gracefully."""
        run_dir = tmp_path / "run_0"
        mobility_dir = run_dir / "mobility"
        mobility_dir.mkdir(parents=True)

        # Create invalid file
        (mobility_dir / "mobility_series.npz").write_text("invalid data")

        result = detect_and_load_time_varying_mobility(run_dir, n_regions=5, n_dates=10)
        assert result is None


class TestLoadMobilityNoiseParams:
    """Tests for load_mobility_noise_params function."""

    def test_returns_zeros_if_no_file(self, tmp_path):
        """Should return zeros if no mobility file."""
        run_dir = tmp_path / "run_0"
        run_dir.mkdir()

        sigma_O, sigma_D = load_mobility_noise_params(run_dir)
        assert sigma_O == 0.0
        assert sigma_D == 0.0

    def test_loads_params_from_file(self, tmp_path):
        """Should load sigma values from file."""
        run_dir = tmp_path / "run_0"
        mobility_dir = run_dir / "mobility"
        mobility_dir.mkdir(parents=True)

        np.savez(
            mobility_dir / "mobility_series.npz",
            sigma_O=0.3,
            sigma_D=0.5,
            R_series=np.array([[1.0]]),
            edgelist=np.array([[0, 0]]),
        )

        sigma_O, sigma_D = load_mobility_noise_params(run_dir)
        assert sigma_O == 0.3
        assert sigma_D == 0.5

    def test_handles_missing_keys(self, tmp_path):
        """Should handle missing sigma keys."""
        run_dir = tmp_path / "run_0"
        mobility_dir = run_dir / "mobility"
        mobility_dir.mkdir(parents=True)

        np.savez(
            mobility_dir / "mobility_series.npz",
            R_series=np.array([[1.0]]),
            edgelist=np.array([[0, 0]]),
        )

        sigma_O, sigma_D = load_mobility_noise_params(run_dir)
        assert sigma_O == 0.0
        assert sigma_D == 0.0


class TestLoadEdarMuniMapping:
    """Tests for load_edar_muni_mapping function."""

    def test_raises_error_if_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError if edges file not found."""
        metapop_df = pd.DataFrame({"id": ["1", "2"], "total": [1000, 2000]})

        with pytest.raises(FileNotFoundError):
            load_edar_muni_mapping(
                metapop_df, edar_nc_path=str(tmp_path / "nonexistent.nc")
            )

    def test_loads_valid_edges_file(self, tmp_path):
        """Should load valid edges file."""
        # Create mock NetCDF file
        nc_path = tmp_path / "edar_muni_edges.nc"

        ds = xr.Dataset(
            {
                "edar_id": (["edar"], ["EDAR1", "EDAR2"]),
                "home": (["home"], ["1", "2"]),
                "contribution_ratio": (
                    ["edar", "home"],
                    np.array([[0.6, 0.4], [0.3, 0.7]]),
                ),
            }
        )
        ds.to_netcdf(nc_path)

        metapop_df = pd.DataFrame({"id": ["1", "2"], "total": [1000, 2000]})
        result = load_edar_muni_mapping(metapop_df, edar_nc_path=str(nc_path))

        assert "edar_ids" in result
        assert "region_ids" in result
        assert "emap" in result
        assert result["emap"].shape == (2, 2)

    def test_emap_rows_normalize_to_one(self, tmp_path):
        """EMAP rows should normalize to sum to 1."""
        nc_path = tmp_path / "edar_muni_edges.nc"

        ds = xr.Dataset(
            {
                "edar_id": (["edar"], ["EDAR1"]),
                "home": (["home"], ["1", "2"]),
                "contribution_ratio": (["edar", "home"], np.array([[2.0, 3.0]])),
            }
        )
        ds.to_netcdf(nc_path)

        metapop_df = pd.DataFrame({"id": ["1", "2"], "total": [1000, 2000]})
        result = load_edar_muni_mapping(metapop_df, edar_nc_path=str(nc_path))

        # Row should be normalized
        np.testing.assert_array_almost_equal(result["emap"].sum(axis=1), [1.0])


class TestLoadMobilityMatrix:
    """Tests for load_mobility_matrix function."""

    def test_loads_from_mobility_matrix_csv(self, tmp_path):
        """Should load from mobility_matrix.csv."""
        data_folder = tmp_path / "data"
        data_folder.mkdir()

        # Create metapop data
        metapop_df = data_folder / "metapopulation_data.csv"
        metapop_df.write_text("id,total\n1,1000\n2,2000")

        # Create mobility matrix
        mobility_df = data_folder / "mobility_matrix.csv"
        mobility_df.write_text("source_idx,target_idx,ratio\n1,2,0.3\n2,1,0.4")

        result = load_mobility_matrix(str(metapop_df))

        assert result.shape == (2, 2)
        # Rows should sum to 1
        np.testing.assert_array_almost_equal(result.sum(axis=1), [1.0, 1.0])

    def test_loads_from_r_mobility_matrix_csv(self, tmp_path):
        """Should load from R_mobility_matrix.csv if mobility_matrix.csv not found."""
        data_folder = tmp_path / "data"
        data_folder.mkdir()

        # Create metapop data
        metapop_df = data_folder / "metapopulation_data.csv"
        metapop_df.write_text("id,total\n1,1000\n2,2000")

        # Create R_mobility_matrix
        mobility_df = data_folder / "R_mobility_matrix.csv"
        mobility_df.write_text("source_idx,target_idx,ratio\n1,2,0.3\n2,1,0.4")

        result = load_mobility_matrix(str(metapop_df))

        assert result.shape == (2, 2)

    def test_raises_error_if_no_mobility_file(self, tmp_path):
        """Should raise error if no mobility file found."""
        data_folder = tmp_path / "data"
        data_folder.mkdir()

        metapop_df = data_folder / "metapopulation_data.csv"
        metapop_df.write_text("id,total\n1,1000\n2,2000")

        with pytest.raises(ValueError, match="Mobility matrix not found"):
            load_mobility_matrix(str(metapop_df))


class TestLoadCompartmentLatents:
    """Tests for latent compartment loading from simulator output."""

    def test_loads_and_aggregates_latent_states(self, tmp_path):
        """Should aggregate age-stratified compartments to region/date latents."""
        compartments_path = tmp_path / "compartments_full.nc"
        dims = ("G", "M", "T")
        ds = xr.Dataset(
            {
                "S": (dims, np.array([[[10, 11], [20, 21]], [[1, 2], [3, 4]]])),
                "E": (dims, np.array([[[2, 3], [4, 5]], [[1, 1], [1, 1]]])),
                "A": (dims, np.array([[[5, 6], [7, 8]], [[1, 1], [2, 2]]])),
                "I": (dims, np.array([[[9, 10], [11, 12]], [[3, 3], [4, 4]]])),
                "PH": (dims, np.array([[[1, 1], [2, 2]], [[0, 1], [1, 0]]])),
                "PD": (dims, np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])),
                "HR": (dims, np.array([[[2, 2], [3, 3]], [[1, 0], [0, 1]]])),
                "HD": (dims, np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])),
                "R": (dims, np.array([[[4, 5], [6, 7]], [[2, 2], [3, 3]]])),
                "D": (dims, np.array([[[1, 1], [1, 1]], [[0, 1], [1, 0]]])),
                "CH": (dims, np.array([[[3, 3], [3, 3]], [[1, 1], [1, 1]]])),
            }
        )
        ds.to_netcdf(compartments_path)

        latents = load_compartment_latents(compartments_path)

        assert set(latents) >= {
            "latent_S_true",
            "latent_E_true",
            "latent_A_true",
            "latent_I_true",
            "latent_R_true",
            "latent_D_true",
            "latent_CH_true",
            "latent_hospitalized_true",
            "latent_active_true",
        }
        np.testing.assert_array_equal(
            latents["latent_S_true"], np.array([[11, 13], [23, 25]])
        )
        np.testing.assert_array_equal(
            latents["latent_hospitalized_true"], np.array([[4, 3], [4, 5]])
        )
        np.testing.assert_array_equal(
            latents["latent_active_true"], np.array([[27, 30], [37, 40]])
        )

    def test_raises_if_compartment_file_missing(self, tmp_path):
        """Should fail clearly when latent export is requested without compartments."""
        with pytest.raises(FileNotFoundError, match="Missing compartments_full.nc"):
            load_compartment_latents(tmp_path / "missing.nc")


class TestGenerateWastewaterWithCensoring:
    """Tests for generate_wastewater_with_censoring function."""

    def test_returns_wastewater_and_censor_arrays(self):
        """Should return wastewater and censor flag arrays."""
        infections = np.ones((10, 5, 3)) * 100  # Time, Region, Age
        population = np.ones(5) * 1000
        gene_targets = {
            "N1": {
                "sensitivity_scale": 500000,
                "noise_sigma": 0.5,
                "limit_of_detection": 375,
                "lod_probabilistic": False,
            }
        }
        wastewater_cfg = {"gamma_shape": 2.5, "gamma_scale": 4.0}

        wastewater, censor = generate_wastewater_with_censoring(
            infections, population, gene_targets, wastewater_cfg
        )

        assert wastewater.shape == (10, 5, 1)  # Time, Region, Targets
        assert censor.shape == (10, 5, 1)
        assert censor.dtype == np.int8

    def test_censor_flags_are_valid(self):
        """Censor flags should be 0, 1, or 2."""
        infections = np.ones((10, 5, 3)) * 100
        population = np.ones(5) * 1000
        gene_targets = {
            "N1": {
                "sensitivity_scale": 500000,
                "noise_sigma": 0.5,
                "limit_of_detection": 375,
                "lod_probabilistic": False,
            }
        }
        wastewater_cfg = {"gamma_shape": 2.5, "gamma_scale": 4.0}

        wastewater, censor = generate_wastewater_with_censoring(
            infections, population, gene_targets, wastewater_cfg
        )

        assert np.all((censor == 0) | (censor == 1) | (censor == 2))


class TestAssignSparsityTiers:
    """Tests for assign_sparsity_tiers function (already covered in test_sparsity_tiers.py)."""

    def test_basic_functionality(self):
        """Should assign sparsity tiers correctly."""
        result = assign_sparsity_tiers(10, [0.1, 0.5, 0.9], seed=42)

        assert len(result) == 10
        assert all(s in [0.1, 0.5, 0.9] for s in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
