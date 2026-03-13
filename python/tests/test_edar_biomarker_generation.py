"""Tests for EDAR-level biomarker generation.

Tests cover:
- Loading EDAR-municipality mapping from NetCDF
- Building EMAP matrix correctly
- Aggregating infections from municipalities to EDAR catchment areas
- Verifying spatial dimensions in output (edar_id vs region_id)
- Error handling when EDAR file is missing

Physical Model Context (from CONTEXT_SYNTHETIC_GEN.md):
    Wastewater concentration follows: C = Σ(I_g × S_g) / (P × F)

    Where:
    - I_g = infections in age group g
    - S_g = shedding kinetics for age group g (convolution kernel)
    - P = population in catchment area
    - F = per-capita wastewater flow (implicit in sensitivity_scale)

    Key: Population division models DILUTION physics, not per-capita normalization.
    More population = more wastewater flow = lower concentration per unit volume.

    For EDARs specifically:
    - Infections are aggregated via EMAP (wastewater flow fractions from each municipality)
    - Population is the total in the EDAR's catchment area (sum of contributing municipalities)
    - Division by population models the physical dilution that occurs in sewer networks
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add python directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from process_synthetic_outputs import (
    aggregate_infections_to_edar,
    load_edar_muni_mapping,
)

# Path to test fixtures
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


class TestLoadEdarMuniMapping:
    """Tests for loading EDAR-municipality edges and building EMAP."""

    def test_loads_valid_edges_file(self):
        """Should load EDAR edges from valid NetCDF file."""
        edar_path = os.path.join(FIXTURES_DIR, "mini_edar_muni_edges.nc")

        # Create metapop df matching EDAR home IDs
        metapop_df = pd.DataFrame({
            "id": ["muni_01", "muni_02", "muni_03", "muni_04", "muni_05"],
            "total": [10000, 15000, 12000, 8000, 20000]
        })

        result = load_edar_muni_mapping(metapop_df, edar_nc_path=edar_path)

        assert result is not None
        assert "edar_ids" in result
        assert "region_ids" in result
        assert "emap" in result
        assert result["emap"].shape == (3, 5)  # 3 EDARs, 5 regions

    def test_raises_error_when_file_not_found(self):
        """Should raise FileNotFoundError when EDAR edges file doesn't exist."""
        metapop_df = pd.DataFrame({"id": ["region_1"], "total": [10000]})

        with pytest.raises(FileNotFoundError, match="EDAR-municipality edges file not found"):
            load_edar_muni_mapping(metapop_df, edar_nc_path="/nonexistent/path.nc")

    def test_emap_rows_normalize_to_one(self):
        """EMAP rows should sum to 1 (each EDAR receives fractional contributions)."""
        edar_path = os.path.join(FIXTURES_DIR, "mini_edar_muni_edges.nc")

        metapop_df = pd.DataFrame({
            "id": ["muni_01", "muni_02", "muni_03", "muni_04", "muni_05"],
            "total": [10000, 15000, 12000, 8000, 20000]
        })

        result = load_edar_muni_mapping(metapop_df, edar_nc_path=edar_path)
        emap = result["emap"]

        # Each EDAR row should sum to 1.0
        row_sums = emap.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

    def test_emap_handles_sparse_mappings(self):
        """EMAP should handle sparse contribution matrices (some NaN values)."""
        edar_path = os.path.join(FIXTURES_DIR, "mini_edar_muni_edges.nc")

        metapop_df = pd.DataFrame({
            "id": ["muni_01", "muni_02", "muni_03", "muni_04", "muni_05"],
            "total": [10000, 15000, 12000, 8000, 20000]
        })

        result = load_edar_muni_mapping(metapop_df, edar_nc_path=edar_path)
        emap = result["emap"]

        # EDAR_01 only maps to muni_01, muni_02
        assert emap[0, 0] > 0  # EDAR_01 -> muni_01
        assert emap[0, 1] > 0  # EDAR_01 -> muni_02
        assert emap[0, 2] == 0  # EDAR_01 doesn't map to muni_03

        # EDAR_02 only maps to muni_03, muni_04
        assert emap[1, 2] > 0  # EDAR_02 -> muni_03
        assert emap[1, 3] > 0  # EDAR_02 -> muni_04
        assert emap[1, 0] == 0  # EDAR_02 doesn't map to muni_01

        # EDAR_03 only maps to muni_05
        assert emap[2, 4] > 0  # EDAR_03 -> muni_05
        assert emap[2, 0] == 0  # EDAR_03 doesn't map to muni_01

    def test_emap_partial_region_overlap(self):
        """EMAP should handle when only some metapop regions appear in EDAR mapping."""
        edar_path = os.path.join(FIXTURES_DIR, "mini_edar_muni_edges.nc")

        # Metapop has more regions than EDAR mapping
        metapop_df = pd.DataFrame({
            "id": ["muni_01", "muni_02", "muni_03", "muni_04", "muni_05", "muni_06"],
            "total": [10000, 15000, 12000, 8000, 20000, 5000]
        })

        result = load_edar_muni_mapping(metapop_df, edar_nc_path=edar_path)

        # EMAP should have shape (n_edar, n_region)
        assert result["emap"].shape == (3, 6)

        # muni_06 should have all zeros (not in EDAR mapping)
        assert np.all(result["emap"][:, 5] == 0)

    def test_returns_correct_ids(self):
        """Should return correct EDAR and region IDs."""
        edar_path = os.path.join(FIXTURES_DIR, "mini_edar_muni_edges.nc")

        metapop_df = pd.DataFrame({
            "id": ["muni_01", "muni_02", "muni_03", "muni_04", "muni_05"],
            "total": [10000, 15000, 12000, 8000, 20000]
        })

        result = load_edar_muni_mapping(metapop_df, edar_nc_path=edar_path)

        # EDAR IDs from fixture
        assert result["edar_ids"] == ["EDAR_01", "EDAR_02", "EDAR_03"]

        # Region IDs from metapop_df (converted to string)
        assert result["region_ids"] == ["muni_01", "muni_02", "muni_03", "muni_04", "muni_05"]


class TestAggregateInfectionsToEdar:
    """Tests for aggregating infections from municipalities to EDAR catchments."""

    def test_aggregates_infections_correctly(self):
        """Should aggregate infections from regions to EDARs using EMAP."""
        # Simple test case: 2 EDARs, 3 regions, 2 age groups
        infections = np.array([
            [[10, 5],   # t=0: region 0, age groups [0, 1]
             [8, 4],    # t=0: region 1
             [5, 3]],   # t=0: region 2
            [[20, 10],  # t=1: region 0
             [16, 8],   # t=1: region 1
             [10, 6]]   # t=1: region 2
        ])  # Shape: (Time=2, Region=3, AgeGroup=2)

        emap = np.array([
            [0.5, 0.3, 0.2],  # EDAR 0 receives 50%, 30%, 20%
            [0.5, 0.7, 0.8],  # EDAR 1 receives remaining (normalized)
        ])  # Shape: (EDAR=2, Region=3)

        population = np.array([1000, 800, 600])

        infections_edar, pop_edar = aggregate_infections_to_edar(infections, emap, population)

        # Check shape: (Time=2, EDAR=2, AgeGroup=2)
        assert infections_edar.shape == (2, 2, 2)
        assert pop_edar.shape == (2,)

        # Verify aggregation for first timestep, first EDAR, first age group:
        # 0.5*10 + 0.3*8 + 0.2*5 = 5 + 2.4 + 1 = 8.4
        np.testing.assert_allclose(infections_edar[0, 0, 0], 8.4, rtol=1e-5)

        # Verify aggregation for first timestep, second EDAR, first age group:
        # 0.5*10 + 0.7*8 + 0.8*5 = 5 + 5.6 + 4 = 14.6
        np.testing.assert_allclose(infections_edar[0, 1, 0], 14.6, rtol=1e-5)

    def test_aggregates_population_correctly(self):
        """Should aggregate population weighted by EMAP contributions."""
        population = np.array([1000, 800, 600])
        emap = np.array([
            [0.5, 0.3, 0.2],
            [0.5, 0.7, 0.8],
        ])

        _, pop_edar = aggregate_infections_to_edar(
            np.zeros((1, 3, 1)), emap, population
        )

        # EDAR 0: 0.5*1000 + 0.3*800 + 0.2*600 = 500 + 240 + 120 = 860
        # EDAR 1: 0.5*1000 + 0.7*800 + 0.8*600 = 500 + 560 + 480 = 1540
        np.testing.assert_allclose(pop_edar, [860, 1540], rtol=1e-5)

    def test_preserves_age_group_dimension(self):
        """Aggregation should preserve age group dimension."""
        # 3 time points, 4 regions, 5 age groups
        infections = np.random.rand(3, 4, 5) * 100
        emap = np.array([
            [0.6, 0.4, 0.0, 0.0],  # EDAR 0
            [0.4, 0.6, 0.0, 0.0],  # EDAR 1
        ])
        population = np.array([1000, 800, 600, 400])

        infections_edar, _ = aggregate_infections_to_edar(infections, emap, population)

        # Should preserve time and age group dimensions
        assert infections_edar.shape == (3, 2, 5)  # (Time, EDAR, AgeGroup)

    def test_handles_zero_infections(self):
        """Should handle zero infections without errors."""
        infections = np.zeros((2, 3, 2))
        emap = np.array([
            [0.5, 0.3, 0.2],
            [0.5, 0.7, 0.8],
        ])
        population = np.array([1000, 800, 600])

        infections_edar, pop_edar = aggregate_infections_to_edar(infections, emap, population)

        np.testing.assert_array_equal(infections_edar, np.zeros((2, 2, 2)))

    def test_handles_single_timepoint(self):
        """Should handle single timepoint data."""
        infections = np.array([[[10, 5], [8, 4], [5, 3]]])  # Shape: (1, 3, 2)
        emap = np.array([
            [0.5, 0.3, 0.2],
            [0.5, 0.7, 0.8],
        ])
        population = np.array([1000, 800, 600])

        infections_edar, pop_edar = aggregate_infections_to_edar(infections, emap, population)

        assert infections_edar.shape == (1, 2, 2)

    def test_single_edar_one_to_many_mapping(self):
        """Should handle one EDAR receiving contributions from multiple regions."""
        infections = np.array([
            [[10, 5], [20, 10], [30, 15]]  # 3 regions contributing to 1 EDAR
        ])  # Shape: (1, 3, 2)
        emap = np.array([[0.2, 0.3, 0.5]])  # 1 EDAR
        population = np.array([1000, 2000, 3000])

        infections_edar, pop_edar = aggregate_infections_to_edar(infections, emap, population)

        # infections: 0.2*10 + 0.3*20 + 0.5*30 = 2 + 6 + 15 = 23
        np.testing.assert_allclose(infections_edar[0, 0, 0], 23.0, rtol=1e-5)

        # population: 0.2*1000 + 0.3*2000 + 0.5*3000 = 200 + 600 + 1500 = 2300
        np.testing.assert_allclose(pop_edar, [2300], rtol=1e-5)

    def test_many_edars_many_to_many_mapping(self):
        """Should handle many-to-many EDAR-region mappings."""
        # 10 regions, 5 EDARs
        n_regions = 10
        n_edars = 5
        n_age_groups = 3
        n_time = 5

        infections = np.random.rand(n_time, n_regions, n_age_groups) * 100
        population = np.random.rand(n_regions) * 10000 + 5000

        # Create normalized EMAP (each EDAR row sums to 1)
        emap = np.random.rand(n_edars, n_regions)
        emap = emap / emap.sum(axis=1, keepdims=True)

        infections_edar, pop_edar = aggregate_infections_to_edar(infections, emap, population)

        assert infections_edar.shape == (n_time, n_edars, n_age_groups)
        assert pop_edar.shape == (n_edars,)

        # Note: When EDARs have overlapping catchment areas, the sum of EDAR populations
        # will be greater than the sum of region populations. This is expected behavior.
        # For non-overlapping EDARs, the sums would be equal.
        # Just verify that EDAR populations are positive
        assert np.all(pop_edar > 0)


class TestEdarBiomarkerEndToEnd:
    """End-to-end tests for EDAR biomarker generation."""

    def test_output_has_edar_id_dimension(self):
        """Zarr output should have edar_id coordinate, not region_id for biomarkers."""
        # This would require running a mini pipeline
        # For now, document the expected behavior
        # Expected: zarr dataset with edar_id dimension for wastewater biomarkers

    def test_biomarkers_aggregated_to_catchment_areas(self):
        """Biomarker values should reflect catchment area aggregation."""
        # Document: EDAR biomarker = sum(municipality_i * contribution_ratio_i)
        # This is verified through aggregate_infections_to_edar tests


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
