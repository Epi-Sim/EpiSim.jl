"""
Tests for Metapopulation class
"""

import pytest
import pandas as pd
import xarray as xr
from pathlib import Path

from episim_python import Metapopulation


class TestMetapopulation:
    """Test cases for Metapopulation class"""

    def test_init_with_metapop_only(self, test_metapopulation_csv):
        """Test initialization with only metapopulation CSV"""
        metapop = Metapopulation(str(test_metapopulation_csv))

        assert metapop._region_ids == ["region_1", "region_2", "region_3"]
        assert metapop._agent_types == ["Y", "M", "O"]
        assert metapop._region_areas == {
            "region_1": 100.0,
            "region_2": 85.0,
            "region_3": 120.0,
        }

    def test_init_with_rosetta(self, test_metapopulation_csv, test_rosetta_csv):
        """Test initialization with both metapopulation and rosetta CSV"""
        metapop = Metapopulation(str(test_metapopulation_csv), str(test_rosetta_csv))

        assert metapop._region_ids == ["region_1", "region_2", "region_3"]
        assert metapop._agent_types == ["Y", "M", "O"]
        assert list(metapop._levels) == ["level_1", "province", "region"]

    def test_as_dataarray(self, test_metapopulation_csv):
        """Test conversion to xarray DataArray"""
        metapop = Metapopulation(str(test_metapopulation_csv))
        da = metapop.as_datarray()

        assert isinstance(da, xr.DataArray)
        assert da.name == "population"
        assert list(da.dims) == ["M", "G"]
        assert list(da.coords["M"]) == ["region_1", "region_2", "region_3"]
        assert list(da.coords["G"]) == ["Y", "M", "O"]
        assert da.shape == (3, 3)  # 3 regions, 3 age groups

        # Check specific values
        assert da.sel(M="region_1", G="Y").values == 5000
        assert da.sel(M="region_2", G="M").values == 7500
        assert da.sel(M="region_3", G="O").values == 3500

    def test_aggregate_to_level_1(self, test_metapopulation_csv, test_rosetta_csv):
        """Test aggregation to level_1 (no aggregation)"""
        metapop = Metapopulation(str(test_metapopulation_csv), str(test_rosetta_csv))

        # Test as array
        da = metapop.aggregate_to_level("level_1", as_array=True)
        assert isinstance(da, xr.DataArray)
        assert da.shape == (3, 3)

        # Test as DataFrame
        df = metapop.aggregate_to_level("level_1", as_array=False)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)

    def test_aggregate_to_province(self, test_metapopulation_csv, test_rosetta_csv):
        """Test aggregation to province level"""
        metapop = Metapopulation(str(test_metapopulation_csv), str(test_rosetta_csv))
        da = metapop.aggregate_to_level("province", as_array=True)

        assert isinstance(da, xr.DataArray)
        assert da.shape == (2, 3)  # 2 provinces, 3 age groups
        assert set(da.coords["M"].values) == {"prov_A", "prov_B"}

        # Check aggregated values
        # prov_A should have region_1 + region_3
        prov_a_y = da.sel(M="prov_A", G="Y").values
        expected_y = 5000 + 6000  # region_1 + region_3
        assert prov_a_y == expected_y

    def test_aggregate_to_region(self, test_metapopulation_csv, test_rosetta_csv):
        """Test aggregation to region level"""
        metapop = Metapopulation(str(test_metapopulation_csv), str(test_rosetta_csv))
        da = metapop.aggregate_to_level("region", as_array=True)

        assert isinstance(da, xr.DataArray)
        assert da.shape == (2, 3)  # 2 regions, 3 age groups
        assert set(da.coords["M"].values) == {"reg_X", "reg_Y"}

        # Check aggregated values
        # reg_X should have region_1 + region_3
        reg_x_m = da.sel(M="reg_X", G="M").values
        expected_m = 8000 + 9000  # region_1 + region_3
        assert reg_x_m == expected_m

    def test_rosetta_index_mismatch(self, test_metapopulation_csv, temp_dir):
        """Test error when rosetta indices don't match metapopulation IDs"""
        # Create mismatched rosetta file
        bad_rosetta = Path(temp_dir) / "bad_rosetta.csv"
        bad_rosetta.write_text("level_1,province,region\nwrong_id,prov_A,reg_X\n")

        with pytest.raises(AssertionError):
            Metapopulation(str(test_metapopulation_csv), str(bad_rosetta))

    def test_missing_files(self):
        """Test error handling for missing files"""
        with pytest.raises(FileNotFoundError):
            Metapopulation("nonexistent.csv")

    def test_population_data_structure(self, test_metapopulation_csv):
        """Test internal population data structure"""
        metapop = Metapopulation(str(test_metapopulation_csv))

        # Check that _populations DataFrame has correct structure
        assert isinstance(metapop._populations, pd.DataFrame)
        assert metapop._populations.shape == (3, 3)  # 3 regions, 3 age groups
        assert list(metapop._populations.columns) == ["Y", "M", "O"]
        assert list(metapop._populations.index) == ["region_1", "region_2", "region_3"]

        # Check specific values
        assert metapop._populations.loc["region_1", "Y"] == 5000
        assert metapop._populations.loc["region_2", "M"] == 7500
        assert metapop._populations.loc["region_3", "O"] == 3500

    def test_total_column_handling(self, temp_dir):
        """Test that total column is properly dropped"""
        # Create CSV with total column
        csv_with_total = Path(temp_dir) / "with_total.csv"
        csv_with_total.write_text(
            "id,area,Y,M,O,total\n"
            "region_1,100.0,5000,8000,3000,16000\n"
            "region_2,85.0,4500,7500,2800,14800\n"
        )

        metapop = Metapopulation(str(csv_with_total))

        # Total column should be dropped
        assert "total" not in metapop._populations.columns
        assert list(metapop._populations.columns) == ["Y", "M", "O"]

    def test_single_age_group_handling(self, temp_dir):
        """Test handling of data with single age group"""
        # Create CSV with single age group
        single_age_csv = Path(temp_dir) / "single_age.csv"
        single_age_csv.write_text(
            "id,area,Population\nregion_1,100.0,16000\nregion_2,85.0,14800\n"
        )

        metapop = Metapopulation(str(single_age_csv))

        assert metapop._agent_types == ["Population"]
        assert metapop._populations.shape == (2, 1)

        da = metapop.as_datarray()
        assert da.shape == (2, 1)
        assert da.sel(M="region_1", G="Population").values == 16000
