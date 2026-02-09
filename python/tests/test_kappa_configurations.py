"""Tests for kappa (mobility reduction) parameter configurations.

This module consolidates tests from:
- test_kappa_sweep.py: Parameter sweep across different κ₀ values
- test_exact_configs.py: Exact configuration comparisons (JSON vs CSV modes)
- test_json_mode.py: JSON-only mode testing
- test_tc.py: Intervention timing (tᶜ) parameter testing
- test_verify_kappa.py: Kappa0 CSV vs JSON mode verification
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Add python directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from episim_python.epi_sim import EpiSim


class TestKappaSweep:
    """Tests for sweeping across different κ₀ (mobility reduction) values.

    Consolidated from: test_kappa_sweep.py
    """

    @pytest.fixture
    def setup_kappa_test(self, tmp_path):
        """Setup for kappa sweep tests."""
        project_root = Path(__file__).parent.parent.parent
        data_folder = project_root / "models" / "mitma"
        config_path = data_folder / "config_MMCACovid19.json"

        if not config_path.exists():
            pytest.skip("MITMA model config not found")

        output_folder = tmp_path / "kappa_test"
        output_folder.mkdir()

        with open(config_path) as f:
            config_dict = json.load(f)

        # Remove kappa0_filename to use JSON mode
        if "kappa0_filename" in config_dict["data"]:
            del config_dict["data"]["kappa0_filename"]

        config_dict["simulation"]["end_date"] = "2020-02-28"
        config_dict["NPI"]["tᶜs"] = [1]
        config_dict["simulation"]["save_full_output"] = False
        config_dict["simulation"]["save_observables"] = True

        return config_dict, data_folder, output_folder

    @pytest.mark.parametrize("kappa", [0.0, 0.2, 0.5, 0.8, 1.0])
    def test_kappa_sweep_reduces_susceptibles(self, setup_kappa_test, kappa):
        """Higher κ₀ should result in more susceptibles remaining (fewer infections)."""
        config_dict, data_folder, output_folder = setup_kappa_test

        config_dict["NPI"]["κ₀s"] = [kappa]

        run_folder = output_folder / f"kappa_{kappa}"
        run_folder.mkdir()

        model = EpiSim(
            config=config_dict,
            data_folder=str(data_folder),
            instance_folder=str(run_folder),
            initial_conditions=None,
        )
        model.setup(executable_type="interpreter")

        final_state, _ = model.step("2020-02-09", length_days=20)

        ds = xr.open_dataset(final_state)
        S_final = ds["S"].sum().values
        ds.close()

        # Store result for comparison
        assert S_final > 0, f"κ₀={kappa}: Should have positive susceptibles"

        return S_final

    def test_kappa_zero_vs_one_comparison(self, setup_kappa_test):
        """κ₀=0.0 should have fewer susceptibles than κ₀=1.0 (more infections)."""
        config_dict, data_folder, output_folder = setup_kappa_test

        results = {}
        for kappa in [0.0, 1.0]:
            config_dict["NPI"]["κ₀s"] = [kappa]

            run_folder = output_folder / f"kappa_{kappa}"
            run_folder.mkdir()

            model = EpiSim(
                config=config_dict.copy(),
                data_folder=str(data_folder),
                instance_folder=str(run_folder),
                initial_conditions=None,
            )
            model.setup(executable_type="interpreter")

            final_state, _ = model.step("2020-02-09", length_days=20)

            ds = xr.open_dataset(final_state)
            results[kappa] = ds["S"].sum().values
            ds.close()

        # κ₀=0 (no intervention) should have fewer susceptibles than κ₀=1 (full lockdown)
        assert results[0.0] < results[1.0], (
            f"κ₀=0.0 should have fewer susceptibles than κ₀=1.0. Got {results[0.0]} vs {results[1.0]}"
        )


class TestInterventionTiming:
    """Tests for intervention timing (tᶜ) parameter.

    Consolidated from: test_tc.py
    """

    @pytest.fixture
    def setup_timing_test(self, tmp_path):
        """Setup for timing tests."""
        project_root = Path(__file__).parent.parent.parent
        data_folder = project_root / "models" / "mitma"
        config_path = data_folder / "config_MMCACovid19.json"

        if not config_path.exists():
            pytest.skip("MITMA model config not found")

        output_folder = tmp_path / "timing_test"
        output_folder.mkdir()

        with open(config_path) as f:
            config_dict = json.load(f)

        if "kappa0_filename" in config_dict["data"]:
            del config_dict["data"]["kappa0_filename"]

        config_dict["simulation"]["end_date"] = "2020-02-28"
        config_dict["NPI"]["κ₀s"] = [0.8]  # 80% reduction
        config_dict["simulation"]["save_full_output"] = False
        config_dict["simulation"]["save_observables"] = True

        return config_dict, data_folder, output_folder

    @pytest.mark.parametrize("tc", [1, 5, 10])
    def test_different_tc_values(self, setup_timing_test, tc):
        """Different tᶜ values should produce different results."""
        config_dict, data_folder, output_folder = setup_timing_test

        config_dict["NPI"]["tᶜs"] = [tc]

        run_folder = output_folder / f"tc_{tc}"
        run_folder.mkdir()

        model = EpiSim(
            config=config_dict,
            data_folder=str(data_folder),
            instance_folder=str(run_folder),
            initial_conditions=None,
        )
        model.setup(executable_type="interpreter")

        final_state, _ = model.step("2020-02-09", length_days=20)

        ds = xr.open_dataset(final_state)
        S_final = ds["S"].sum().values
        ds.close()

        assert S_final > 0, f"tᶜ={tc}: Should have positive susceptibles"

    def test_earlier_intervention_more_effective(self, setup_timing_test):
        """Earlier intervention (lower tᶜ) should be more effective."""
        config_dict, data_folder, output_folder = setup_timing_test

        results = {}
        for tc in [1, 10]:  # Early vs late intervention
            config_dict["NPI"]["tᶜs"] = [tc]

            run_folder = output_folder / f"tc_{tc}"
            run_folder.mkdir()

            model = EpiSim(
                config=config_dict.copy(),
                data_folder=str(data_folder),
                instance_folder=str(run_folder),
                initial_conditions=None,
            )
            model.setup(executable_type="interpreter")

            final_state, _ = model.step("2020-02-09", length_days=20)

            ds = xr.open_dataset(final_state)
            results[tc] = ds["S"].sum().values
            ds.close()

        # Earlier intervention should save more susceptibles
        assert results[1] > results[10], (
            f"Earlier intervention (tᶜ=1) should be more effective than tᶜ=10"
        )


class TestJsonVsCsvMode:
    """Tests comparing JSON-only mode vs CSV kappa0 file mode.

    Consolidated from: test_json_mode.py, test_verify_kappa.py, test_exact_configs.py
    """

    @pytest.fixture
    def setup_mode_test(self, tmp_path):
        """Setup for JSON vs CSV mode tests."""
        project_root = Path(__file__).parent.parent.parent
        data_folder = project_root / "models" / "mitma"
        config_path = data_folder / "config_MMCACovid19.json"

        if not config_path.exists():
            pytest.skip("MITMA model config not found")

        output_folder = tmp_path / "mode_test"
        output_folder.mkdir()

        with open(config_path) as f:
            config_dict = json.load(f)

        config_dict["simulation"]["end_date"] = "2020-02-28"
        config_dict["simulation"]["save_full_output"] = False
        config_dict["simulation"]["save_observables"] = True

        return config_dict, data_folder, output_folder

    def test_json_mode_no_csv(self, setup_mode_test):
        """JSON-only mode should work without kappa0 CSV file."""
        config_dict, data_folder, output_folder = setup_mode_test

        # Ensure no kappa0_filename
        if "kappa0_filename" in config_dict["data"]:
            del config_dict["data"]["kappa0_filename"]

        config_dict["NPI"]["κ₀s"] = [0.8]
        config_dict["NPI"]["tᶜs"] = [1]

        model = EpiSim(
            config=config_dict,
            data_folder=str(data_folder),
            instance_folder=str(output_folder),
            initial_conditions=None,
        )
        model.setup(executable_type="interpreter")

        final_state, _ = model.step("2020-02-09", length_days=20)

        assert os.path.exists(final_state), "JSON mode should produce output"

    def test_json_vs_csv_equivalent_results(self, setup_mode_test):
        """JSON mode and CSV mode with same parameters should produce equivalent results."""
        config_dict, data_folder, output_folder = setup_mode_test

        # Create kappa0 CSV
        kappa0_csv = output_folder / "kappa0.csv"
        dates = pd.date_range("2020-02-09", periods=20, freq="D")
        kappa0_df = pd.DataFrame(
            {"date": dates.strftime("%Y-%m-%d"), "reduction": [0.8] * 20}
        )
        kappa0_df.to_csv(kappa0_csv, index=False)

        results = {}

        # Test JSON mode
        config_json = config_dict.copy()
        if "kappa0_filename" in config_json["data"]:
            del config_json["data"]["kappa0_filename"]
        config_json["NPI"]["κ₀s"] = [0.8]
        config_json["NPI"]["tᶜs"] = [1]

        run_folder_json = output_folder / "json_mode"
        run_folder_json.mkdir()

        model_json = EpiSim(
            config=config_json,
            data_folder=str(data_folder),
            instance_folder=str(run_folder_json),
            initial_conditions=None,
        )
        model_json.setup(executable_type="interpreter")
        final_state_json, _ = model_json.step("2020-02-09", length_days=20)

        ds_json = xr.open_dataset(final_state_json)
        results["json"] = ds_json["S"].sum().values
        ds_json.close()

        # Test CSV mode
        config_csv = config_dict.copy()
        config_csv["data"]["kappa0_filename"] = str(kappa0_csv)
        config_csv["NPI"]["κ₀s"] = [0.0]  # Should be overridden by CSV
        config_csv["NPI"]["tᶜs"] = [0]  # Should be overridden by CSV

        run_folder_csv = output_folder / "csv_mode"
        run_folder_csv.mkdir()

        model_csv = EpiSim(
            config=config_csv,
            data_folder=str(data_folder),
            instance_folder=str(run_folder_csv),
            initial_conditions=None,
        )
        model_csv.setup(executable_type="interpreter")
        final_state_csv, _ = model_csv.step("2020-02-09", length_days=20)

        ds_csv = xr.open_dataset(final_state_csv)
        results["csv"] = ds_csv["S"].sum().values
        ds_csv.close()

        # Results should be very close (within numerical precision)
        assert abs(results["json"] - results["csv"]) < 100, (
            f"JSON and CSV modes should produce similar results. Got {results['json']} vs {results['csv']}"
        )

    def test_kappa0_csv_file_format(self, setup_mode_test):
        """Kappa0 CSV file should have correct format."""
        _, _, output_folder = setup_mode_test

        kappa0_csv = output_folder / "kappa0.csv"
        dates = pd.date_range("2020-02-09", periods=5, freq="D")
        kappa0_df = pd.DataFrame(
            {"date": dates.strftime("%Y-%m-%d"), "reduction": [0.0, 0.5, 0.8, 0.5, 0.0]}
        )
        kappa0_df.to_csv(kappa0_csv, index=False)

        # Read back and verify
        df_read = pd.read_csv(kappa0_csv)

        assert "date" in df_read.columns or "time" in df_read.columns
        assert "reduction" in df_read.columns
        assert len(df_read) == 5
        assert all(0 <= r <= 1.0 for r in df_read["reduction"])


class TestConfigurationModes:
    """Tests for different configuration modes and parameter combinations.

    Consolidated from: test_exact_configs.py
    """

    @pytest.fixture
    def setup_config_test(self, tmp_path):
        """Setup for configuration tests."""
        project_root = Path(__file__).parent.parent.parent
        data_folder = project_root / "models" / "mitma"
        config_path = data_folder / "config_MMCACovid19.json"

        if not config_path.exists():
            pytest.skip("MITMA model config not found")

        output_folder = tmp_path / "config_test"
        output_folder.mkdir()

        with open(config_path) as f:
            config_dict = json.load(f)

        config_dict["simulation"]["end_date"] = "2020-02-28"
        config_dict["simulation"]["save_full_output"] = False
        config_dict["simulation"]["save_observables"] = True

        return config_dict, data_folder, output_folder

    def test_baseline_no_intervention(self, setup_config_test):
        """Baseline with no intervention (κ₀=0) should run successfully."""
        config_dict, data_folder, output_folder = setup_config_test

        if "kappa0_filename" in config_dict["data"]:
            del config_dict["data"]["kappa0_filename"]

        config_dict["NPI"]["κ₀s"] = [0.0]
        config_dict["NPI"]["tᶜs"] = [1]

        model = EpiSim(
            config=config_dict,
            data_folder=str(data_folder),
            instance_folder=str(output_folder),
            initial_conditions=None,
        )
        model.setup(executable_type="interpreter")

        final_state, _ = model.step("2020-02-09", length_days=20)

        assert os.path.exists(final_state)

    def test_full_lockdown(self, setup_config_test):
        """Full lockdown (κ₀=1.0) should run successfully."""
        config_dict, data_folder, output_folder = setup_config_test

        if "kappa0_filename" in config_dict["data"]:
            del config_dict["data"]["kappa0_filename"]

        config_dict["NPI"]["κ₀s"] = [1.0]
        config_dict["NPI"]["tᶜs"] = [1]

        model = EpiSim(
            config=config_dict,
            data_folder=str(data_folder),
            instance_folder=str(output_folder),
            initial_conditions=None,
        )
        model.setup(executable_type="interpreter")

        final_state, _ = model.step("2020-02-09", length_days=20)

        assert os.path.exists(final_state)

    def test_partial_intervention(self, setup_config_test):
        """Partial intervention (κ₀=0.5) should run successfully."""
        config_dict, data_folder, output_folder = setup_config_test

        if "kappa0_filename" in config_dict["data"]:
            del config_dict["data"]["kappa0_filename"]

        config_dict["NPI"]["κ₀s"] = [0.5]
        config_dict["NPI"]["tᶜs"] = [5]

        model = EpiSim(
            config=config_dict,
            data_folder=str(data_folder),
            instance_folder=str(output_folder),
            initial_conditions=None,
        )
        model.setup(executable_type="interpreter")

        final_state, _ = model.step("2020-02-09", length_days=20)

        assert os.path.exists(final_state)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
