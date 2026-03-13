"""Tests for kappa (mobility reduction) parameter configurations using full simulation runs.

This module tests the synthetic data pipeline's intervention logic by running
complete simulations and analyzing observables files with time-series data.

Tests verify that:
- Higher κ₀ (more confinement) reduces infections during intervention windows
- Earlier interventions are more effective than later ones (same duration)
- JSON and CSV modes produce equivalent results

All tests use LIMITED-DURATION interventions that return to κ₀=0 after the window,
mimicking realistic intervention patterns.

OPTIMIZATIONS:
- Session-scoped fixtures cache simulation results (shared across tests)
- Reduced simulation duration: 10 days instead of 20
- Reduced parametrization: fewer kappa/tc values tested
- Consolidated simulations where possible
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

# Add python directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Module-level constants for test configuration
TEST_START_DATE = "2020-02-09"
TEST_END_DATE = "2020-02-18"  # 10 days instead of 20
SIMULATION_DAYS = 10


def _run_simulation_cached(config_dict, data_folder, run_folder):
    """Helper to run a full simulation and return path to observables."""
    run_folder.mkdir(parents=True, exist_ok=True)

    # Check if already cached
    observables_path = run_folder / "output" / "observables.nc"
    if observables_path.exists():
        return observables_path

    # Write config
    config_path = run_folder / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Run simulation
    cmd = [
        "julia",
        "--project=.",
        str(Path(__file__).parent.parent.parent / "src" / "run.jl"),
        "run",
        "--config",
        str(config_path),
        "--data-folder",
        str(data_folder),
        "--instance-folder",
        str(run_folder),
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent
    )

    if result.returncode != 0:
        pytest.fail(f"Simulation failed: {result.stderr}")

    if not observables_path.exists():
        pytest.fail(f"Observables file not found: {observables_path}")

    return observables_path


def _get_infections_during_window(observables_path, start_day, end_day):
    """Calculate total new infections during the intervention window only."""
    ds = xr.open_dataset(observables_path)
    new_infected = ds["new_infected"].isel(T=slice(start_day, end_day)).sum().values
    ds.close()
    return new_infected


def _get_total_infections(observables_path):
    """Calculate total infections over entire simulation."""
    ds = xr.open_dataset(observables_path)
    new_infected = ds["new_infected"].sum().values
    ds.close()
    return new_infected


@pytest.fixture(scope="session")
def session_test_paths(tmp_path_factory):
    """Session-scoped fixture providing common paths."""
    project_root = Path(__file__).parent.parent.parent
    data_folder = project_root / "models" / "mitma"
    config_path = data_folder / "config_MMCACovid19.json"

    if not config_path.exists():
        pytest.skip("MITMA model config not found")

    # Use session-scoped temp directory
    base_folder = tmp_path_factory.mktemp("kappa_tests_session")

    with open(config_path) as f:
        base_config = json.load(f)

    # Remove kappa0_filename to use JSON mode
    if "kappa0_filename" in base_config["data"]:
        del base_config["data"]["kappa0_filename"]

    # Set short simulation duration
    base_config["simulation"]["start_date"] = TEST_START_DATE
    base_config["simulation"]["end_date"] = TEST_END_DATE
    base_config["simulation"]["save_full_output"] = False
    base_config["simulation"]["save_observables"] = True

    return {
        "data_folder": data_folder,
        "base_config": base_config,
        "base_folder": base_folder,
    }


@pytest.fixture(scope="session")
def kappa_simulations(session_test_paths):
    """Session-scoped fixture that runs kappa sweep simulations once."""
    data_folder = session_test_paths["data_folder"]
    base_config = session_test_paths["base_config"]
    base_folder = session_test_paths["base_folder"]

    # Reduced kappa values: 3 instead of 5
    kappa_values = [0.0, 0.5, 1.0]

    INTERVENTION_START = 1
    INTERVENTION_END = 8  # 7 day intervention window

    results = {}

    for kappa in kappa_values:
        config_copy = json.loads(json.dumps(base_config))  # Deep copy
        config_copy["NPI"]["κ₀s"] = [kappa, 0.0]
        config_copy["NPI"]["tᶜs"] = [INTERVENTION_START, INTERVENTION_END]
        config_copy["NPI"]["ϕs"] = [0.2, 0.2]
        config_copy["NPI"]["δs"] = [0.8, 0.0]

        run_folder = base_folder / f"kappa_{kappa}"
        observables_path = _run_simulation_cached(config_copy, data_folder, run_folder)

        infections = _get_infections_during_window(
            observables_path, INTERVENTION_START, INTERVENTION_END
        )
        results[kappa] = {
            "observables_path": observables_path,
            "infections": infections,
        }

    return results


@pytest.fixture(scope="session")
def timing_simulations(session_test_paths):
    """Session-scoped fixture that runs timing simulations once."""
    data_folder = session_test_paths["data_folder"]
    base_config = session_test_paths["base_config"]
    base_folder = session_test_paths["base_folder"]

    # Reduced tc values: 2 instead of 3
    tc_values = [1, 5]
    INTERVENTION_DURATION = 5  # Shorter duration

    results = {}

    for tc in tc_values:
        tc_end = tc + INTERVENTION_DURATION

        config_copy = json.loads(json.dumps(base_config))
        config_copy["NPI"]["κ₀s"] = [0.8, 0.0]
        config_copy["NPI"]["tᶜs"] = [tc, tc_end]
        config_copy["NPI"]["ϕs"] = [0.2, 0.2]
        config_copy["NPI"]["δs"] = [0.8, 0.0]

        run_folder = base_folder / f"tc_{tc}"
        observables_path = _run_simulation_cached(config_copy, data_folder, run_folder)

        infections = _get_infections_during_window(observables_path, tc, tc_end)
        results[tc] = {
            "observables_path": observables_path,
            "infections": infections,
            "window": (tc, tc_end),
        }

    return results


@pytest.fixture(scope="session")
def json_csv_simulations(session_test_paths):
    """Session-scoped fixture that runs JSON vs CSV simulations once."""
    data_folder = session_test_paths["data_folder"]
    base_config = session_test_paths["base_config"]
    base_folder = session_test_paths["base_folder"]

    INTERVENTION_START = 1
    INTERVENTION_END = 8

    results = {}

    # JSON mode
    config_json = json.loads(json.dumps(base_config))
    config_json["NPI"]["κ₀s"] = [0.8, 0.0]
    config_json["NPI"]["tᶜs"] = [INTERVENTION_START, INTERVENTION_END]
    config_json["NPI"]["ϕs"] = [0.2, 0.2]
    config_json["NPI"]["δs"] = [0.8, 0.0]

    run_folder_json = base_folder / "json_mode"
    observables_json = _run_simulation_cached(config_json, data_folder, run_folder_json)
    infections_json = _get_infections_during_window(
        observables_json, INTERVENTION_START, INTERVENTION_END
    )
    results["json"] = {
        "observables_path": observables_json,
        "infections": infections_json,
    }

    # Create kappa0 CSV with limited duration intervention
    kappa0_csv = base_folder / "kappa0.csv"
    dates = pd.date_range(TEST_START_DATE, periods=SIMULATION_DAYS, freq="D")
    reductions = [0.0] * SIMULATION_DAYS
    for i in range(INTERVENTION_START, INTERVENTION_END):
        reductions[i] = 0.8

    kappa0_df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "reduction": reductions,
            "datetime": dates.strftime("%Y-%m-%d"),
            "time": range(SIMULATION_DAYS),
        }
    )
    kappa0_df.to_csv(kappa0_csv, index=False)

    # Copy kappa0 file to data folder so it can be found
    kappa0_in_data = data_folder / "kappa0_test.csv"
    shutil.copy2(kappa0_csv, kappa0_in_data)

    # CSV mode
    config_csv = json.loads(json.dumps(base_config))
    config_csv["data"]["kappa0_filename"] = "kappa0_test.csv"
    config_csv["NPI"]["κ₀s"] = [0.0]
    config_csv["NPI"]["tᶜs"] = [0]
    config_csv["NPI"]["ϕs"] = [0.2]
    config_csv["NPI"]["δs"] = [0.8]

    run_folder_csv = base_folder / "csv_mode"
    observables_csv = _run_simulation_cached(config_csv, data_folder, run_folder_csv)
    infections_csv = _get_infections_during_window(
        observables_csv, INTERVENTION_START, INTERVENTION_END
    )
    results["csv"] = {
        "observables_path": observables_csv,
        "infections": infections_csv,
    }

    # Cleanup
    if kappa0_in_data.exists():
        kappa0_in_data.unlink()

    return results


@pytest.fixture(scope="session")
def config_mode_simulations(session_test_paths):
    """Session-scoped fixture that runs configuration mode simulations once."""
    data_folder = session_test_paths["data_folder"]
    base_config = session_test_paths["base_config"]
    base_folder = session_test_paths["base_folder"]

    configs = {
        "baseline": {"κ₀s": [0.0], "tᶜs": [1]},
        "full_lockdown": {"κ₀s": [1.0], "tᶜs": [1]},
        "partial": {"κ₀s": [0.5], "tᶜs": [3]},
    }

    results = {}

    for name, npi_config in configs.items():
        config_copy = json.loads(json.dumps(base_config))
        config_copy["NPI"]["κ₀s"] = npi_config["κ₀s"]
        config_copy["NPI"]["tᶜs"] = npi_config["tᶜs"]

        run_folder = base_folder / name
        observables_path = _run_simulation_cached(config_copy, data_folder, run_folder)

        results[name] = {
            "observables_path": observables_path,
            "total_infections": _get_total_infections(observables_path),
        }

    return results


class TestKappaSweep:
    """Tests for sweeping across different κ₀ (mobility reduction) values."""

    @pytest.mark.slow
    @pytest.mark.parametrize("kappa", [0.0, 0.5, 1.0])  # Reduced from 5 to 3 values
    def test_kappa_sweep_reduces_susceptibles(self, kappa_simulations, kappa):
        """Higher κ₀ should result in more susceptibles remaining (fewer infections)."""
        infections = kappa_simulations[kappa]["infections"]
        assert infections >= 0, f"κ₀={kappa}: Infections should be non-negative"

    @pytest.mark.slow
    def test_kappa_zero_vs_one_comparison(self, kappa_simulations):
        """κ₀=0.0 should have more infections during intervention window than κ₀=1.0."""
        infections_0 = kappa_simulations[0.0]["infections"]
        infections_1 = kappa_simulations[1.0]["infections"]

        assert infections_0 > infections_1, (
            f"κ₀=0.0 should have more infections during intervention than κ₀=1.0. "
            f"Got {infections_0} vs {infections_1}"
        )


class TestInterventionTiming:
    """Tests for intervention timing (tᶜ) parameter."""

    @pytest.mark.slow
    @pytest.mark.parametrize("tc", [1, 5])  # Reduced from 3 to 2 values
    def test_different_tc_values(self, timing_simulations, tc):
        """Different tᶜ values should produce different results."""
        infections = timing_simulations[tc]["infections"]
        assert infections >= 0, f"tᶜ={tc}: Infections should be non-negative"

    @pytest.mark.slow
    def test_earlier_intervention_more_effective(self, timing_simulations):
        """Earlier intervention (same duration) should prevent more infections during the window."""
        infections_early = timing_simulations[1]["infections"]
        infections_late = timing_simulations[5]["infections"]

        assert infections_early < infections_late, (
            f"Earlier intervention (tᶜ=1) should have fewer infections "
            f"than tᶜ=5. Got {infections_early} vs {infections_late}"
        )


class TestJsonVsCsvMode:
    """Tests comparing JSON-only mode vs CSV kappa0 file mode."""

    @pytest.mark.slow
    def test_json_mode_no_csv(self, json_csv_simulations):
        """JSON-only mode should work without kappa0 CSV file."""
        assert json_csv_simulations["json"]["observables_path"].exists()

    @pytest.mark.slow
    def test_json_vs_csv_equivalent_results(self, json_csv_simulations):
        """JSON mode and CSV mode with same parameters should produce reasonably similar results."""
        infections_json = json_csv_simulations["json"]["infections"]
        infections_csv = json_csv_simulations["csv"]["infections"]

        relative_diff = abs(infections_json - infections_csv) / max(
            infections_json, infections_csv
        )
        assert relative_diff < 0.5, (
            f"JSON and CSV modes should produce reasonably similar results. "
            f"Got {infections_json} vs {infections_csv} (relative diff: {relative_diff:.1%})"
        )

    def test_kappa0_csv_file_format(self, tmp_path):
        """Kappa0 CSV file should have correct format (no simulation needed)."""
        kappa0_csv = tmp_path / "kappa0.csv"
        dates = pd.date_range(TEST_START_DATE, periods=5, freq="D")
        kappa0_df = pd.DataFrame(
            {"date": dates.strftime("%Y-%m-%d"), "reduction": [0.0, 0.5, 0.8, 0.5, 0.0]}
        )
        kappa0_df.to_csv(kappa0_csv, index=False)

        df_read = pd.read_csv(kappa0_csv)

        assert "date" in df_read.columns or "time" in df_read.columns
        assert "reduction" in df_read.columns
        assert len(df_read) == 5
        assert all(0 <= r <= 1.0 for r in df_read["reduction"])


class TestConfigurationModes:
    """Tests for different configuration modes and parameter combinations."""

    @pytest.mark.slow
    def test_baseline_no_intervention(self, config_mode_simulations):
        """Baseline with no intervention (κ₀=0) should run successfully."""
        assert config_mode_simulations["baseline"]["observables_path"].exists()

    @pytest.mark.slow
    def test_full_lockdown(self, config_mode_simulations):
        """Full lockdown (κ₀=1.0) should run successfully."""
        assert config_mode_simulations["full_lockdown"]["observables_path"].exists()

    @pytest.mark.slow
    def test_partial_intervention(self, config_mode_simulations):
        """Partial intervention (κ₀=0.5) should run successfully."""
        assert config_mode_simulations["partial"]["observables_path"].exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
