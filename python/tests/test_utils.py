"""
Tests for utility functions in episim_utils
"""

import json
import os

import numpy as np
import xarray as xr

from episim_python.epi_sim import date_addition
from episim_python.episim_utils import compute_observables, update_params

from .conftest import BaseTestCase


class TestUpdateParams(BaseTestCase):
    """Test cases for update_params function"""

    def test_update_beta_I(self):
        """Test updating βᴵ parameter"""
        params = {"epidemic_params": {}}
        update_dict = {"βᴵ": 0.12}

        result = update_params(params, update_dict)
        assert result["epidemic_params"]["βᴵ"] == 0.12

    def test_update_beta_alias(self):
        """Test updating β parameter (alias for βᴵ)"""
        params = {"epidemic_params": {}}
        update_dict = {"β": 0.10}

        result = update_params(params, update_dict)
        assert result["epidemic_params"]["βᴵ"] == 0.10

    def test_update_beta_A_direct(self):
        """Test updating βᴬ parameter directly"""
        params = {"epidemic_params": {}}
        update_dict = {"βᴬ": 0.05}

        result = update_params(params, update_dict)
        assert result["epidemic_params"]["βᴬ"] == 0.05

    def test_update_beta_A_scale(self):
        """Test updating βᴬ using scale_β"""
        params = {"epidemic_params": {"βᴵ": 0.10}}
        update_dict = {"scale_β": 0.5}

        result = update_params(params, update_dict)
        assert result["epidemic_params"]["βᴬ"] == 0.05  # 0.10 * 0.5

    def test_update_eta_g_uniform(self):
        """Test updating ηᵍ with uniform values"""
        params = {"epidemic_params": {}}
        update_dict = {"ηᵍ": 0.4}

        result = update_params(params, update_dict)
        assert result["epidemic_params"]["ηᵍ"] == [0.4, 0.4, 0.4]

    def test_update_eta_g_from_tau_inc_scale_ea(self):
        """Test updating ηᵍ from τ_inc and scale_ea"""
        params = {"epidemic_params": {}}
        update_dict = {"τ_inc": 4.0, "scale_ea": 0.5}

        result = update_params(params, update_dict, G=3)
        expected = 1.0 / (4.0 * (1.0 - 0.5))  # 1/(4*0.5) = 0.5
        assert result["epidemic_params"]["ηᵍ"] == [expected, expected, expected]

    def test_update_eta_g_individual(self):
        """Test updating individual ηᵍ values"""
        params = {"epidemic_params": {}}
        update_dict = {"ηᵍY": 0.3, "ηᵍM": 0.4, "ηᵍO": 0.5}

        result = update_params(params, update_dict)
        assert result["epidemic_params"]["ηᵍ"] == [0.3, 0.4, 0.5]

    def test_update_alpha_g_uniform(self):
        """Test updating αᵍ with uniform value"""
        params = {"epidemic_params": {}}
        update_dict = {"αᵍ": 0.6}

        result = update_params(params, update_dict)
        expected_0 = 0.6 / 2.5  # First group gets scaled value
        assert result["epidemic_params"]["αᵍ"] == [expected_0, 0.6, 0.6]

    def test_update_alpha_g_from_tau_params(self):
        """Test updating αᵍ from τ parameters"""
        params = {"epidemic_params": {}}
        update_dict = {"τ_inc": 4.0, "scale_ea": 0.5, "τᵢ": 2.0}

        result = update_params(params, update_dict)

        # Expected calculations
        t_inc = 4.0
        s_ea = 0.5
        ti = 2.0
        n1 = 1.0 / (ti - 1 + t_inc * s_ea)  # 1/(2-1+4*0.5) = 1/3
        n2 = 1.0 / (t_inc * s_ea)  # 1/(4*0.5) = 0.5
        n3 = 1.0 / (t_inc * s_ea)  # 1/(4*0.5) = 0.5

        assert result["epidemic_params"]["αᵍ"] == [n1, n2, n3]

    def test_update_mu_g_uniform(self):
        """Test updating μᵍ with uniform value"""
        params = {"epidemic_params": {}}
        update_dict = {"μᵍ": 0.4}

        result = update_params(params, update_dict)
        assert result["epidemic_params"]["μᵍ"] == [1, 0.4, 0.4]

    def test_update_mu_g_from_tau_i(self):
        """Test updating μᵍ from τᵢ"""
        params = {"epidemic_params": {}}
        update_dict = {"τᵢ": 3.0}

        result = update_params(params, update_dict)
        expected = 1.0 / 3.0
        assert result["epidemic_params"]["μᵍ"] == [1.0, expected, expected]

    def test_update_gamma_g_individual(self):
        """Test updating individual γᵍ values"""
        params = {"epidemic_params": {"γᵍ": [0.001, 0.002, 0.003]}}
        update_dict = {"γᵍY": 0.005, "γᵍM": 0.010, "γᵍO": 0.080}

        result = update_params(params, update_dict)
        assert result["epidemic_params"]["γᵍ"] == [0.005, 0.010, 0.080]

    def test_update_npi_phi_s_single(self):
        """Test updating ϕs with single value"""
        params = {"NPI": {}}
        update_dict = {"ϕs": 0.3}

        result = update_params(params, update_dict)
        assert result["NPI"]["ϕs"] == [0.3]

    def test_update_npi_phi_s_list(self):
        """Test updating ϕs with list of values"""
        params = {"NPI": {}}
        update_dict = {"ϕs": [0.3, 0.2, 0.1]}

        result = update_params(params, update_dict)
        assert result["NPI"]["ϕs"] == [0.3, 0.2, 0.1]

    def test_update_npi_phi_s_numbered(self):
        """Test updating ϕs with numbered parameters"""
        params = {"NPI": {}}
        update_dict = {"ϕs1": 0.3, "ϕs2": 0.2, "ϕs3": 0.1, "ϕs4": 0.05}

        result = update_params(params, update_dict)
        assert result["NPI"]["ϕs"] == [0.3, 0.2, 0.1, 0.05]

    def test_update_npi_delta_s(self):
        """Test updating δs parameters"""
        params = {"NPI": {}}
        update_dict = {"δs": 0.7}

        result = update_params(params, update_dict)
        assert result["NPI"]["δs"] == [0.7]

    def test_update_vaccination_params(self):
        """Test updating vaccination parameters"""
        params = {"vaccination": {}}
        update_dict = {
            "ϵᵍ": [0.3, 0.5, 0.7],
            "percentage_of_vacc_per_day": 0.01,
            "start_vacc": 60,
            "dur_vacc": 180,
        }

        result = update_params(params, update_dict)
        assert result["vaccination"]["ϵᵍ"] == [0.3, 0.5, 0.7]
        assert result["vaccination"]["percentage_of_vacc_per_day"] == 0.01
        assert result["vaccination"]["start_vacc"] == 60
        assert result["vaccination"]["dur_vacc"] == 180

    def test_update_initial_condition_filename(self):
        """Test updating initial condition filename"""
        params = {"data": {}}
        update_dict = {"initial_condition_filename": "new_initial.nc"}

        result = update_params(params, update_dict)
        assert result["data"]["initial_condition_filename"] == "new_initial.nc"

    def test_update_multiple_params(self):
        """Test updating multiple parameters at once"""
        params = {"epidemic_params": {"γᵍ": [0.001, 0.002, 0.003]}, "NPI": {}}
        update_dict = {"βᴵ": 0.12, "γᵍY": 0.005, "ϕs": 0.3}

        result = update_params(params, update_dict)
        assert result["epidemic_params"]["βᴵ"] == 0.12
        assert result["epidemic_params"]["γᵍ"][0] == 0.005
        assert result["NPI"]["ϕs"] == [0.3]


class TestComputeObservables(BaseTestCase):
    """Test cases for compute_observables function"""

    def test_compute_observables_basic(self, temp_dir):
        """Test basic computation of observables"""
        # Create test configuration file
        config = {
            "population_params": {"G_labels": ["Y", "M", "O"]},
            "epidemic_params": {
                "αᵍ": [0.25, 0.6, 0.6],
                "μᵍ": [1.0, 0.3, 0.3],
                "θᵍ": [0.0, 0.0, 0.0],
                "γᵍ": [0.003, 0.01, 0.08],
            },
        }

        config_path = os.path.join(temp_dir, "episim_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Create test simulation data with correct dimension order for compute_observables
        # Function expects dimensions: [anything, anything, anything, epi_states]
        data = np.random.rand(
            3,
            2,
            5,
            10,
        )  # 3 age groups, 2 regions, 5 time steps, 10 states
        coords = {
            "G": ["Y", "M", "O"],
            "M": ["region_1", "region_2"],
            "T": np.arange(5),
            "epi_states": ["S", "E", "A", "I", "PH", "PD", "HR", "HD", "R", "D"],
        }
        sim_xa = xr.DataArray(data, coords=coords, dims=["G", "M", "T", "epi_states"])

        result = compute_observables(sim_xa, temp_dir, temp_dir)

        # Check result structure
        assert isinstance(result, xr.DataArray)
        assert list(result.coords["epi_states"]) == ["I", "H", "D"]
        assert list(result.coords["G"]) == ["Y", "M", "O"]
        assert result.dims == ("epi_states", "G", "M", "T")

        # Check that values were modified (should be different from original)
        original_a = sim_xa.sel(epi_states="A")
        result_i = result.sel(epi_states="I")

        # Values should be scaled by alphas
        expected_y = original_a.sel(G="Y") * 0.25
        np.testing.assert_array_almost_equal(
            result_i.sel(G="Y").values,
            expected_y.values,
        )


class TestDateAddition(BaseTestCase):
    """Test cases for date_addition function"""

    def test_date_addition_basic(self):
        """Test basic date addition"""
        result = date_addition("2020-01-01", 7)
        assert result == "2020-01-08"

    def test_date_addition_month_boundary(self):
        """Test date addition across month boundary"""
        result = date_addition("2020-01-25", 10)
        assert result == "2020-02-04"

    def test_date_addition_year_boundary(self):
        """Test date addition across year boundary"""
        result = date_addition("2020-12-25", 10)
        assert result == "2021-01-04"

    def test_date_addition_leap_year(self):
        """Test date addition in leap year"""
        result = date_addition("2020-02-25", 10)
        assert result == "2020-03-06"  # 2020 is a leap year

    def test_date_addition_zero_days(self):
        """Test date addition with zero days"""
        result = date_addition("2020-01-01", 0)
        assert result == "2020-01-01"

    def test_date_addition_negative_days(self):
        """Test date addition with negative days (subtraction)"""
        result = date_addition("2020-01-10", -5)
        assert result == "2020-01-05"

    def test_date_addition_different_formats(self):
        """Test that output format is consistent"""
        result1 = date_addition("2020-1-1", 1)  # Single digit month/day
        result2 = date_addition("2020-01-01", 1)  # Double digit month/day

        assert result1 == "2020-01-02"
        assert result2 == "2020-01-02"
        assert result1 == result2
