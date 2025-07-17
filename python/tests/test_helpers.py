"""
Test helper utilities for episim_python tests

This module provides common utilities, fixtures, and helper functions
to reduce code duplication across test files.
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
from typing import Dict, Any, Tuple, Optional, List

import pytest
import pandas as pd
import numpy as np
import xarray as xr

from episim_python import EpiSim, EpiSimConfig


class MockResult:
    """Mock subprocess result for testing"""
    
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestHelpers:
    """Collection of test helper methods"""
    
    @staticmethod
    def create_mock_success_result(output: str = "Simulation completed successfully") -> MockResult:
        """Create a mock successful subprocess result"""
        return MockResult(returncode=0, stdout=output, stderr="")
    
    @staticmethod
    def create_mock_failure_result(
        error_msg: str = "Simulation failed", 
        output: str = "Some output"
    ) -> MockResult:
        """Create a mock failed subprocess result"""
        return MockResult(returncode=1, stdout=output, stderr=error_msg)
    
    @staticmethod
    def create_minimal_config(group_size: int = 3) -> Dict[str, Any]:
        """Create a minimal valid configuration with specified group size"""
        group_labels = ["Y", "M", "O"][:group_size] if group_size <= 3 else [f"G{i}" for i in range(group_size)]
        
        # Create group-dependent parameters
        eta_g = [0.3] * group_size
        alpha_g = [0.25 if i == 0 else 0.6 for i in range(group_size)]
        mu_g = [1.0 if i == 0 else 0.3 for i in range(group_size)]
        theta_g = [0.0] * group_size
        gamma_g = [0.003 if i == 0 else (0.01 if i == 1 else 0.08) for i in range(group_size)]
        zeta_g = [0.13] * group_size
        lambda_g = [1.0] * group_size
        omega_g = [0.0 if i == 0 else (0.04 if i == 1 else 0.3) for i in range(group_size)]
        psi_g = [0.14] * group_size
        chi_g = [0.05] * group_size
        
        # Create contact matrix
        contact_matrix = [[0.6 if i == j else 0.4/(group_size-1) for j in range(group_size)] for i in range(group_size)]
        
        k_g = [12.0 if i == 0 else (13.0 if i == 1 else 7.0) for i in range(group_size)]
        k_g_h = [3.0] * group_size
        k_g_w = [2.0 if i == 0 else (5.0 if i == 1 else 0.0) for i in range(group_size)]
        p_g = [0.0 if i == 0 else (1.0 if i == 1 else 0.0) for i in range(group_size)]
        
        return {
            "simulation": {
                "engine": "MMCACovid19",
                "start_date": "2020-01-01",
                "end_date": "2020-01-10",
                "save_full_output": True,
                "output_format": "netcdf",
            },
            "data": {
                "initial_condition_filename": "test_initial.nc",
                "metapopulation_data_filename": "test_metapop.csv",
                "mobility_matrix_filename": "test_mobility.csv",
                "kappa0_filename": "test_kappa0.csv",
            },
            "epidemic_params": {
                "βᴵ": 0.09,
                "βᴬ": 0.045,
                "ηᵍ": eta_g,
                "αᵍ": alpha_g,
                "μᵍ": mu_g,
                "θᵍ": theta_g,
                "γᵍ": gamma_g,
                "ζᵍ": zeta_g,
                "λᵍ": lambda_g,
                "ωᵍ": omega_g,
                "ψᵍ": psi_g,
                "χᵍ": chi_g,
            },
            "population_params": {
                "G_labels": group_labels,
                "C": contact_matrix,
                "kᵍ": k_g,
                "kᵍ_h": k_g_h,
                "kᵍ_w": k_g_w,
                "pᵍ": p_g,
                "ξ": 0.01,
                "σ": 2.5,
            },
            "NPI": {
                "κ₀s": [0.8],
                "ϕs": [0.2],
                "δs": [0.8],
                "tᶜs": [50],
                "are_there_npi": True,
            },
        }
    
    @staticmethod
    def create_test_csv_content(
        regions: List[str] = None,
        age_groups: List[str] = None,
        include_totals: bool = True
    ) -> str:
        """Create test CSV content for metapopulation data"""
        if regions is None:
            regions = ["region_1", "region_2", "region_3"]
        if age_groups is None:
            age_groups = ["Y", "M", "O"]
        
        header = ["id", "area"] + age_groups
        if include_totals:
            header.append("total")
        
        lines = [",".join(header)]
        
        # Match the original test fixture values exactly
        test_data = {
            "region_1": {"area": 100.0, "Y": 5000, "M": 8000, "O": 3000},
            "region_2": {"area": 85.0, "Y": 4500, "M": 7500, "O": 2800},
            "region_3": {"area": 120.0, "Y": 6000, "M": 9000, "O": 3500},
        }
        
        for i, region in enumerate(regions):
            if region in test_data:
                area = test_data[region]["area"]
                populations = [test_data[region][group] for group in age_groups]
            else:
                # Fallback for custom regions
                area = 100.0 + i * 5.0
                populations = [5000 + i * 500 + j * 100 for j in range(len(age_groups))]
            
            row = [region, str(area)] + [str(pop) for pop in populations]
            if include_totals:
                row.append(str(sum(populations)))
            
            lines.append(",".join(row))
        
        return "\n".join(lines)
    
    @staticmethod
    def create_test_rosetta_content(
        regions: List[str] = None,
        provinces: List[str] = None,
        higher_regions: List[str] = None
    ) -> str:
        """Create test rosetta mapping content"""
        if regions is None:
            regions = ["region_1", "region_2", "region_3"]
        if provinces is None:
            provinces = ["prov_A", "prov_B", "prov_A"]
        if higher_regions is None:
            higher_regions = ["reg_X", "reg_Y", "reg_X"]
        
        header = "level_1,province,region"
        lines = [header]
        
        for region, province, higher_region in zip(regions, provinces, higher_regions):
            lines.append(f"{region},{province},{higher_region}")
        
        return "\n".join(lines)
    
    @staticmethod
    def setup_episim_model(
        config: Dict[str, Any],
        temp_dir: str,
        instance_folder: str = None,
        initial_conditions: str = None
    ) -> EpiSim:
        """Set up an EpiSim model with standard configuration"""
        if instance_folder is None:
            instance_folder = os.path.join(temp_dir, "instances")
        
        if not os.path.exists(instance_folder):
            os.makedirs(instance_folder)
        
        return EpiSim(config, temp_dir, instance_folder, initial_conditions)
    
    @staticmethod
    def setup_episim_with_interpreter(
        config: Dict[str, Any],
        temp_dir: str,
        instance_folder: str = None,
        mock_julia_path: str = "/usr/bin/julia"
    ) -> EpiSim:
        """Set up an EpiSim model with interpreter mode"""
        from unittest.mock import patch
        
        model = TestHelpers.setup_episim_model(config, temp_dir, instance_folder)
        
        with patch("shutil.which", return_value=mock_julia_path):
            model.setup(executable_type="interpreter")
        
        return model
    
    @staticmethod
    def setup_episim_with_compiled(
        config: Dict[str, Any],
        temp_dir: str,
        instance_folder: str = None,
        executable_name: str = "episim"
    ) -> EpiSim:
        """Set up an EpiSim model with compiled mode"""
        model = TestHelpers.setup_episim_model(config, temp_dir, instance_folder)
        
        # Create dummy executable
        executable_path = os.path.join(temp_dir, executable_name)
        Path(executable_path).write_text("#!/bin/bash\necho 'compiled episim'")
        os.chmod(executable_path, 0o755)
        
        model.setup(executable_type="compiled", executable_path=executable_path)
        return model
    
    @staticmethod
    def assert_episim_model_structure(model: EpiSim, expected_engine: str = "MMCACovid19Vac"):
        """Assert standard EpiSim model structure"""
        assert model.instance_folder is not None
        assert model.data_folder is not None
        assert len(model.uuid) > 0
        assert os.path.exists(model.model_state_folder)
        assert model.backend_engine == expected_engine
        assert os.path.exists(model.config_path)
    
    @staticmethod
    def assert_config_structure(config: EpiSimConfig, expected_groups: List[str] = None):
        """Assert standard EpiSimConfig structure"""
        if expected_groups is None:
            expected_groups = ["Y", "M", "O"]
        
        assert config.group_labels == expected_groups
        assert config.group_size == len(expected_groups)
        assert config.config["simulation"]["engine"] in ["MMCACovid19", "MMCACovid19Vac"]
    
    @staticmethod
    def create_invalid_config_missing_section(base_config: Dict[str, Any], section: str) -> Dict[str, Any]:
        """Create invalid config by removing a section"""
        invalid_config = base_config.copy()
        del invalid_config[section]
        return invalid_config
    
    @staticmethod
    def create_invalid_config_missing_key(base_config: Dict[str, Any], section: str, key: str) -> Dict[str, Any]:
        """Create invalid config by removing a key from a section"""
        import copy
        invalid_config = copy.deepcopy(base_config)
        del invalid_config[section][key]
        return invalid_config
    
    @staticmethod
    def create_invalid_config_wrong_group_size(base_config: Dict[str, Any], param: str, wrong_size: int) -> Dict[str, Any]:
        """Create invalid config with wrong group size for a parameter"""
        import copy
        invalid_config = copy.deepcopy(base_config)
        
        # Find the parameter and modify it
        if param in invalid_config["epidemic_params"]:
            invalid_config["epidemic_params"][param] = [0.3] * wrong_size
        elif param in invalid_config["population_params"]:
            invalid_config["population_params"][param] = [0.3] * wrong_size
        
        return invalid_config
    
    @staticmethod
    def create_test_xarray_data(
        regions: List[str] = None,
        age_groups: List[str] = None,
        time_steps: int = 5,
        epi_states: List[str] = None
    ) -> xr.DataArray:
        """Create test xarray data for compute_observables testing"""
        if regions is None:
            regions = ["region_1", "region_2"]
        if age_groups is None:
            age_groups = ["Y", "M", "O"]
        if epi_states is None:
            epi_states = ["S", "E", "A", "I", "PH", "PD", "HR", "HD", "R", "D"]
        
        data = np.random.rand(len(age_groups), len(regions), time_steps, len(epi_states))
        coords = {
            "G": age_groups,
            "M": regions,
            "T": np.arange(time_steps),
            "epi_states": epi_states,
        }
        
        return xr.DataArray(data, coords=coords, dims=["G", "M", "T", "epi_states"])


class AssertionHelpers:
    """Collection of common assertion helpers"""
    
    @staticmethod
    def assert_file_exists(file_path: str, message: str = None):
        """Assert that a file exists"""
        if message is None:
            message = f"File should exist: {file_path}"
        assert os.path.exists(file_path), message
    
    @staticmethod
    def assert_file_not_exists(file_path: str, message: str = None):
        """Assert that a file does not exist"""
        if message is None:
            message = f"File should not exist: {file_path}"
        assert not os.path.exists(file_path), message
    
    @staticmethod
    def assert_config_saved_correctly(config_path: str, expected_values: Dict[str, Any]):
        """Assert that configuration was saved correctly"""
        AssertionHelpers.assert_file_exists(config_path)
        
        with open(config_path) as f:
            saved_config = json.load(f)
        
        for key, expected_value in expected_values.items():
            keys = key.split(".")
            current = saved_config
            for k in keys[:-1]:
                current = current[k]
            assert current[keys[-1]] == expected_value, f"Config key {key} should be {expected_value}"
    
    @staticmethod
    def assert_subprocess_called_correctly(
        mock_run, 
        expected_args: List[str], 
        expected_in_args: List[str] = None
    ):
        """Assert that subprocess was called with correct arguments"""
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        
        for expected_arg in expected_args:
            assert expected_arg in args, f"Expected argument {expected_arg} not found in {args}"
        
        if expected_in_args:
            for expected_in_arg in expected_in_args:
                assert any(expected_in_arg in arg for arg in args), f"Expected substring {expected_in_arg} not found in args"
    
    @staticmethod
    def assert_dataarray_structure(
        da: xr.DataArray,
        expected_dims: List[str],
        expected_coords: Dict[str, List[str]],
        expected_shape: Tuple[int, ...] = None
    ):
        """Assert xarray DataArray structure"""
        assert isinstance(da, xr.DataArray)
        assert list(da.dims) == expected_dims
        
        for coord_name, expected_values in expected_coords.items():
            assert list(da.coords[coord_name]) == expected_values
        
        if expected_shape:
            assert da.shape == expected_shape