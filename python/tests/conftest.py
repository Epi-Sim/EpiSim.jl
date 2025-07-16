"""
pytest configuration and fixtures for episim_python tests
"""

import shutil
import tempfile
from pathlib import Path

import pytest

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config_json():
    """Path to test configuration JSON file"""
    return TEST_DATA_DIR / "test_config.json"


@pytest.fixture
def test_metapopulation_csv():
    """Path to test metapopulation CSV file"""
    return TEST_DATA_DIR / "test_metapopulation.csv"


@pytest.fixture
def test_rosetta_csv():
    """Path to test rosetta CSV file"""
    return TEST_DATA_DIR / "test_rosetta.csv"


@pytest.fixture
def minimal_config():
    """Minimal valid configuration dictionary"""
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
            "ηᵍ": [0.3, 0.3, 0.3],
            "αᵍ": [0.25, 0.6, 0.6],
            "μᵍ": [1.0, 0.3, 0.3],
            "θᵍ": [0.0, 0.0, 0.0],
            "γᵍ": [0.003, 0.01, 0.08],
            "ζᵍ": [0.13, 0.13, 0.13],
            "λᵍ": [1.0, 1.0, 1.0],
            "ωᵍ": [0.0, 0.04, 0.3],
            "ψᵍ": [0.14, 0.14, 0.14],
            "χᵍ": [0.05, 0.05, 0.05],
        },
        "population_params": {
            "G_labels": ["Y", "M", "O"],
            "C": [[0.6, 0.4, 0.02], [0.25, 0.7, 0.04], [0.2, 0.55, 0.25]],
            "kᵍ": [12.0, 13.0, 7.0],
            "kᵍ_h": [3.0, 3.0, 3.0],
            "kᵍ_w": [2.0, 5.0, 0.0],
            "pᵍ": [0.0, 1.0, 0.0],
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


@pytest.fixture
def sample_metapopulation_data():
    """Sample metapopulation data as dict"""
    return {
        "id": ["region_1", "region_2", "region_3"],
        "area": [100.0, 85.0, 120.0],
        "Y": [5000, 4500, 6000],
        "M": [8000, 7500, 9000],
        "O": [3000, 2800, 3500],
        "total": [16000, 14800, 18500],
    }


@pytest.fixture
def sample_rosetta_data():
    """Sample rosetta mapping data as dict"""
    return {
        "level_1": ["region_1", "region_2", "region_3"],
        "province": ["prov_A", "prov_B", "prov_A"],
        "region": ["reg_X", "reg_Y", "reg_X"],
    }
