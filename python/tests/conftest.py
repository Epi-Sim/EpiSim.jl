"""
pytest configuration and fixtures for episim_python tests
"""

import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

import pytest

from .test_helpers import TestHelpers, MockResult, AssertionHelpers

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "fixtures"


class BaseTestCase:
    """Base test case class with common setup methods"""
    
    def setup_method(self):
        """Setup method called before each test method"""
        self.helpers = TestHelpers()
        self.assertions = AssertionHelpers()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_helpers():
    """Test helpers instance"""
    return TestHelpers()


@pytest.fixture
def assertion_helpers():
    """Assertion helpers instance"""
    return AssertionHelpers()


# Configuration fixtures
@pytest.fixture
def test_config_json():
    """Path to test configuration JSON file"""
    return TEST_DATA_DIR / "test_config.json"


@pytest.fixture
def minimal_config():
    """Minimal valid configuration dictionary (3 age groups)"""
    return TestHelpers.create_minimal_config(group_size=3)


@pytest.fixture
def two_group_config():
    """Configuration with 2 age groups"""
    return TestHelpers.create_minimal_config(group_size=2)


@pytest.fixture
def four_group_config():
    """Configuration with 4 age groups"""
    return TestHelpers.create_minimal_config(group_size=4)


@pytest.fixture(params=[2, 3, 4])
def parametrized_config(request):
    """Parametrized configuration with different group sizes"""
    return TestHelpers.create_minimal_config(group_size=request.param)


@pytest.fixture
def integration_config(minimal_config):
    """Configuration suitable for integration testing"""
    config = minimal_config.copy()
    # Use very short simulation period for faster tests
    config["simulation"]["start_date"] = "2020-01-01"
    config["simulation"]["end_date"] = "2020-01-03"  # Just 3 days
    return config


# Test data fixtures
@pytest.fixture
def test_metapopulation_csv(temp_dir):
    """Create test metapopulation CSV file"""
    csv_path = Path(temp_dir) / "test_metapopulation.csv"
    content = TestHelpers.create_test_csv_content()
    csv_path.write_text(content)
    return csv_path


@pytest.fixture
def test_rosetta_csv(temp_dir):
    """Create test rosetta CSV file"""
    csv_path = Path(temp_dir) / "test_rosetta.csv"
    content = TestHelpers.create_test_rosetta_content()
    csv_path.write_text(content)
    return csv_path


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


# Mock fixtures
@pytest.fixture
def mock_success_result():
    """Mock successful subprocess result"""
    return TestHelpers.create_mock_success_result()


@pytest.fixture
def mock_failure_result():
    """Mock failed subprocess result"""
    return TestHelpers.create_mock_failure_result()


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run with success result"""
    mock_result = TestHelpers.create_mock_success_result()
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        yield mock_run


@pytest.fixture
def mock_subprocess_run_failure():
    """Mock subprocess.run with failure result"""
    mock_result = TestHelpers.create_mock_failure_result()
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        yield mock_run


@pytest.fixture
def mock_julia_available():
    """Mock Julia interpreter availability"""
    with patch("shutil.which", return_value="/usr/bin/julia"):
        yield


@pytest.fixture
def mock_julia_unavailable():
    """Mock Julia interpreter unavailability"""
    with patch("shutil.which", return_value=None):
        yield


# EpiSim model fixtures
@pytest.fixture
def basic_episim_model(minimal_config, temp_dir):
    """Basic EpiSim model instance"""
    return TestHelpers.setup_episim_model(minimal_config, temp_dir)


@pytest.fixture
def episim_model_with_interpreter(minimal_config, temp_dir):
    """EpiSim model set up with interpreter mode"""
    return TestHelpers.setup_episim_with_interpreter(minimal_config, temp_dir)


@pytest.fixture
def episim_model_with_compiled(minimal_config, temp_dir):
    """EpiSim model set up with compiled mode"""
    return TestHelpers.setup_episim_with_compiled(minimal_config, temp_dir)


# Invalid configuration fixtures
@pytest.fixture
def invalid_config_missing_simulation(minimal_config):
    """Invalid config with missing simulation section"""
    return TestHelpers.create_invalid_config_missing_section(minimal_config, "simulation")


@pytest.fixture
def invalid_config_missing_engine(minimal_config):
    """Invalid config with missing engine key"""
    return TestHelpers.create_invalid_config_missing_key(minimal_config, "simulation", "engine")


@pytest.fixture
def invalid_config_wrong_group_size(minimal_config):
    """Invalid config with wrong group size"""
    return TestHelpers.create_invalid_config_wrong_group_size(minimal_config, "ηᵍ", 2)


# XArray test data
@pytest.fixture
def test_xarray_data():
    """Test xarray data for compute_observables testing"""
    return TestHelpers.create_test_xarray_data()


# Utility fixtures
@pytest.fixture
def instance_folder(temp_dir):
    """Create instance folder"""
    folder = Path(temp_dir) / "instances"
    folder.mkdir(exist_ok=True)
    return str(folder)


@pytest.fixture
def dummy_initial_conditions(temp_dir):
    """Create dummy initial conditions file"""
    ic_path = Path(temp_dir) / "initial.nc"
    ic_path.write_text("dummy nc content")
    return str(ic_path)


@pytest.fixture
def dummy_executable(temp_dir):
    """Create dummy executable file"""
    executable_path = Path(temp_dir) / "episim"
    executable_path.write_text("#!/bin/bash\necho 'compiled episim'")
    executable_path.chmod(0o755)
    return str(executable_path)
