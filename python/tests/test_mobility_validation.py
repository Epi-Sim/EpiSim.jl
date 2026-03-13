"""Tests for mobility validation functionality.

This module tests the validation logic for time-varying mobility NetCDF files,
ensuring row-stochastic matrices are properly validated.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# Add python directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def create_row_stochastic_matrix(M: int, rng: np.random.Generator) -> np.ndarray:
    """Create a row-stochastic mobility matrix (each row sums to 1.0)."""
    mobility = np.zeros((M, M))
    for i in range(M):
        self_loop = 0.8 + rng.random() * 0.1
        mobility[i, i] = self_loop
        remaining = 1.0 - self_loop
        if M > 1:
            other_indices = [j for j in range(M) if j != i]
            cross_flows = rng.dirichlet(np.ones(M - 1)) * remaining
            mobility[i, other_indices] = cross_flows
    return mobility


def create_non_stochastic_matrix(M: int, rng: np.random.Generator) -> np.ndarray:
    """Create a NON row-stochastic mobility matrix (for invalid test)."""
    mobility = np.zeros((M, M))
    for i in range(M):
        self_loop = 0.5 + rng.random() * 0.3
        mobility[i, i] = self_loop
        remaining = 0.8 - self_loop
        if M > 1:
            other_indices = [j for j in range(M) if j != i]
            cross_flows = rng.random(M - 1) * remaining
            mobility[i, other_indices] = cross_flows
    return mobility


def create_valid_mobility_netcdf(output_path: Path, M: int = 3, T: int = 10) -> None:
    """Create a VALID mobility NetCDF file with proper row-stochastic matrices."""
    rng = np.random.default_rng(42)
    dates = [f"2020-03-{i + 1:02d}" for i in range(T)]
    region_ids = [f"region_{i:02d}" for i in range(M)]

    mobility_data = np.zeros((T, M, M), dtype=np.float64)
    for t in range(T):
        mobility_data[t] = create_row_stochastic_matrix(M, rng)

    data_vars = {
        "mobility": (
            ["date", "origin", "destination"],
            mobility_data,
            {
                "long_name": "Mobility flow probability",
                "units": "probability",
                "description": "Row-stochastic: each origin row sums to 1.0",
            },
        ),
    }

    coords = {
        "date": (["date"], dates),
        "origin": (["origin"], np.array(region_ids)),
        "destination": (["destination"], np.array(region_ids)),
    }

    dataset = xr.Dataset(data_vars, coords)
    dataset.attrs["title"] = "VALID Time-Varying Mobility Data"
    dataset.attrs["test_type"] = "VALID - row-stochastic matrices"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)


def create_invalid_mobility_netcdf(output_path: Path, M: int = 3, T: int = 10) -> None:
    """Create an INVALID mobility NetCDF file with non-row-stochastic matrices."""
    rng = np.random.default_rng(43)
    dates = [f"2020-03-{i + 1:02d}" for i in range(T)]
    region_ids = [f"region_{i:02d}" for i in range(M)]

    mobility_data = np.zeros((T, M, M), dtype=np.float64)
    for t in range(T):
        mobility_data[t] = create_non_stochastic_matrix(M, rng)

    data_vars = {
        "mobility": (
            ["date", "origin", "destination"],
            mobility_data,
            {
                "long_name": "Mobility flow probability",
                "units": "probability",
                "description": "INVALID: rows do NOT sum to 1.0",
            },
        ),
    }

    coords = {
        "date": (["date"], dates),
        "origin": (["origin"], np.array(region_ids)),
        "destination": (["destination"], np.array(region_ids)),
    }

    dataset = xr.Dataset(data_vars, coords)
    dataset.attrs["title"] = "INVALID Time-Varying Mobility Data"
    dataset.attrs["test_type"] = "INVALID - non-row-stochastic matrices"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)


def validate_mobility_netcdf(
    netcdf_path: Path, tolerance: float = 1e-4
) -> tuple[bool, list[str]]:
    """Validate a mobility NetCDF file."""
    errors = []

    if not netcdf_path.exists():
        errors.append(f"File does not exist: {netcdf_path}")
        return False, errors

    try:
        ds = xr.open_dataset(netcdf_path)
    except (FileNotFoundError, PermissionError, OSError, ValueError) as e:
        errors.append(f"Failed to load NetCDF file: {e}")
        return False, errors

    if "mobility" not in ds:
        errors.append("Missing required variable: 'mobility'")
        ds.close()
        return False, errors

    mobility = ds["mobility"].values
    ds.close()

    if mobility.ndim != 3:
        errors.append(
            f"Mobility must be 3D (date, origin, destination), got shape {mobility.shape}"
        )
        return False, errors

    T, M, _ = mobility.shape

    if np.any(mobility < 0):
        negative_count = np.sum(mobility < 0)
        errors.append(f"Found {negative_count} negative values in mobility data")

    invalid_rows = []
    for t in range(T):
        row_sums = mobility[t].sum(axis=1)
        for i, row_sum in enumerate(row_sums):
            if abs(row_sum - 1.0) > tolerance:
                invalid_rows.append((t, i, row_sum))

    if invalid_rows:
        errors.append(f"Row-stochastic validation failed for {len(invalid_rows)} rows")

    is_valid = len(errors) == 0
    return is_valid, errors


class TestMobilityValidation:
    """Tests for mobility NetCDF validation."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for test files."""
        return tmp_path

    def test_valid_mobility_passes_validation(self, temp_dir):
        """Valid row-stochastic mobility should pass validation."""
        valid_path = temp_dir / "mobility_valid.nc"
        create_valid_mobility_netcdf(valid_path, M=3, T=5)

        is_valid, errors = validate_mobility_netcdf(valid_path)

        assert is_valid, f"Valid file should pass validation! Errors: {errors}"
        assert len(errors) == 0

    def test_invalid_mobility_fails_validation(self, temp_dir):
        """Non-row-stochastic mobility should fail validation."""
        invalid_path = temp_dir / "mobility_invalid.nc"
        create_invalid_mobility_netcdf(invalid_path, M=3, T=5)

        is_valid, errors = validate_mobility_netcdf(invalid_path)

        assert not is_valid, "Invalid file should fail validation"
        assert len(errors) > 0
        assert any("Row-stochastic" in e for e in errors)

    def test_missing_file_fails_validation(self, temp_dir):
        """Non-existent file should fail validation."""
        missing_path = temp_dir / "nonexistent.nc"

        is_valid, errors = validate_mobility_netcdf(missing_path)

        assert not is_valid
        assert any("does not exist" in e for e in errors)

    def test_missing_mobility_variable_fails(self, temp_dir):
        """NetCDF without 'mobility' variable should fail."""
        bad_path = temp_dir / "bad.nc"
        ds = xr.Dataset({"other_var": (["x"], [1, 2, 3])})
        ds.to_netcdf(bad_path)

        is_valid, errors = validate_mobility_netcdf(bad_path)

        assert not is_valid
        assert any("Missing required variable" in e for e in errors)

    def test_wrong_dimensions_fails(self, temp_dir):
        """2D mobility data should fail validation."""
        bad_path = temp_dir / "bad_dims.nc"
        ds = xr.Dataset({"mobility": (["x", "y"], np.random.rand(3, 3))})
        ds.to_netcdf(bad_path)

        is_valid, errors = validate_mobility_netcdf(bad_path)

        assert not is_valid
        assert any("must be 3D" in e for e in errors)

    def test_negative_values_fails(self, temp_dir):
        """Negative mobility values should fail validation."""
        bad_path = temp_dir / "negative.nc"
        rng = np.random.default_rng(42)
        M, T = 3, 5
        dates = [f"2020-03-{i + 1:02d}" for i in range(T)]
        region_ids = [f"region_{i:02d}" for i in range(M)]

        mobility_data = create_row_stochastic_matrix(M, rng)
        mobility_data[0, 0] = -0.1  # Add negative value
        mobility_data = np.stack([mobility_data] * T)

        ds = xr.Dataset(
            {"mobility": (["date", "origin", "destination"], mobility_data)},
            coords={
                "date": dates,
                "origin": region_ids,
                "destination": region_ids,
            },
        )
        ds.to_netcdf(bad_path)

        is_valid, errors = validate_mobility_netcdf(bad_path)

        assert not is_valid
        assert any("negative" in e.lower() for e in errors)

    def test_tolerance_parameter(self, temp_dir):
        """Tolerance parameter should affect validation."""
        valid_path = temp_dir / "mobility_valid.nc"
        create_valid_mobility_netcdf(valid_path, M=3, T=5)

        # Should pass with default tolerance
        is_valid, _ = validate_mobility_netcdf(valid_path, tolerance=1e-4)
        assert is_valid

        # Should still pass with stricter tolerance
        is_valid, _ = validate_mobility_netcdf(valid_path, tolerance=1e-10)
        assert is_valid

    def test_row_sums_close_to_one(self, temp_dir):
        """Rows summing to ~1.0 within tolerance should pass."""
        path = temp_dir / "almost_valid.nc"
        rng = np.random.default_rng(42)
        M, T = 3, 2
        dates = [f"2020-03-{i + 1:02d}" for i in range(T)]
        region_ids = [f"region_{i:02d}" for i in range(M)]

        # Create matrix with small deviation from 1.0
        mobility_data = create_row_stochastic_matrix(M, rng)
        mobility_data[0] *= 1.00001  # 0.001% deviation
        mobility_data = np.stack([mobility_data] * T)

        ds = xr.Dataset(
            {"mobility": (["date", "origin", "destination"], mobility_data)},
            coords={
                "date": dates,
                "origin": region_ids,
                "destination": region_ids,
            },
        )
        ds.to_netcdf(path)

        # Should pass with loose tolerance
        is_valid, _ = validate_mobility_netcdf(path, tolerance=1e-3)
        assert is_valid

        # Should fail with strict tolerance
        is_valid, _ = validate_mobility_netcdf(path, tolerance=1e-10)
        assert not is_valid


class TestMobilityMatrixCreation:
    """Tests for mobility matrix creation functions."""

    def test_row_stochastic_matrix_sums_to_one(self):
        """Row-stochastic matrix should have rows summing to 1.0."""
        rng = np.random.default_rng(42)
        M = 5
        matrix = create_row_stochastic_matrix(M, rng)

        row_sums = matrix.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(M))

    def test_row_stochastic_has_high_self_loops(self):
        """Row-stochastic matrix should have high self-loop values."""
        rng = np.random.default_rng(42)
        M = 5
        matrix = create_row_stochastic_matrix(M, rng)

        for i in range(M):
            assert matrix[i, i] >= 0.8
            assert matrix[i, i] <= 0.9

    def test_non_stochastic_matrix_does_not_sum_to_one(self):
        """Non-stochastic matrix should NOT have rows summing to 1.0."""
        rng = np.random.default_rng(43)
        M = 5
        matrix = create_non_stochastic_matrix(M, rng)

        row_sums = matrix.sum(axis=1)
        assert not np.allclose(row_sums, np.ones(M))

    def test_non_stochastic_has_lower_self_loops(self):
        """Non-stochastic matrix should have lower self-loop values."""
        rng = np.random.default_rng(43)
        M = 5
        matrix = create_non_stochastic_matrix(M, rng)

        for i in range(M):
            assert matrix[i, i] >= 0.5
            assert matrix[i, i] <= 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
