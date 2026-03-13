"""
Test mobility generator and validator
"""

import numpy as np
import pandas as pd

from episim_python.mobility import (
    MobilityGenerator,
    MobilityValidator,
    load_baseline_mobility,
)


def test_mobility_generator_basic():
    """Test basic mobility generation with a simple example."""
    # Create a simple 3x3 baseline matrix
    M = 3
    # Create sparse edgelist with self-loops and some cross-edges
    edgelist = np.array([
        [0, 0], [0, 1],
        [1, 1], [1, 2],
        [2, 0], [2, 2],
    ], dtype=np.int64)

    # Row-stochastic weights
    baseline_R = np.array([0.9, 0.1,  # Row 0: 90% stay, 10% go to 1
                          0.8, 0.2,  # Row 1: 80% stay, 20% go to 2
                          0.05, 0.95])  # Row 2: 5% go to 0, 95% stay

    # Create generator with some noise
    generator = MobilityGenerator(
        baseline_R=(edgelist, baseline_R),
        sigma_O=0.1,
        sigma_D=0.1,
        rng_seed=42
    )

    # Test single timestep generation
    R_t = generator.generate_R_t(t=0)

    # Verify row-stochasticity
    M_check = generator.M
    row_sums = np.zeros(M_check)
    for e, (i, _) in enumerate(edgelist):
        row_sums[i] += R_t[e]

    assert np.allclose(row_sums, 1.0, atol=1e-3), f"Row sums not stochastic: {row_sums}"
    print("✓ Single timestep generation: row-stochastic")

    # Test series generation
    T = 10
    R_series = generator.generate_series(T=T)

    assert R_series.shape == (T, len(baseline_R)), f"Unexpected shape: {R_series.shape}"

    # Verify all timesteps are row-stochastic
    for t in range(T):
        row_sums = np.zeros(M_check)
        for e, (i, _) in enumerate(edgelist):
            row_sums[i] += R_series[t, e]
        assert np.allclose(row_sums, 1.0, atol=1e-3), f"Row {t} not stochastic: {row_sums}"

    print("✓ Series generation: all timesteps row-stochastic")


def test_mobility_validator():
    """Test mobility validation."""
    M = 3
    edgelist = np.array([
        [0, 0], [0, 1],
        [1, 1], [1, 2],
        [2, 0], [2, 2],
    ], dtype=np.int64)

    # Valid row-stochastic matrices
    baseline_R = np.array([0.9, 0.1, 0.8, 0.2, 0.05, 0.95])
    R_series = np.tile(baseline_R, (10, 1))  # 10 identical timesteps

    validator = MobilityValidator()
    is_valid, errors = validator.validate_series(
        R_series, edgelist, M=M, T=10, verbose=True
    )

    assert is_valid, f"Validation failed unexpectedly: {errors}"
    print("✓ Validation passed for valid mobility series")

    # Test with invalid data (non-row-stochastic)
    R_invalid = R_series.copy()
    R_invalid[0, 0] = 2.0  # Make first row not stochastic

    is_valid, errors = validator.validate_series(
        R_invalid, edgelist, M=M, T=10, verbose=False
    )

    assert not is_valid, "Validation should fail for non-stochastic matrix"
    assert len(errors) > 0, "Should have error messages"
    print(f"✓ Validation correctly failed with {len(errors)} errors")


def test_dense_sparse_conversion():
    """Test conversion between dense and sparse formats."""
    M = 3
    edgelist = np.array([
        [0, 0], [0, 1],
        [1, 1], [1, 2],
        [2, 0], [2, 2],
    ], dtype=np.int64)

    baseline_R = np.array([0.9, 0.1, 0.8, 0.2, 0.05, 0.95])

    generator = MobilityGenerator(
        baseline_R=(edgelist, baseline_R),
        sigma_O=0.0,
        sigma_D=0.0,
    )

    # Convert to dense
    R_dense = generator.to_dense(baseline_R)

    assert R_dense.shape == (M, M), f"Unexpected dense shape: {R_dense.shape}"

    # Check some values
    assert R_dense[0, 0] == 0.9, f"R_dense[0,0] = {R_dense[0,0]}, expected 0.9"
    assert R_dense[0, 1] == 0.1, f"R_dense[0,1] = {R_dense[0,1]}, expected 0.1"
    assert R_dense[1, 0] == 0.0, "R_dense[1,0] should be 0"

    print("✓ Dense/sparse conversion working correctly")


def test_load_baseline_mobility():
    """Test loading baseline mobility from CSV."""
    # Create a temporary CSV file
    import os
    import tempfile

    # Create sparse format CSV
    sparse_data = pd.DataFrame({
        "origin": [0, 0, 1, 1, 2, 2],
        "destination": [0, 1, 1, 2, 0, 2],
        "value": [90, 10, 80, 20, 5, 95]
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sparse_path = f.name
        sparse_data.to_csv(f, index=False)

    try:
        edgelist, R_baseline, M = load_baseline_mobility(sparse_path)

        assert edgelist.shape == (6, 2), f"Unexpected edgelist shape: {edgelist.shape}"
        assert len(R_baseline) == 6, f"Unexpected R_baseline length: {len(R_baseline)}"
        assert M == 3, f"Unexpected M: {M}"

        # Verify row-stochasticity after normalization
        row_sums = np.zeros(M)
        for e, (i, j) in enumerate(edgelist):
            row_sums[i] += R_baseline[e]

        assert np.allclose(row_sums, 1.0, atol=1e-3), f"Row sums not stochastic: {row_sums}"

        print("✓ CSV loading working correctly")

    finally:
        os.unlink(sparse_path)


if __name__ == "__main__":
    print("Testing Mobility Generator and Validator\n")
    print("=" * 50)

    test_mobility_generator_basic()
    test_mobility_validator()
    test_dense_sparse_conversion()
    test_load_baseline_mobility()

    print("\n" + "=" * 50)
    print("All tests passed!")
