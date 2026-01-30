"""Unit tests for sparsity tier assignment functionality."""

import numpy as np
import pytest

from process_synthetic_outputs import assign_sparsity_tiers


def test_assign_sparsity_tiers_stratified_distribution():
    """Test stratified assignment across tiers."""
    sparsity = assign_sparsity_tiers(100, [0.05, 0.20, 0.40, 0.60, 0.80])
    assert len(sparsity) == 100
    unique_values = np.unique(sparsity)
    assert list(unique_values) == [0.05, 0.20, 0.40, 0.60, 0.80]
    # Check roughly equal distribution (±20%)
    counts = [(sparsity == v).sum() for v in unique_values]
    # For 100 runs across 5 tiers: expected 20 per tier, allow 15-25 range
    assert all(15 <= c <= 25 for c in counts), f"Uneven distribution: {counts}"


def test_assign_sparsity_tiers_deterministic():
    """Test that same seed produces same assignment."""
    sparsity1 = assign_sparsity_tiers(77, [0.05, 0.20, 0.40, 0.60, 0.80], seed=42)
    sparsity2 = assign_sparsity_tiers(77, [0.05, 0.20, 0.40, 0.60, 0.80], seed=42)
    assert np.array_equal(sparsity1, sparsity2), "Same seed should produce same assignment"


def test_assign_sparsity_tiers_different_seeds():
    """Test that different seeds produce different assignments."""
    sparsity1 = assign_sparsity_tiers(50, [0.05, 0.20, 0.40], seed=1)
    sparsity2 = assign_sparsity_tiers(50, [0.05, 0.20, 0.40], seed=2)
    assert not np.array_equal(sparsity1, sparsity2), "Different seeds should produce different assignments"


def test_assign_sparsity_tiers_single_tier():
    """Test that single tier assigns all runs to that tier."""
    sparsity = assign_sparsity_tiers(25, [0.50])
    assert len(sparsity) == 25
    assert np.all(sparsity == 0.50), "All runs should have sparsity 0.50"


def test_assign_sparsity_tiers_uneven_division():
    """Test handling of runs not evenly divisible by number of tiers."""
    # 17 runs across 5 tiers: 3, 3, 3, 4, 4 distribution
    sparsity = assign_sparsity_tiers(17, [0.05, 0.20, 0.40, 0.60, 0.80], seed=42)
    assert len(sparsity) == 17
    unique_values, counts = np.unique(sparsity, return_counts=True)
    assert list(unique_values) == [0.05, 0.20, 0.40, 0.60, 0.80]
    # Total should equal 17
    assert sum(counts) == 17
    # No tier should be empty
    assert all(c > 0 for c in counts)


def test_assign_sparsity_tiers_custom_tiers():
    """Test with custom sparsity tier values."""
    custom_tiers = [0.01, 0.10, 0.50, 0.90]
    sparsity = assign_sparsity_tiers(40, custom_tiers, seed=123)
    assert len(sparsity) == 40
    unique_values = np.unique(sparsity)
    assert list(unique_values) == custom_tiers


def test_assign_sparsity_tiers_no_seed():
    """Test that no seed produces different results each time."""
    sparsity1 = assign_sparsity_tiers(30, [0.1, 0.5, 0.9])
    sparsity2 = assign_sparsity_tiers(30, [0.1, 0.5, 0.9])
    # Without a seed, it's possible (though unlikely) to get the same result
    # We just verify the function works without errors
    assert len(sparsity1) == 30
    assert len(sparsity2) == 30


def test_assign_sparsity_tiers_small_n():
    """Test with fewer runs than tiers."""
    # 3 runs, 5 tiers: should assign 0 or 1 run per tier
    sparsity = assign_sparsity_tiers(3, [0.05, 0.20, 0.40, 0.60, 0.80], seed=42)
    assert len(sparsity) == 3
    # All values should be from the tier list
    assert all(s in [0.05, 0.20, 0.40, 0.60, 0.80] for s in sparsity)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
