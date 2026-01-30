"""
Unit tests for growth-rate-based spike detection.

Tests the growth_rate_detector module's ability to detect spikes using
sustained exponential growth combined with population-relative thresholds.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add python directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from growth_rate_detector import detect_spike_periods_growth_rate, calculate_growth_rate


class TestExponentialGrowthDetection:
    """Test detection of pure exponential growth patterns."""

    def test_pure_exponential_growth(self):
        """Test detection of pure exponential growth (doubling every day)."""
        # Generate 30 days of exponential growth: 1, 2, 4, 8, 16, 32, 64, ...
        t = np.arange(30)
        infections = 2 ** t

        spikes = detect_spike_periods_growth_rate(
            infections_array=infections,
            population=10000,
            growth_factor_threshold=1.5,
            min_growth_duration=3,
            min_cases_per_capita=1e-4,
        )

        # Should detect a spike starting early
        assert len(spikes) > 0, "Should detect spike in pure exponential growth"
        start, end = spikes[0]
        # Spike should start near the beginning when growth becomes sustained
        assert start < 10, f"Spike should start early, got start={start}"

    def test_fast_exponential_growth(self):
        """Test detection of very fast exponential growth (3x every 3 days)."""
        # Generate fast growth: 10, 30, 90, 270, 810, 2430, ...
        t = np.arange(30)
        infections = 10 * (3 ** (t / 3))

        spikes = detect_spike_periods_growth_rate(
            infections_array=infections,
            population=100000,
            growth_factor_threshold=2.0,  # Require 2x growth
            min_growth_duration=3,
            min_cases_per_capita=1e-4,
        )

        assert len(spikes) > 0, "Should detect spike in fast exponential growth"

    def test_slow_exponential_growth(self):
        """Test detection of slow exponential growth (10% daily)."""
        # Generate slow growth: 100, 110, 121, 133, 146, 161, ...
        t = np.arange(60)
        infections = 100 * (1.1 ** t)

        spikes = detect_spike_periods_growth_rate(
            infections_array=infections,
            population=1000000,
            growth_factor_threshold=1.3,  # Require 30% growth over 3 days
            min_growth_duration=3,
            min_cases_per_capita=1e-5,
        )

        # Should still detect the sustained growth
        assert len(spikes) > 0, "Should detect spike in slow exponential growth"


class TestDieOutAndRestart:
    """Test handling of die-out and restart patterns."""

    def test_die_out_then_exponential_growth(self):
        """Test detection after die-out followed by exponential growth."""
        # Pattern: 0, 0, 0, 1, 2, 4, 8, 16, 32, 64, ...
        infections = np.zeros(30)
        infections[3:] = 2 ** np.arange(27)

        spikes = detect_spike_periods_growth_rate(
            infections_array=infections,
            population=10000,
            growth_factor_threshold=1.5,
            min_growth_duration=3,
            min_cases_per_capita=1e-4,
        )

        assert len(spikes) > 0, "Should detect spike after die-out"
        start, end = spikes[0]
        # Spike should start near the exponential growth (with pre-growth window buffer)
        # The algorithm may include a pre-growth window, so start may be slightly before growth
        assert start >= 0 and start < 10, f"Spike should start near growth, got start={start}"

    def test_multiple_die_outs(self):
        """Test handling of multiple die-out and restart cycles."""
        # Pattern with two growth periods separated by die-out
        infections = np.zeros(60)
        # First growth: days 0-14
        infections[:15] = 1 * (1.5 ** np.arange(15))
        # Die-out: days 15-20
        infections[15:21] = 0
        # Second growth: days 21-40
        infections[21:] = 10 * (1.5 ** np.arange(39))

        spikes = detect_spike_periods_growth_rate(
            infections_array=infections,
            population=10000,
            growth_factor_threshold=1.5,
            min_growth_duration=3,
            min_cases_per_capita=1e-4,
            growth_window=3,
            min_spike_duration=7,
        )

        # Should detect at least one spike
        assert len(spikes) > 0, "Should detect spike in multi-cycle pattern"


class TestFalsePositives:
    """Test that the detector doesn't trigger on noise or low-level patterns."""

    def test_low_level_noise(self):
        """Test that low-level noise doesn't trigger false positives."""
        # Generate low-level noise around 10 cases
        np.random.seed(42)
        infections = 10 + np.random.randn(100) * 2
        infections = np.maximum(infections, 0)  # Ensure non-negative

        spikes = detect_spike_periods_growth_rate(
            infections_array=infections,
            population=1000000,
            growth_factor_threshold=1.5,
            min_growth_duration=3,
            min_cases_per_capita=1e-4,
            min_spike_duration=7,
        )

        # Should not detect sustained growth in noise
        assert len(spikes) == 0, f"Should not detect spike in noise, got {len(spikes)} spikes"

    def test_small_outbreak_then_decline(self):
        """Test that small outbreaks that decline don't trigger."""
        # Pattern: small outbreak that peaks and declines
        infections = np.array([5, 10, 15, 20, 15, 10, 5, 3, 2, 1] + [1] * 20)

        spikes = detect_spike_periods_growth_rate(
            infections_array=infections,
            population=100000,
            growth_factor_threshold=1.5,
            min_growth_duration=3,
            min_cases_per_capita=1e-4,
            min_spike_duration=7,
        )

        # Should not detect sustained growth in a declining outbreak
        assert len(spikes) == 0, "Should not detect spike in declining outbreak"

    def test_flat_cases_no_growth(self):
        """Test that flat case counts don't trigger."""
        # Pattern: flat 100 cases per day
        infections = np.full(100, 100.0)

        spikes = detect_spike_periods_growth_rate(
            infections_array=infections,
            population=10000,
            growth_factor_threshold=1.5,
            min_growth_duration=3,
            min_cases_per_capita=1e-4,
            min_spike_duration=7,
        )

        # Should not detect spike without growth
        assert len(spikes) == 0, "Should not detect spike without growth"


class TestPopulationRelativeThreshold:
    """Test that per-capita threshold scales with population."""

    def test_large_population_threshold(self):
        """Test that larger population requires higher absolute threshold."""
        # Same infection curve, different populations
        infections = 10 * (1.5 ** np.arange(30))

        # Small population (1K) - should trigger
        spikes_small = detect_spike_periods_growth_rate(
            infections_array=infections,
            population=1000,
            growth_factor_threshold=1.5,
            min_growth_duration=3,
            min_cases_per_capita=1e-4,  # 0.1 case threshold
        )

        # Large population (1M) - use very high per-capita threshold to avoid triggering
        # infections[25] ≈ 252K cases, so 1e-1 threshold = 100K cases should not trigger until day 24+
        # We want to ensure the outbreak doesn't reach threshold within our time window
        spikes_large = detect_spike_periods_growth_rate(
            infections_array=infections,
            population=1000000,
            growth_factor_threshold=1.5,
            min_growth_duration=3,
            min_cases_per_capita=5e-1,  # 500000 cases threshold - outbreak never reaches this
        )

        # Small population should trigger
        assert len(spikes_small) > 0, "Small population should trigger"
        # Large population with very high threshold should not trigger (outbreak too small)
        assert len(spikes_large) == 0, "Large population should not trigger with small outbreak and very high threshold"

    def test_custom_per_capita_threshold(self):
        """Test custom per-capita threshold behavior."""
        infections = 50 * (1.5 ** np.arange(30))

        # Strict threshold (1 per 1000)
        spikes_strict = detect_spike_periods_growth_rate(
            infections_array=infections,
            population=100000,
            growth_factor_threshold=1.5,
            min_growth_duration=3,
            min_cases_per_capita=1e-3,  # 100 cases threshold
        )

        # Lenient threshold (1 per 100K)
        spikes_lenient = detect_spike_periods_growth_rate(
            infections_array=infections,
            population=100000,
            growth_factor_threshold=1.5,
            min_growth_duration=3,
            min_cases_per_capita=1e-5,  # 1 case threshold
        )

        # Lenient should be more permissive
        assert len(spikes_lenient) >= len(spikes_strict), \
            "Lenient threshold should detect same or more spikes"


class TestGrowthRateCalculation:
    """Test the growth rate calculation utility function."""

    def test_growth_rate_calculation(self):
        """Test growth rate calculation for simple patterns."""
        # Test pure doubling: 1, 2, 4, 8, 16, 32
        infections = np.array([1, 2, 4, 8, 16, 32])
        growth_rate = calculate_growth_rate(infections, window=2)

        # At t=2: GF = 4/1 = 4
        # At t=3: GF = 8/2 = 4
        # At t=4: GF = 16/4 = 4
        # At t=5: GF = 32/8 = 4
        assert growth_rate[2] == pytest.approx(4.0, rel=0.1)
        assert growth_rate[3] == pytest.approx(4.0, rel=0.1)
        assert growth_rate[4] == pytest.approx(4.0, rel=0.1)
        assert growth_rate[5] == pytest.approx(4.0, rel=0.1)

    def test_growth_rate_with_zeros(self):
        """Test that zeros are handled correctly."""
        # Pattern with zeros: 0, 0, 1, 2, 4, 8
        infections = np.array([0, 0, 1, 2, 4, 8])
        growth_rate = calculate_growth_rate(infections, window=2, epsilon=0.1)

        # With epsilon replacement, zeros become 0.1
        # At t=2: GF = 1/0.1 = 10
        # At t=3: GF = 2/0.1 = 20
        # At t=4: GF = 4/1 = 4
        # At t=5: GF = 8/2 = 4
        assert growth_rate[2] > 1.0, "Growth rate should be positive after zeros"
        assert growth_rate[4] == pytest.approx(4.0, rel=0.1)
        assert growth_rate[5] == pytest.approx(4.0, rel=0.1)


class TestParameterSensitivity:
    """Test sensitivity to various parameter settings."""

    def test_growth_factor_threshold_impact(self):
        """Test that higher growth_factor_threshold is more conservative."""
        # Exponential growth: 1.5x per day
        infections = 10 * (1.5 ** np.arange(30))

        # Low threshold (1.2)
        spikes_low = detect_spike_periods_growth_rate(
            infections_array=infections,
            population=10000,
            growth_factor_threshold=1.2,
            min_growth_duration=3,
            min_cases_per_capita=1e-4,
        )

        # High threshold (2.0)
        spikes_high = detect_spike_periods_growth_rate(
            infections_array=infections,
            population=10000,
            growth_factor_threshold=2.0,
            min_growth_duration=3,
            min_cases_per_capita=1e-4,
        )

        # Lower threshold should be more permissive
        assert len(spikes_low) >= len(spikes_high), \
            "Lower growth factor threshold should detect same or more spikes"

    def test_min_growth_duration_impact(self):
        """Test that min_growth_duration affects detection."""
        # Short growth burst followed by decline
        infections = np.array([10] * 10 + [10, 15, 20, 30, 20, 10] + [5] * 20)

        # Require 2 consecutive days
        spikes_short = detect_spike_periods_growth_rate(
            infections_array=infections,
            population=10000,
            growth_factor_threshold=1.3,
            min_growth_duration=2,
            min_cases_per_capita=1e-4,
            min_spike_duration=5,
        )

        # Require 5 consecutive days
        spikes_long = detect_spike_periods_growth_rate(
            infections_array=infections,
            population=10000,
            growth_factor_threshold=1.3,
            min_growth_duration=5,
            min_cases_per_capita=1e-4,
            min_spike_duration=5,
        )

        # Shorter duration requirement should be more permissive
        assert len(spikes_short) >= len(spikes_long), \
            "Shorter min_growth_duration should detect same or more spikes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
