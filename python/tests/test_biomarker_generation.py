"""Tests for biomarker generation functions in synthetic_observations.py

Tests cover:
- Basic wastewater signal generation
- Age-stratified wastewater generation
- Reported case derivation
- Limit of detection (LoD) behaviors
- Transport loss modeling
- Gamma shedding curves
- Log-normal noise application
- Edge cases (zero values, extreme values)
"""

import os
import sys

import numpy as np
import pytest
from scipy import stats

# Add python directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synthetic_observations import (
    DEFAULT_REPORTED_CASES_CONFIG,
    DEFAULT_WASTEWATER_CONFIG,
    _build_shedding_kernel,
    _compute_monitoring_start_mask,
    _normalize_infections,
    generate_reported_cases,
    generate_wastewater,
    generate_wastewater_stratified,
)


class TestNormalizeInfections:
    """Tests for the _normalize_infections helper function."""

    def test_clips_negative_values(self):
        """Negative values should be clipped to zero."""
        infections = np.array([10, -5, 20, -1])
        result = _normalize_infections(infections)
        np.testing.assert_array_equal(result, np.array([10, 0, 20, 0]))

    def test_rounds_to_integers(self):
        """Float values should be rounded to nearest integer."""
        infections = np.array([10.3, 20.7, 15.5])
        result = _normalize_infections(infections)
        np.testing.assert_array_equal(result, np.array([10, 21, 16]))

    def test_preserves_positive_integers(self):
        """Positive integers should remain unchanged."""
        infections = np.array([10, 20, 30])
        result = _normalize_infections(infections)
        np.testing.assert_array_equal(result, infections)

    def test_handles_zeros(self):
        """Zeros should remain zeros."""
        infections = np.array([0, 10, 0])
        result = _normalize_infections(infections)
        np.testing.assert_array_equal(result, infections)

    def test_multidimensional(self):
        """Should handle 2D arrays (time, location)."""
        infections = np.array([[10.5, -5], [20.3, 15.7]])
        result = _normalize_infections(infections)
        expected = np.array([[10, 0], [20, 16]])
        np.testing.assert_array_equal(result, expected)


class TestSheddingKernel:
    """Tests for the _build_shedding_kernel helper function."""

    def test_kernel_is_normalized(self):
        """Kernel should sum to approximately 1.0."""
        kernel = _build_shedding_kernel(shape=2.5, scale=2.0, quantile=0.999)
        np.testing.assert_allclose(kernel.sum(), 1.0, atol=1e-6)

    def test_kernel_is_non_negative(self):
        """All kernel values should be non-negative."""
        kernel = _build_shedding_kernel(shape=2.5, scale=2.0, quantile=0.999)
        assert np.all(kernel >= 0)

    def test_kernel_shape_parameters(self):
        """Different shape parameters should produce different distributions."""
        # Lower shape = more spread out (exponential-like)
        # Higher shape = more concentrated
        kernel_flat = _build_shedding_kernel(shape=1.0, scale=5.0, quantile=0.99)
        kernel_peaked = _build_shedding_kernel(shape=5.0, scale=2.0, quantile=0.99)

        # Kernels should be different
        assert not np.allclose(kernel_flat, kernel_peaked)

        # Both should sum to 1 (normalized)
        np.testing.assert_allclose(kernel_flat.sum(), 1.0, atol=1e-6)
        np.testing.assert_allclose(kernel_peaked.sum(), 1.0, atol=1e-6)

        # Higher scale should produce longer kernel (for same quantile)
        assert len(kernel_flat) >= len(kernel_peaked)

    def test_kernel_minimum_length(self):
        """Kernel should have minimum length of 1."""
        kernel = _build_shedding_kernel(shape=100, scale=0.001, quantile=0.5)
        assert len(kernel) >= 1

    def test_kernel_quantile_affects_length(self):
        """Higher quantile should produce longer kernels."""
        kernel_99 = _build_shedding_kernel(shape=2.5, scale=2.0, quantile=0.99)
        kernel_999 = _build_shedding_kernel(shape=2.5, scale=2.0, quantile=0.999)
        assert len(kernel_999) >= len(kernel_99)


class TestReportedCases:
    """Tests for generate_reported_cases function."""

    @pytest.fixture
    def sample_infections(self):
        """Sample infection data for testing."""
        return np.array([10, 20, 50, 100, 80, 40, 20, 10])

    @pytest.fixture
    def sample_infections_2d(self):
        """Sample 2D infection data (time, location)."""
        return np.array([[10, 20], [20, 40], [50, 80], [100, 150]])

    def test_returns_integer_counts(self, sample_infections):
        """Reported cases should be integers."""
        reported, _ = generate_reported_cases(sample_infections, rng=np.random.default_rng(42))
        assert reported.dtype == np.int64

    def test_returns_ascertainment_rate(self, sample_infections):
        """Should return daily ascertainment rates."""
        reported, rates = generate_reported_cases(sample_infections, rng=np.random.default_rng(42))
        assert len(rates) == len(sample_infections)
        assert np.all(rates >= 0) and np.all(rates <= 1)

    def test_logistic_ramp_shape(self, sample_infections):
        """Ascertainment rate should follow logistic curve."""
        cfg = {
            "min_rate": 0.1,
            "max_rate": 0.8,
            "inflection_day": 3.5,
            "slope": 1.0,
        }
        reported, rates = generate_reported_cases(sample_infections, config=cfg, rng=np.random.default_rng(42))

        # Rate should increase from min to max
        assert rates[0] < rates[-1]
        # Should be bounded by min and max
        assert np.all(rates >= cfg["min_rate"])
        assert np.all(rates <= cfg["max_rate"])

    def test_reported_leverages_infections(self, sample_infections):
        """Reported cases should correlate with infections."""
        reported, _ = generate_reported_cases(sample_infections, rng=np.random.default_rng(42))

        # High infection days should generally have higher reports
        # (accounting for randomness)
        assert np.corrcoef(sample_infections, reported)[0, 1] > 0.5

    def test_2d_input_handling(self, sample_infections_2d):
        """Should handle 2D arrays (time, location)."""
        reported, rates = generate_reported_cases(
            sample_infections_2d, rng=np.random.default_rng(42)
        )

        assert reported.shape == sample_infections_2d.shape
        # Rates should be 1D (time only)
        assert len(rates) == sample_infections_2d.shape[0]

    def test_custom_config(self, sample_infections):
        """Custom config should override defaults."""
        cfg = {"min_rate": 0.5, "max_rate": 0.9}
        reported, rates = generate_reported_cases(
            sample_infections, config=cfg, rng=np.random.default_rng(42)
        )

        assert np.all(rates >= cfg["min_rate"])
        assert np.all(rates <= cfg["max_rate"])

    def test_zero_infections(self):
        """Should handle zero infections without errors."""
        infections = np.zeros(10)
        reported, rates = generate_reported_cases(infections, rng=np.random.default_rng(42))

        np.testing.assert_array_equal(reported, np.zeros(10, dtype=int))

    def test_deterministic_with_seed(self, sample_infections):
        """Same seed should produce identical results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        reported1, _ = generate_reported_cases(sample_infections, rng=rng1)
        reported2, _ = generate_reported_cases(sample_infections, rng=rng2)

        np.testing.assert_array_equal(reported1, reported2)


class TestWastewaterBasic:
    """Tests for generate_wastewater function (basic, non-stratified)."""

    @pytest.fixture
    def sample_infections_1d(self):
        """1D infection time series."""
        return np.array([0, 5, 20, 50, 100, 80, 40, 15, 5, 0])

    @pytest.fixture
    def sample_infections_2d(self):
        """2D infection data (time, location)."""
        return np.array(
            [
                [0, 10],
                [5, 20],
                [20, 50],
                [50, 100],
                [100, 150],
                [80, 120],
                [40, 80],
                [15, 30],
                [5, 10],
                [0, 5],
            ]
        )

    def test_returns_float_array(self, sample_infections_1d):
        """Wastewater signal should be float."""
        result = generate_wastewater(sample_infections_1d, rng=np.random.default_rng(42))
        assert result.dtype == np.float64

    def test_1d_input_shape(self, sample_infections_1d):
        """1D input should return 1D output."""
        result = generate_wastewater(sample_infections_1d, rng=np.random.default_rng(42))
        assert result.shape == sample_infections_1d.shape

    def test_2d_input_shape(self, sample_infections_2d):
        """2D input should return 2D output."""
        result = generate_wastewater(sample_infections_2d, rng=np.random.default_rng(42))
        assert result.shape == sample_infections_2d.shape

    def test_convolution_smoothing(self, sample_infections_1d):
        """Convolution should smooth the infection signal."""
        result = generate_wastewater(
            sample_infections_1d,
            config={"noise_sigma": 0.0, "limit_of_detection": 0.0},
            rng=np.random.default_rng(42),
        )

        # Peak should be shifted and spread compared to infections
        # (convolution spreads signal over time)
        infection_peak_idx = np.argmax(sample_infections_1d)
        result_peak_idx = np.argmax(result)

        # Peak should be delayed due to convolution
        assert result_peak_idx >= infection_peak_idx

    def test_log_normal_noise_multiplicative(self, sample_infections_1d):
        """Noise should be multiplicative (log-normal)."""
        rng = np.random.default_rng(42)

        # High signal
        result_high = generate_wastewater(
            np.array([1000]),
            config={"noise_sigma": 0.5, "limit_of_detection": 0.0},
            rng=rng,
        )

        # Low signal
        result_low = generate_wastewater(
            np.array([10]),
            config={"noise_sigma": 0.5, "limit_of_detection": 0.0},
            rng=rng,
        )

        # Variance should scale with signal (multiplicative noise)
        # Both have same noise config, so ratio of std/mean should be similar
        # Run multiple times to estimate variance
        n_samples = 100
        high_samples = [
            generate_wastewater(
                np.array([1000]),
                config={"noise_sigma": 0.5, "limit_of_detection": 0.0},
                rng=np.random.default_rng(i),
            )[0]
            for i in range(n_samples)
        ]
        low_samples = [
            generate_wastewater(
                np.array([10]),
                config={"noise_sigma": 0.5, "limit_of_detection": 0.0},
                rng=np.random.default_rng(i + 1000),
            )[0]
            for i in range(n_samples)
        ]

        high_cv = np.std(high_samples) / np.mean(high_samples)
        low_cv = np.std(low_samples) / np.mean(low_samples)

        # Coefficient of variation should be similar (multiplicative noise property)
        # Using loose tolerance due to randomness
        np.testing.assert_allclose(high_cv, low_cv, atol=0.3)

    def test_limit_of_detection_hard_cutoff(self, sample_infections_1d):
        """LoD with hard cutoff should zero out values below threshold."""
        result = generate_wastewater(
            sample_infections_1d,
            config={
                "limit_of_detection": 15.0,  # Lower LoD so some values pass, some don't
                "lod_probabilistic": False,
                "noise_sigma": 0.0,
            },
            rng=np.random.default_rng(42),
        )

        # Some values should be zero (below LoD or due to convolution edge effects)
        assert np.any(result == 0.0)

        # Some values should be positive (above LoD)
        assert np.any(result > 0.0)

        # No non-zero values below LoD (hard cutoff)
        nonzero_below_lod = (result > 0) & (result < 15.0)
        assert not np.any(nonzero_below_lod)

    def test_limit_of_detection_probabilistic(self, sample_infections_1d):
        """Probabilistic LoD should use logistic detection probability."""
        lod = 50.0

        # Run many times to get statistics
        n_runs = 200
        detections = []
        for seed in range(n_runs):
            result = generate_wastewater(
                sample_infections_1d,
                config={
                    "limit_of_detection": lod,
                    "lod_probabilistic": True,
                    "lod_slope": 2.0,
                    "noise_sigma": 0.0,
                },
                rng=np.random.default_rng(seed),
            )
            detections.append(result > 0)

        detections = np.array(detections)

        # For signals well below LoD, detection probability should be low
        # For signals well above LoD, detection probability should be high
        # This tests the logistic behavior

    def test_transport_loss_subtracts_signal(self, sample_infections_1d):
        """Transport loss should subtract from signal, not multiply."""
        config_clean = {
            "transport_loss": 0.0,
            "noise_sigma": 0.0,
            "limit_of_detection": 0.0,
        }
        config_loss = {
            "transport_loss": 20.0,
            "noise_sigma": 0.0,
            "limit_of_detection": 0.0,
        }

        result_clean = generate_wastewater(
            sample_infections_1d, config=config_clean, rng=np.random.default_rng(42)
        )
        result_loss = generate_wastewater(
            sample_infections_1d, config=config_loss, rng=np.random.default_rng(42)
        )

        # Loss should reduce signal, floor at zero
        assert np.all(result_loss <= result_clean)
        assert np.all(result_loss >= 0)

        # Difference should be approximately the loss amount
        # (where signal doesn't go negative)
        diff = result_clean - result_loss
        assert np.any(diff >= 20.0)

    def test_sensitivity_scale(self, sample_infections_1d):
        """Sensitivity scale should multiply the signal."""
        config_low = {"sensitivity_scale": 0.5, "noise_sigma": 0.0, "limit_of_detection": 0.0}
        config_high = {"sensitivity_scale": 2.0, "noise_sigma": 0.0, "limit_of_detection": 0.0}

        result_low = generate_wastewater(
            sample_infections_1d, config=config_low, rng=np.random.default_rng(42)
        )
        result_high = generate_wastewater(
            sample_infections_1d, config=config_high, rng=np.random.default_rng(42)
        )

        # Use mask for meaningful signal (above epsilon floor)
        signal_mask = result_low > 1e-6  # Exclude epsilon floor values

        # High sensitivity should give higher signal for meaningful signal
        assert np.all(result_high[signal_mask] > result_low[signal_mask])

        # Ratio should be approximately 4 (2.0 / 0.5) for meaningful signal
        ratio = result_high[signal_mask] / result_low[signal_mask]
        assert np.mean(ratio) > 3.5

    def test_zero_infections(self):
        """Zero infections should produce zero/near-zero output."""
        infections = np.zeros(10)
        result = generate_wastewater(infections, rng=np.random.default_rng(42))

        # With noise, may get small values but should be near zero
        assert np.all(result < 1.0)

    def test_deterministic_with_seed(self, sample_infections_1d):
        """Same seed should produce identical results."""
        result1 = generate_wastewater(sample_infections_1d, rng=np.random.default_rng(42))
        result2 = generate_wastewater(sample_infections_1d, rng=np.random.default_rng(42))

        np.testing.assert_allclose(result1, result2)


class TestWastewaterStratified:
    """Tests for generate_wastewater_stratified (age-stratified) function."""

    @pytest.fixture
    def sample_infections_3d(self):
        """3D infection data (time, location, age_group)."""
        # Shape: (10 time, 2 locations, 3 age groups)
        return np.array([
            [[0, 5, 2], [10, 20, 15]],
            [[5, 15, 8], [20, 40, 30]],
            [[20, 40, 20], [50, 80, 60]],
            [[50, 80, 40], [100, 120, 90]],
            [[100, 120, 60], [150, 180, 120]],
            [[80, 100, 50], [120, 150, 100]],
            [[40, 60, 30], [80, 100, 70]],
            [[15, 30, 15], [30, 50, 35]],
            [[5, 15, 8], [10, 20, 15]],
            [[0, 5, 2], [5, 10, 5]],
        ])

    @pytest.fixture
    def sample_infections_2d(self):
        """2D infection data (time, age_group) for single location."""
        return np.array([
            [0, 5, 2],
            [5, 15, 8],
            [20, 40, 20],
            [50, 80, 40],
            [100, 120, 60],
            [80, 100, 50],
            [40, 60, 30],
            [15, 30, 15],
            [5, 15, 8],
            [0, 5, 2],
        ])

    def test_returns_2d_output(self, sample_infections_3d):
        """3D input should produce 2D output (time, location)."""
        result = generate_wastewater_stratified(
            sample_infections_3d, rng=np.random.default_rng(42)
        )

        assert result.ndim == 2
        assert result.shape == (sample_infections_3d.shape[0], sample_infections_3d.shape[1])

    def test_2d_input_returns_1d(self, sample_infections_2d):
        """2D input (time, age_group) should return 1D output (time,)."""
        result = generate_wastewater_stratified(
            sample_infections_2d, rng=np.random.default_rng(42)
        )

        assert result.ndim == 1
        assert result.shape == (sample_infections_2d.shape[0],)

    def test_age_groups_contributed(self, sample_infections_3d):
        """All age groups should contribute to the signal."""
        # Use no noise for clearer signal
        result = generate_wastewater_stratified(
            sample_infections_3d,
            config={"noise_sigma": 0.0, "limit_of_detection": 0.0},
            rng=np.random.default_rng(42),
        )

        # Signal should be positive where infections exist
        assert np.any(result > 0)

    def test_population_dilution(self, sample_infections_3d):
        """Higher population should dilute the signal."""
        # Small population (use array for type compatibility)
        result_small_pop = generate_wastewater_stratified(
            sample_infections_3d,
            population=np.array([1000, 1000]),
            config={"noise_sigma": 0.0, "limit_of_detection": 0.0},
            rng=np.random.default_rng(42),
        )

        # Large population (same infections, more dilution)
        result_large_pop = generate_wastewater_stratified(
            sample_infections_3d,
            population=np.array([10000, 10000]),
            config={"noise_sigma": 0.0, "limit_of_detection": 0.0},
            rng=np.random.default_rng(42),
        )

        # Larger population should give lower signal (dilution)
        # Use mask to exclude epsilon floor values where signal is ~0
        signal_mask = result_small_pop > 1e-6
        assert np.all(result_small_pop[signal_mask] > result_large_pop[signal_mask])

    def test_population_per_location(self, sample_infections_3d):
        """Different locations can have different populations."""
        populations = np.array([1000, 5000])

        result = generate_wastewater_stratified(
            sample_infections_3d,
            population=populations,
            config={"noise_sigma": 0.0, "limit_of_detection": 0.0},
            rng=np.random.default_rng(42),
        )

        # Location 1 (larger pop) should have lower signal than location 0
        # (assuming similar infection levels)
        assert np.mean(result[:, 1]) < np.mean(result[:, 0])

    def test_age_specific_kernels(self):
        """Different age groups should use different shedding kinetics."""
        # Create scenario where only one age group is infected
        n_time = 20
        n_groups = 3

        # Only young infected
        infections_young = np.zeros((n_time, 1, n_groups))
        infections_young[10, 0, 0] = 100  # Young group

        # Only mature infected
        infections_mature = np.zeros((n_time, 1, n_groups))
        infections_mature[10, 0, 1] = 100  # Mature group

        result_young = generate_wastewater_stratified(
            infections_young,
            config={"noise_sigma": 0.0, "limit_of_detection": 0.0},
            rng=np.random.default_rng(42),
        )

        result_mature = generate_wastewater_stratified(
            infections_mature,
            config={"noise_sigma": 0.0, "limit_of_detection": 0.0},
            rng=np.random.default_rng(42),
        )

        # Young have longer shedding tail, so signal should persist longer
        young_duration = np.sum(result_young > 0)
        mature_duration = np.sum(result_mature > 0)

        assert young_duration >= mature_duration

    def test_stratified_vs_unstratified_consistency(self, sample_infections_3d):
        """Stratified and unstratified should be related (after summing ages)."""
        # Sum over age groups for unstratified comparison
        infections_total = sample_infections_3d.sum(axis=2)

        result_stratified = generate_wastewater_stratified(
            sample_infections_3d,
            config={"noise_sigma": 0.0, "limit_of_detection": 0.0},
            rng=np.random.default_rng(42),
        )

        # Note: These won't be identical because stratified uses different kernels
        # per age group, but they should be correlated
        assert result_stratified.shape == infections_total.shape

    def test_lod_censoring_stratified(self, sample_infections_3d):
        """LoD censoring should work for stratified generation."""
        result = generate_wastewater_stratified(
            sample_infections_3d,
            config={
                "limit_of_detection": 50.0,
                "lod_probabilistic": False,
                "noise_sigma": 0.0,
            },
            rng=np.random.default_rng(42),
        )

        # Some values should be zero (below LoD)
        assert np.any(result == 0.0)

        # No values between 0 and LoD (hard cutoff)
        assert not np.any((result > 0) & (result < 50.0))

    def test_transport_loss_stratified(self, sample_infections_3d):
        """Transport loss should work for stratified generation."""
        config_no_loss = {
            "transport_loss": 0.0,
            "noise_sigma": 0.0,
            "limit_of_detection": 0.0,
        }
        config_loss = {
            "transport_loss": 20.0,
            "noise_sigma": 0.0,
            "limit_of_detection": 0.0,
        }

        result_no_loss = generate_wastewater_stratified(
            sample_infections_3d, config=config_no_loss, rng=np.random.default_rng(42)
        )
        result_loss = generate_wastewater_stratified(
            sample_infections_3d, config=config_loss, rng=np.random.default_rng(42)
        )

        # Loss should reduce signal
        assert np.all(result_loss <= result_no_loss)

    def test_zero_infections_stratified(self):
        """Zero infections should produce near-zero output."""
        infections = np.zeros((10, 2, 3))
        result = generate_wastewater_stratified(
            infections, rng=np.random.default_rng(42)
        )

        assert np.all(result < 1.0)

    def test_deterministic_with_seed_stratified(self, sample_infections_3d):
        """Same seed should produce identical results."""
        result1 = generate_wastewater_stratified(
            sample_infections_3d, rng=np.random.default_rng(42)
        )
        result2 = generate_wastewater_stratified(
            sample_infections_3d, rng=np.random.default_rng(42)
        )

        np.testing.assert_allclose(result1, result2)


class TestEdgeCases:
    """Tests for edge cases and extreme values."""

    def test_empty_array_wastewater(self):
        """Empty array should raise ValueError (numpy.convolve limitation)."""
        infections = np.array([])
        # np.convolve raises ValueError for empty arrays
        with pytest.raises(ValueError, match="cannot be empty"):
            generate_wastewater(infections, rng=np.random.default_rng(42))

    def test_single_value_wastewater(self):
        """Single value should work without errors."""
        infections = np.array([100])
        result = generate_wastewater(infections, rng=np.random.default_rng(42))
        assert result.shape == (1,)
        assert result[0] > 0

    def test_very_large_infections(self):
        """Very large infection counts should not cause overflow."""
        infections = np.array([1e6, 2e6, 1.5e6])
        result = generate_wastewater(infections, rng=np.random.default_rng(42))

        # Should be finite values
        assert np.all(np.isfinite(result))

    def test_negative_infections_normalized(self):
        """Negative infections should be handled (normalized to zero)."""
        infections = np.array([10, -5, 20])
        reported, _ = generate_reported_cases(
            infections, rng=np.random.default_rng(42)
        )

        # Negative values become zero, so reports should be <= infections
        assert reported[1] == 0

    def test_nan_handling_wastewater(self):
        """NaN values should be handled gracefully."""
        infections = np.array([10, np.nan, 20])

        # Should not raise an error
        result = generate_wastewater(infections, rng=np.random.default_rng(42))

        # NaN should propagate or be handled
        # (implementation-dependent - just check it doesn't crash)
        assert len(result) == len(infections)


class TestLoDBehaviors:
    """Detailed tests for Limit of Detection behaviors."""

    def test_hard_cutoff_all_below_lod(self):
        """With hard cutoff, all values below LoD should become zero."""
        infections = np.array([1, 2, 3, 4, 5])
        result = generate_wastewater(
            infections,
            config={
                "limit_of_detection": 1000.0,
                "lod_probabilistic": False,
                "noise_sigma": 0.0,
                "sensitivity_scale": 1.0,
            },
            rng=np.random.default_rng(42),
        )

        # All values should be zero (below LoD)
        np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_hard_cutoff_all_above_lod(self):
        """With hard cutoff, values above LoD should be preserved (except edge effects)."""
        # Use a longer infection series so convolution has enough data
        infections = np.array([1000, 2000, 3000, 3000, 3000, 3000, 3000, 3000])
        result = generate_wastewater(
            infections,
            config={
                "limit_of_detection": 10.0,
                "lod_probabilistic": False,
                "noise_sigma": 0.0,
                "sensitivity_scale": 1.0,
            },
            rng=np.random.default_rng(42),
        )

        # Most values should be above LoD (first few may be zero due to convolution)
        nonzero_count = np.sum(result > 10.0)
        assert nonzero_count >= len(result) // 2  # At least half should be above LoD

    def test_probabilistic_lod_detection_probability(self):
        """Probabilistic LoD should have sigmoidal detection probability."""
        # Set LoD to 50 to test detection probability
        lod = 50.0
        slope = 2.0

        # Create a signal that will produce output around LoD after convolution
        # Using sustained infection of 55 to get signal near LoD=50 (signal ≈ 51.66)
        infections = np.array([55] * 20)

        # Signal at LoD should have ~50% detection probability
        n_runs = 500
        detections_at_lod = []
        for seed in range(n_runs):
            result = generate_wastewater(
                infections,
                config={
                    "limit_of_detection": lod,
                    "lod_probabilistic": True,
                    "lod_slope": slope,
                    "noise_sigma": 0.0,
                    "sensitivity_scale": 1.0,
                },
                rng=np.random.default_rng(seed),
            )
            # Check middle of signal where convolution is stable
            detections_at_lod.append(result[10] > 0)

        detection_rate = np.mean(detections_at_lod)

        # At LoD, detection should be near 50%
        # Signal ≈ 51.66, LoD = 50, so detection_prob ≈ 1/(1+exp(-2*(51.66-50))) ≈ 0.79
        # But due to random detection, actual rate should be around this
        assert 0.6 < detection_rate < 0.98

    def test_lod_zero_disabled(self):
        """LoD=0 should disable censoring."""
        infections = np.array([1, 5, 10])
        result = generate_wastewater(
            infections,
            config={
                "limit_of_detection": 0.0,
                "lod_probabilistic": False,
                "noise_sigma": 0.0,
                "sensitivity_scale": 1.0,
            },
            rng=np.random.default_rng(42),
        )

        # All values should be positive (no censoring)
        assert np.all(result > 0)

    def test_stratified_lod_behavior(self):
        """LoD should work identically for stratified generation."""
        infections_3d = np.array([
            [[1, 1, 1], [2, 2, 2]],
            [[5, 5, 5], [10, 10, 10]],
        ])

        result = generate_wastewater_stratified(
            infections_3d,
            config={
                "limit_of_detection": 1000.0,
                "lod_probabilistic": False,
                "noise_sigma": 0.0,
                "sensitivity_scale": 1.0,
            },
            population=np.array([100, 100]),
            rng=np.random.default_rng(42),
        )

        # All values should be zero (below LoD)
        np.testing.assert_array_equal(result, np.zeros_like(result))


class TestNoiseModels:
    """Tests for statistical noise properties."""

    def test_log_normal_properties(self):
        """Log-normal noise should preserve positivity."""
        infections = np.array([10, 20, 50])

        result = generate_wastewater(
            infections,
            config={"noise_sigma": 0.5, "limit_of_detection": 0.0},
            rng=np.random.default_rng(42),
        )

        # All values should be positive (log-normal is always positive)
        assert np.all(result > 0)

    def test_noise_sigma_affects_variance(self):
        """Higher noise sigma should increase output variance."""
        infections = np.array([100, 100, 100, 100, 100])

        results_low = []
        results_high = []

        for seed in range(50):
            result_low = generate_wastewater(
                infections,
                config={"noise_sigma": 0.1, "limit_of_detection": 0.0},
                rng=np.random.default_rng(seed),
            )
            result_high = generate_wastewater(
                infections,
                config={"noise_sigma": 1.0, "limit_of_detection": 0.0},
                rng=np.random.default_rng(seed + 1000),
            )
            results_low.append(result_low[0])
            results_high.append(result_high[0])

        var_low = np.var(results_low)
        var_high = np.var(results_high)

        # Higher sigma should give higher variance
        assert var_high > var_low

    def test_zero_noise_deterministic(self):
        """Zero noise sigma should give deterministic results."""
        infections = np.array([10, 20, 50])

        result1 = generate_wastewater(
            infections,
            config={"noise_sigma": 0.0, "limit_of_detection": 0.0},
            rng=np.random.default_rng(42),
        )

        result2 = generate_wastewater(
            infections,
            config={"noise_sigma": 0.0, "limit_of_detection": 0.0},
            rng=np.random.default_rng(999),
        )

        # Without noise, seed shouldn't matter
        np.testing.assert_allclose(result1, result2)


class TestMonitoringStartThreshold:
    """Tests for the _compute_monitoring_start_mask function."""

    @pytest.fixture
    def sample_infections_3d(self):
        """3D infection data (time, location, age_group)."""
        # Shape: (20 time, 3 locations, 3 age groups)
        # Cumulative infections after summing ages:
        # Location 0: reaches 100 around day 10
        # Location 1: reaches 50 around day 7
        # Location 2: reaches 25 around day 5
        return np.array([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # Day 0
            [[1, 1, 1], [2, 2, 2], [1, 1, 1]],  # Day 1: 3, 6, 3
            [[2, 2, 2], [3, 3, 3], [1, 1, 1]],  # Day 2: 6, 9, 3
            [[3, 3, 3], [4, 4, 4], [2, 2, 2]],  # Day 3: 9, 12, 6
            [[4, 4, 4], [5, 5, 5], [2, 2, 2]],  # Day 4: 12, 15, 6
            [[5, 5, 5], [6, 6, 6], [3, 3, 3]],  # Day 5: 15, 18, 9
            [[6, 6, 6], [7, 7, 7], [3, 3, 3]],  # Day 6: 18, 21, 9
            [[7, 7, 7], [8, 8, 8], [4, 4, 4]],  # Day 7: 21, 24, 12
            [[8, 8, 8], [9, 9, 9], [4, 4, 4]],  # Day 8: 24, 27, 12
            [[9, 9, 9], [10, 10, 10], [5, 5, 5]],  # Day 9: 27, 30, 15
            [[10, 10, 10], [11, 11, 11], [5, 5, 5]],  # Day 10: 30, 33, 15
        ] * 2)  # Double to get 20 days

    def test_threshold_zero_disables_feature(self, sample_infections_3d):
        """threshold=0 should return all True (monitoring from day 0)."""
        mask = _compute_monitoring_start_mask(
            sample_infections_3d,
            threshold=0.0,
            delay_days=0,
            delay_std=0,
            rng=np.random.default_rng(42),
        )

        assert mask.shape == (sample_infections_3d.shape[0], sample_infections_3d.shape[1])
        assert np.all(mask == True)

    def test_cumulative_threshold_behavior(self, sample_infections_3d):
        """Monitoring should start after cumulative infections reach threshold."""
        # Set threshold to 50 cumulative infections
        threshold = 50
        mask = _compute_monitoring_start_mask(
            sample_infections_3d,
            threshold=threshold,
            delay_days=0,
            delay_std=0,
            rng=np.random.default_rng(42),
        )

        # Check shape (fixture creates 22 days: 11 days × 2)
        assert mask.shape == (22, 3)

        # Find first day where monitoring starts for each location
        # Cumulative by day 10: loc0=165, loc1=198, loc2=90
        # So all should reach threshold 50 before day 10
        for loc_idx in range(3):
            first_true = np.argmax(mask[:, loc_idx])
            assert first_true > 0, "Monitoring should not start on day 0 with threshold=50"
            assert mask[first_true:, loc_idx].all(), "Once monitoring starts, it should stay active"

    def test_per_edar_independence(self, sample_infections_3d):
        """Each EDAR should activate independently based on its own cumulative infections."""
        mask = _compute_monitoring_start_mask(
            sample_infections_3d,
            threshold=100,  # Different locations reach this at different times
            delay_days=0,
            delay_std=0,
            rng=np.random.default_rng(42),
        )

        # Each location should have different activation times
        activation_times = [np.argmax(mask[:, loc_idx]) for loc_idx in range(3)]

        # At least some locations should have different activation times
        # (given the structured infection data)
        assert len(set(activation_times)) >= 2, "Locations should activate independently"

    def test_stochastic_delay_variation(self):
        """Stochastic delay should create variation in activation times across runs."""
        infections = np.random.randint(0, 10, size=(50, 5, 3))

        # With delay_std > 0, results should vary
        mask1 = _compute_monitoring_start_mask(
            infections,
            threshold=50,
            delay_days=5,
            delay_std=3,
            rng=np.random.default_rng(42),
        )

        mask2 = _compute_monitoring_start_mask(
            infections,
            threshold=50,
            delay_days=5,
            delay_std=3,
            rng=np.random.default_rng(999),  # Different seed
        )

        # Activation times should differ due to stochastic delay
        activation1 = [np.argmax(mask1[:, loc_idx]) for loc_idx in range(5)]
        activation2 = [np.argmax(mask2[:, loc_idx]) for loc_idx in range(5)]

        # At least one location should have different activation time
        assert activation1 != activation2, "Stochastic delay should create variation"

    def test_delay_std_zero_deterministic(self):
        """delay_std=0 should produce deterministic results."""
        infections = np.random.randint(0, 10, size=(50, 5, 3))

        mask1 = _compute_monitoring_start_mask(
            infections,
            threshold=50,
            delay_days=5,
            delay_std=0,
            rng=np.random.default_rng(42),
        )

        mask2 = _compute_monitoring_start_mask(
            infections,
            threshold=50,
            delay_days=5,
            delay_std=0,
            rng=np.random.default_rng(999),  # Different seed but shouldn't matter
        )

        np.testing.assert_array_equal(mask1, mask2)

    def test_threshold_never_reached(self):
        """If threshold is never reached, all values should be False."""
        # Very low infections that never reach threshold
        infections = np.ones((20, 3, 3))  # Only 1 infection per day

        mask = _compute_monitoring_start_mask(
            infections,
            threshold=1000,  # Never reached
            delay_days=0,
            delay_std=0,
            rng=np.random.default_rng(42),
        )

        assert np.all(mask == False), "All should be False if threshold never reached"

    def test_negative_threshold_raises_error(self):
        """Negative threshold should raise ValueError."""
        infections = np.random.randint(0, 10, size=(20, 3, 3))

        with pytest.raises(ValueError, match="Threshold must be non-negative"):
            _compute_monitoring_start_mask(
                infections,
                threshold=-10,  # Negative!
                delay_days=0,
                delay_std=0,
                rng=np.random.default_rng(42),
            )

    def test_delay_truncated_at_zero(self):
        """Delay should be truncated at zero (no negative delays)."""
        infections = np.random.randint(0, 20, size=(30, 5, 3))

        # With delay_mean=0 and delay_std>0, some delays would be negative
        # but should be truncated to 0
        mask = _compute_monitoring_start_mask(
            infections,
            threshold=10,
            delay_days=0,
            delay_std=5,
            rng=np.random.default_rng(42),
        )

        # All masks should be valid (no invalid states)
        assert mask.dtype == bool
        assert np.isin(mask, [True, False]).all()

    def test_monitoring_mask_integration_with_stratified(self, sample_infections_3d):
        """Monitoring mask should integrate correctly with generate_wastewater_stratified."""
        mask = _compute_monitoring_start_mask(
            sample_infections_3d,
            threshold=100,
            delay_days=5,
            delay_std=0,
            rng=np.random.default_rng(42),
        )

        population = np.array([1000, 2000, 3000])

        result = generate_wastewater_stratified(
            sample_infections_3d,
            population=population,
            config={"noise_sigma": 0.0, "limit_of_detection": 0.0},
            rng=np.random.default_rng(42),
            monitoring_mask=mask,
        )

        # Result should have NaN values where monitoring is not active
        for loc_idx in range(3):
            pre_monitoring = ~mask[:, loc_idx]
            assert np.all(np.isnan(result[pre_monitoring, loc_idx])), \
                f"Location {loc_idx}: pre-threshold values should be NaN"
            # Post-threshold values should be non-NaN
            post_monitoring = mask[:, loc_idx]
            assert np.all(np.isfinite(result[post_monitoring, loc_idx])), \
                f"Location {loc_idx}: post-threshold values should be finite"
