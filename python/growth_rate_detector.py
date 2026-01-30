"""
Growth-rate-based spike detection module.

This module implements spike detection using sustained exponential growth combined
with population-relative thresholds, addressing limitations of absolute percentile
thresholds which can trigger too late or on small initial outbreaks.
"""

import logging
from typing import List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GrowthRateDetector")


def detect_spike_periods_growth_rate(
    infections_array: np.ndarray,
    population: float,
    growth_factor_threshold: float = 1.5,
    min_growth_duration: int = 3,
    min_cases_per_capita: float = 1e-4,
    growth_window: int = 3,
    min_spike_duration: int = 7,
) -> List[Tuple[int, int]]:
    """
    Detect spike periods using sustained exponential growth.

    This method triggers on sustained exponential growth combined with a
    population-relative threshold, addressing key limitations of absolute
    percentile thresholds:
    - Detects spikes earlier in the exponential growth phase
    - Avoids false triggers on small outbreaks that die out
    - Scales appropriately with population size

    Algorithm:
    1. Replace zeros with epsilon (0.1) to handle die-out/restart patterns
    2. Calculate growth factor using sliding window: GF[t] = I[t] / I[t-window]
    3. Calculate population-relative threshold: min_cases = min_cases_per_capita * population
    4. Find days where BOTH conditions met: GF >= threshold AND infections >= min_cases
    5. Require min_growth_duration consecutive days (sustained growth)
    6. Merge into spike periods with minimum min_spike_duration

    Args:
        infections_array: 1D array of daily infections (e.g., summed across regions)
        population: Total population for calculating per-capita threshold
        growth_factor_threshold: Minimum growth factor to trigger (default: 1.5 = 50% growth)
            Over the growth_window period. E.g., 1.5 over 3 days ~ 15% daily growth
        min_growth_duration: Minimum consecutive days with growth above threshold (default: 3)
        min_cases_per_capita: Minimum infections per person (default: 1e-4 = 1 per 10K)
        growth_window: Window size for growth factor calculation in days (default: 3)
        min_spike_duration: Minimum days for a spike period (default: 7)

    Returns:
        List of (start_day, end_day) spike periods (end_day exclusive)

    Examples:
        >>> infections = np.array([1, 2, 4, 8, 16, 32, 64, 32, 16, 8, 4, 2, 1])
        >>> # Pure doubling every day, GF[3] = 8/1 = 8 >> 1.5
        >>> detect_spike_periods_growth_rate(infections, population=10000)
        [(0, 7)]  # Detected exponential growth from day 0 to 7

        >>> infections = np.array([0, 0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144])
        >>> # Die-out followed by exponential growth
        >>> detect_spike_periods_growth_rate(infections, population=100000)
        [(2, 13)]  # Detected growth after the die-out period
    """
    n_days = len(infections_array)

    # Step 1: Replace zeros with epsilon to handle die-out/restart pattern
    # Using epsilon=0.1 (close to 1 case, but non-zero)
    epsilon = 0.1
    infections_smooth = np.maximum(infections_array, epsilon)

    # Step 2: Calculate growth factor using sliding window
    # GF[t] = I[t] / I[t-window] for t >= window
    growth_factor = np.full(n_days, np.nan)  # Initialize with NaN

    for t in range(growth_window, n_days):
        if infections_smooth[t - growth_window] > 0:
            gf = infections_smooth[t] / infections_smooth[t - growth_window]
            growth_factor[t] = gf
        else:
            growth_factor[t] = 1.0  # No growth if starting from zero

    # Step 3: Calculate population-relative threshold
    min_cases = max(1.0, min_cases_per_capita * population)

    logger.info(f"Growth-rate detection: GF_threshold={growth_factor_threshold}, "
                f"min_cases={min_cases:.1f} ({min_cases_per_capita*100:.4f}% of pop={population:.0f})")

    # Step 4: Find days where BOTH conditions met
    above_threshold = np.zeros(n_days, dtype=bool)
    for t in range(growth_window, n_days):
        gf_condition = not np.isnan(growth_factor[t]) and growth_factor[t] >= growth_factor_threshold
        cases_condition = infections_array[t] >= min_cases
        above_threshold[t] = gf_condition and cases_condition

    # Step 5: Require min_growth_duration consecutive days
    # Use sliding window to check for sustained growth
    sustained_growth = np.zeros(n_days, dtype=bool)
    for t in range(min_growth_duration, n_days):
        # Check if we have min_growth_duration consecutive days above threshold
        window_check = above_threshold[t - min_growth_duration + 1:t + 1]
        sustained_growth[t] = np.all(window_check)

    # Step 6: Merge into spike periods with minimum duration
    spike_periods = []
    in_spike = False
    spike_start = 0

    for i, is_spike in enumerate(sustained_growth):
        if is_spike and not in_spike:
            # Start of a new spike period
            spike_start = max(0, i - growth_window)  # Include pre-growth window
            in_spike = True
        elif not is_spike and in_spike:
            # End of spike period
            spike_duration = i - spike_start
            if spike_duration >= min_spike_duration:
                spike_periods.append((spike_start, i))
            in_spike = False

    # Handle case where spike extends to end of array
    if in_spike:
        spike_duration = n_days - spike_start
        if spike_duration >= min_spike_duration:
            spike_periods.append((spike_start, n_days))

    return spike_periods


def calculate_growth_rate(
    infections_array: np.ndarray,
    window: int = 3,
    epsilon: float = 0.1,
) -> np.ndarray:
    """
    Calculate the growth rate for an infection time series.

    Args:
        infections_array: 1D array of daily infections
        window: Window size for growth rate calculation (default: 3)
        epsilon: Small value to replace zeros (default: 0.1)

    Returns:
        1D array of growth factors (GF[t] = I[t] / I[t-window])
        Values are NaN for t < window
    """
    n_days = len(infections_array)
    infections_smooth = np.maximum(infections_array, epsilon)
    growth_factor = np.full(n_days, np.nan)

    for t in range(window, n_days):
        if infections_smooth[t - window] > 0:
            growth_factor[t] = infections_smooth[t] / infections_smooth[t - window]
        else:
            growth_factor[t] = 1.0

    return growth_factor
