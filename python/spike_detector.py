"""
Spike detection module for analyzing infection dynamics in baseline simulations.

This module provides utilities to detect significant infection spike periods in
simulation outputs, which can be used to time interventions realistically.
"""

import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
import xarray as xr
from scipy.signal import find_peaks

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SpikeDetector")


def detect_spike_periods(
    infections_array: np.ndarray,
    threshold_pct: float = 0.1,
    min_duration: int = 7,
    method: str = "percentile",
    population: Optional[float] = None,
    growth_factor_threshold: float = 1.5,
    min_growth_duration: int = 3,
    min_cases_per_capita: float = 1e-4,
    growth_window: int = 3,
) -> List[Tuple[int, int]]:
    """
    Detect significant spike periods in infection time series.

    Args:
        infections_array: 1D array of daily infections (e.g., summed across regions)
        threshold_pct: Percentile threshold for spike detection (default: 0.1 = 10th percentile)
            For method='percentile': values above this percentile are considered elevated
            For method='prominence': relative prominence threshold for peak detection
        min_duration: Minimum days above threshold to qualify as spike (default: 7)
        method: Detection method - 'percentile', 'prominence', or 'growth_rate'
        population: Total population (required for growth_rate method)
        growth_factor_threshold: Minimum growth factor for growth_rate method (default: 1.5)
        min_growth_duration: Minimum consecutive days of growth for growth_rate (default: 3)
        min_cases_per_capita: Minimum cases per person for growth_rate (default: 1e-4)
        growth_window: Window size for growth rate calculation in days (default: 3)

    Returns:
        List of (start_day, end_day) spike periods (end_day exclusive)

    Examples:
        >>> infections = np.array([10, 15, 100, 150, 80, 20, 10])
        >>> detect_spike_periods(infections, threshold_pct=0.5, min_duration=2)
        [(2, 5)]  # Spike from day 2 to 5 (exclusive)
    """
    n_days = len(infections_array)

    if method == "percentile":
        # Calculate baseline threshold from early epidemic (days 0-30 or full array if shorter)
        baseline_window = min(30, n_days)
        baseline_infections = infections_array[:baseline_window]
        threshold = np.percentile(baseline_infections, threshold_pct * 100)

        logger.info(f"Threshold-based detection: threshold={threshold:.1f} ({threshold_pct*100:.0f}th percentile)")

        # Find periods where infections exceed threshold
        above_threshold = infections_array > threshold

        # Identify contiguous periods above threshold
        spike_periods = []
        in_spike = False
        spike_start = 0

        for i, is_above in enumerate(above_threshold):
            if is_above and not in_spike:
                # Start of a new spike period
                spike_start = i
                in_spike = True
            elif not is_above and in_spike:
                # End of spike period
                spike_duration = i - spike_start
                if spike_duration >= min_duration:
                    spike_periods.append((spike_start, i))
                in_spike = False

        # Handle case where spike extends to end of array
        if in_spike:
            spike_duration = n_days - spike_start
            if spike_duration >= min_duration:
                spike_periods.append((spike_start, n_days))

        # Filter out very low-activity spikes (mean infections < 2x threshold)
        filtered_periods = []
        for start, end in spike_periods:
            mean_infections = infections_array[start:end].mean()
            if mean_infections >= threshold * 2:
                filtered_periods.append((start, end))

        return filtered_periods

    elif method == "prominence":
        # Use scipy's find_peaks for more sophisticated peak detection
        # Prominence measures how much a peak stands out from surrounding data
        prominence_threshold = np.max(infections_array) * threshold_pct

        peaks, properties = find_peaks(
            infections_array,
            prominence=prominence_threshold,
            width=min_duration,
        )

        logger.info(f"Peak-based detection: found {len(peaks)} peaks with prominence>{prominence_threshold:.1f}")

        # Convert peaks to windows (using width information)
        spike_periods = []
        for i, peak_pos in enumerate(peaks):
            # Use the width at half prominence as the spike window
            # width is defined as (left, right) indices
            if "widths" in properties and "left_ips" in properties and "right_ips" in properties:
                left = int(properties["left_ips"][i])
                right = int(properties["right_ips"][i])
                # Ensure minimum width
                if right - left < min_duration:
                    half_width = min_duration // 2
                    left = max(0, peak_pos - half_width)
                    right = min(len(infections_array), peak_pos + half_width)
                spike_periods.append((left, min(right, n_days)))
            else:
                # Fallback: fixed window around peak
                half_width = min_duration
                left = max(0, peak_pos - half_width)
                right = min(n_days, peak_pos + half_width)
                spike_periods.append((left, right))

        return spike_periods

    elif method == "growth_rate":
        from growth_rate_detector import detect_spike_periods_growth_rate

        if population is None:
            raise ValueError(
                f"population parameter is required for growth_rate method. "
                f"Provide population size for per-capita threshold calculation."
            )

        logger.info(f"Growth-rate detection: GF_threshold={growth_factor_threshold}, "
                    f"min_growth_duration={min_growth_duration}, "
                    f"min_cases_per_capita={min_cases_per_capita}")

        return detect_spike_periods_growth_rate(
            infections_array=infections_array,
            population=population,
            growth_factor_threshold=growth_factor_threshold,
            min_growth_duration=min_growth_duration,
            min_cases_per_capita=min_cases_per_capita,
            growth_window=growth_window,
            min_spike_duration=min_duration,
        )

    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'percentile', 'prominence', or 'growth_rate'"
        )


def detect_spike_periods_from_zarr(
    zarr_path: str,
    threshold_pct: float = 0.1,
    min_duration: int = 7,
    method: str = "percentile",
    baseline_filter: bool = True,
    growth_factor_threshold: float = 1.5,
    min_growth_duration: int = 3,
    min_cases_per_capita: float = 1e-4,
    growth_window: int = 3,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Detect spike periods for all baseline runs in a zarr file.

    Args:
        zarr_path: Path to raw_synthetic_observations.zarr
        threshold_pct: Percentile threshold for spike detection (default: 0.1)
        min_duration: Minimum days for spike period (default: 7)
        method: Detection method - 'percentile', 'prominence', or 'growth_rate'
        baseline_filter: Only process runs where scenario_type == 'Baseline' (default: True)
        growth_factor_threshold: Minimum growth factor for growth_rate method (default: 1.5)
        min_growth_duration: Minimum consecutive days of growth for growth_rate (default: 3)
        min_cases_per_capita: Minimum cases per person for growth_rate (default: 1e-4)
        growth_window: Window size for growth rate calculation in days (default: 3)

    Returns:
        Dict mapping run_id -> list of (start_day, end_day) tuples

    Raises:
        FileNotFoundError: If zarr file doesn't exist
        ValueError: If no baseline runs found or spike detection fails

    Examples:
        >>> spikes = detect_spike_periods_from_zarr("baselines.zarr", threshold_pct=0.1)
        >>> for run_id, windows in spikes.items():
        ...     print(f"{run_id}: {windows}")
        0_Baseline: [(20, 45), (60, 80)]
        1_Baseline: [(15, 50)]
    """
    import os

    if not os.path.exists(zarr_path):
        raise FileNotFoundError(
            f"Zarr file not found: {zarr_path}\n"
            f"Please run baselines first and process outputs:\n"
            f"  uv run python synthetic_generator.py --baseline-only\n"
            f"  uv run python process_synthetic_outputs.py --runs-dir <output_folder>"
        )

    # Load ground truth infections
    ds = xr.open_zarr(zarr_path)

    if "infections_true" not in ds:
        raise ValueError(
            f"infections_true variable not found in {zarr_path}\n"
            f"Ensure process_synthetic_outputs.py was run with ground truth enabled."
        )

    infections = ds["infections_true"]  # (run_id, region_id, date)
    scenario_types = ds.get("synthetic_scenario_type", None)
    population_var = ds.get("population", None)

    # Extract population for growth_rate method
    # population shape: (run_id, region_id) -> sum over region_id for total population
    if population_var is not None and method == "growth_rate":
        # For each run, sum population across regions
        population_by_run = {}
        for run_id_val in population_var.run_id.values:
            pop = float(population_var.sel(run_id=str(run_id_val)).sum().values)
            population_by_run[str(run_id_val)] = pop
        logger.info(f"Extracted population for {len(population_by_run)} runs for growth_rate method")
    else:
        population_by_run = {}

    spikes_by_run = {}

    # Filter to baseline runs only
    if scenario_types is not None and baseline_filter:
        baseline_mask = [
            str(stype).strip() in ["Baseline", "baseline"]
            for stype in scenario_types.values
        ]
        run_ids_to_process = [
            str(rid) for rid, is_baseline in zip(infections.run_id.values, baseline_mask)
            if is_baseline
        ]
    else:
        run_ids_to_process = [str(rid) for rid in infections.run_id.values]

    if not run_ids_to_process:
        raise ValueError(
            f"No baseline runs found in {zarr_path}\n"
            f"Ensure synthetic_generator.py was run with --baseline-only"
        )

    logger.info(f"Detecting spikes in {len(run_ids_to_process)} baseline runs")

    for run_id in run_ids_to_process:
        run_infections = infections.sel(run_id=run_id)

        # Sum across regions for national-level spike detection
        # infections shape: (region_id, date) -> sum over region_id -> (date,)
        national_infections = run_infections.sum(dim="region_id").values

        # Get population for this run (for growth_rate method)
        run_population = population_by_run.get(run_id, None)

        # Detect spikes
        spike_windows = detect_spike_periods(
            national_infections,
            threshold_pct=threshold_pct,
            min_duration=min_duration,
            method=method,
            population=run_population,
            growth_factor_threshold=growth_factor_threshold,
            min_growth_duration=min_growth_duration,
            min_cases_per_capita=min_cases_per_capita,
            growth_window=growth_window,
        )

        if spike_windows:
            spikes_by_run[run_id] = spike_windows
            logger.info(f"  {run_id}: detected {len(spike_windows)} spike periods")
        else:
            logger.warning(f"  {run_id}: no spikes detected (threshold={threshold_pct}, min_duration={min_duration})")

    ds.close()

    if not spikes_by_run:
        raise ValueError(
            f"No spikes detected in any baseline run\n"
            f"Try lowering --spike-threshold (current={threshold_pct}) or "
            f"reducing --min-spike-duration (current={min_duration})"
        )

    return spikes_by_run


def print_spike_summary(spikes_by_run: Dict[str, List[Tuple[int, int]]]):
    """Print a summary of detected spike periods."""
    print("\n" + "=" * 60)
    print("SPIKE DETECTION SUMMARY")
    print("=" * 60)

    for run_id in sorted(spikes_by_run.keys()):
        windows = spikes_by_run[run_id]
        print(f"\n{run_id}:")
        for i, (start, end) in enumerate(windows):
            duration = end - start
            print(f"  Spike {i+1}: Day {start} -> {end} (duration={duration}d)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect infection spikes in baseline simulation outputs"
    )
    parser.add_argument(
        "zarr_path",
        help="Path to raw_synthetic_observations.zarr"
    )
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=0.1,
        help="Spike detection threshold percentile (default: 0.1 = 10th percentile)",
    )
    parser.add_argument(
        "--min-spike-duration",
        type=int,
        default=7,
        help="Minimum days for spike period (default: 7)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="percentile",
        choices=["percentile", "prominence", "growth_rate"],
        help="Detection method (default: percentile)",
    )
    parser.add_argument(
        "--growth-factor-threshold",
        type=float,
        default=1.5,
        help="Growth factor threshold for growth_rate method (default: 1.5 = 50%% growth)",
    )
    parser.add_argument(
        "--min-growth-duration",
        type=int,
        default=3,
        help="Minimum consecutive days of growth for growth_rate method (default: 3)",
    )
    parser.add_argument(
        "--min-cases-per-capita",
        type=float,
        default=1e-4,
        help="Minimum cases per person for growth_rate method (default: 1e-4 = 1 per 10K)",
    )
    parser.add_argument(
        "--growth-window",
        type=int,
        default=3,
        help="Window size for growth rate calculation in days (default: 3)",
    )
    parser.add_argument(
        "--include-non-baseline",
        action="store_true",
        help="Include non-baseline runs in spike detection",
    )

    args = parser.parse_args()

    try:
        spikes = detect_spike_periods_from_zarr(
            args.zarr_path,
            threshold_pct=args.spike_threshold,
            min_duration=args.min_spike_duration,
            method=args.method,
            baseline_filter=not args.include_non_baseline,
            growth_factor_threshold=args.growth_factor_threshold,
            min_growth_duration=args.min_growth_duration,
            min_cases_per_capita=args.min_cases_per_capita,
            growth_window=args.growth_window,
        )
        print_spike_summary(spikes)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        exit(1)
