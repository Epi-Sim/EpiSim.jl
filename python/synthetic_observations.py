import numpy as np
from scipy import stats

DEFAULT_REPORTED_CASES_CONFIG = {
    "min_rate": 0.05,
    "max_rate": 0.6,
    "inflection_day": 30,
    "slope": 0.2,
}

DEFAULT_WASTEWATER_CONFIG = {
    "gamma_shape": 2.5,
    "gamma_scale": 2.0,
    "noise_sigma": 0.5,
    "kernel_quantile": 0.999,
    "sensitivity_scale": 1.0,
    "limit_of_detection": 0.0,
    "transport_loss": 0.0,  # Signal loss during transport (decay)
    "lod_probabilistic": False,  # Use logistic probability for detection instead of hard cutoff
    "lod_slope": 2.0,  # Steepness of the probabilistic detection curve
}

DEFAULT_HOSP_REPORT_CONFIG = {
    "report_rate": 0.85,  # 85% of hospitalizations reported
    "delay_mean": 3,  # Mean reporting delay in days
    "delay_std": 1,  # Std dev of reporting delay
}

DEFAULT_DEATHS_REPORT_CONFIG = {
    "report_rate": 0.90,  # 90% of deaths reported
    "delay_mean": 7,  # Mean reporting delay in days (death certificates take time)
    "delay_std": 2,  # Std dev of reporting delay
}


def _normalize_infections(infections: np.ndarray) -> np.ndarray:
    infections = np.asarray(infections, dtype=float)
    infections = np.clip(infections, 0, None)
    return np.rint(infections).astype(int)


def generate_reported_cases(infections: np.ndarray, config=None, rng=None):
    """
    Derive reported cases using a logistic ascertainment ramp with binomial noise.

    Args:
        infections: Array of true daily infections (time, location) or (time,).
        config: Dict with min_rate, max_rate, inflection_day, slope.
        rng: Optional NumPy random generator.

    Returns:
        reported_cases: Integer array of reported cases.
        ascertainment_rate: Float array of daily rates (time,).
    """
    cfg = {**DEFAULT_REPORTED_CASES_CONFIG, **(config or {})}
    rng = rng or np.random.default_rng()

    infections_int = _normalize_infections(infections)
    t_indices = np.arange(infections_int.shape[0])

    k = cfg["slope"]
    mid = cfg["inflection_day"]
    p_min = cfg["min_rate"]
    p_max = cfg["max_rate"]

    p_t = p_min + (p_max - p_min) / (1.0 + np.exp(-k * (t_indices - mid)))

    if infections_int.ndim == 2:
        p_matrix = p_t[:, None]
    else:
        p_matrix = p_t

    reported = rng.binomial(infections_int, p_matrix)
    return reported, p_t


def generate_reported_with_delay(
    counts: np.ndarray,
    report_rate: float,
    delay_mean: int = 0,
    delay_std: int = 0,
    rng=None,
) -> np.ndarray:
    """
    Apply reporting rate and stochastic delay to count data (hospitalizations/deaths).

    Model:
    1. Each count is reported with probability report_rate (binomial thinning)
    2. Reported counts are shifted forward by a stochastic delay (truncated normal)
       - delay >= 0 (no negative delays)
       - Independent across locations

    Args:
        counts: Array of true daily counts (time, location) or (time,).
        report_rate: Fraction of events reported (0.0 to 1.0).
        delay_mean: Mean reporting delay in days.
        delay_std: Std dev of reporting delay (0 = deterministic delay).
        rng: Optional NumPy random generator.

    Returns:
        reported_counts: Integer array with reporting noise applied.
    """
    rng = rng or np.random.default_rng()
    counts = np.asarray(counts, dtype=float)

    # Apply reporting rate (binomial thinning)
    reported = rng.binomial(np.clip(counts, 0, None).astype(int), report_rate)

    # Apply stochastic delay if specified
    if delay_mean > 0 or delay_std > 0:
        if reported.ndim == 1:
            # Single location
            n_locations = 1
            reported = reported[:, None]
            squeeze_output = True
        else:
            n_locations = reported.shape[1]
            squeeze_output = False

        n_time = reported.shape[0]

        # Sample delay per location (truncated normal, >= 0)
        if delay_std > 0:
            delays = rng.normal(delay_mean, delay_std, size=n_locations)
            delays = np.maximum(0, delays).astype(int)
        else:
            delays = np.full(n_locations, max(0, delay_mean), dtype=int)

        # Apply delay: shift each location's time series forward
        reported_delayed = np.zeros_like(reported)
        for loc in range(n_locations):
            delay = delays[loc]
            if delay > 0:
                # Shift forward: original[t] -> delayed[t + delay]
                reported_delayed[delay:, loc] = reported[:-delay, loc] if delay < n_time else np.zeros(n_time - delay)
            else:
                reported_delayed[:, loc] = reported[:, loc]

        if squeeze_output:
            return reported_delayed[:, 0].astype(int)
        return reported_delayed.astype(int)

    return reported


def _build_shedding_kernel(shape, scale, quantile):
    max_len = int(stats.gamma.ppf(quantile, a=shape, scale=scale))
    max_len = max(1, max_len)
    t_kernel = np.arange(max_len)
    kernel = stats.gamma.pdf(t_kernel, a=shape, scale=scale)
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel = kernel / kernel_sum
    return kernel


def generate_wastewater(infections: np.ndarray, config=None, rng=None):
    """
    Convolve infections with a Gamma shedding curve and apply log-normal noise.

    Args:
        infections: Array of true daily infections (time, location) or (time,).
        config: Dict with gamma_shape, gamma_scale, noise_sigma, kernel_quantile.
        rng: Optional NumPy random generator.

    Returns:
        observed_wastewater: Float array matching infections shape.
    """
    cfg = {**DEFAULT_WASTEWATER_CONFIG, **(config or {})}
    rng = rng or np.random.default_rng()

    infections = np.asarray(infections, dtype=float)
    if infections.ndim == 1:
        infections = infections[:, None]
        squeeze_output = True
    else:
        squeeze_output = False

    kernel = _build_shedding_kernel(
        cfg["gamma_shape"], cfg["gamma_scale"], cfg["kernel_quantile"]
    )

    n_time, n_locations = infections.shape
    ww_signal = np.zeros_like(infections, dtype=float)

    for loc in range(n_locations):
        conv = np.convolve(infections[:, loc], kernel, mode="full")
        # Apply Sensitivity Scale
        ww_signal[:, loc] = conv[:n_time] * cfg["sensitivity_scale"]

    # Apply Transport Loss (Physical Decay)
    # This subtracts a fixed amount of signal representing degradation in the sewer system.
    # If the signal is weak, it decays to zero before reaching the sampler.
    transport_loss = cfg["transport_loss"]
    if transport_loss > 0.0:
        ww_signal = np.maximum(0.0, ww_signal - transport_loss)

    epsilon = 1e-8
    noise = rng.normal(0.0, cfg["noise_sigma"], size=ww_signal.shape)
    ww_observed = (ww_signal + epsilon) * np.exp(noise)

    # Apply Limit of Detection (LoD)
    lod = cfg["limit_of_detection"]

    if cfg["lod_probabilistic"] and lod > 0.0:
        # Logistic Dropout
        k = cfg.get("lod_slope", 2.0)
        detection_prob = 1.0 / (1.0 + np.exp(-k * (ww_observed - lod)))
        is_detected = rng.random(size=ww_observed.shape) < detection_prob
        ww_observed[~is_detected] = 0.0

    elif lod > 0.0:
        # Hard Cutoff
        ww_observed[ww_observed < lod] = 0.0

    if squeeze_output:
        return ww_observed[:, 0]
    return ww_observed


def _compute_monitoring_start_mask(
    infections_stratified: np.ndarray,
    threshold: float = 0.0,
    delay_days: int = 0,
    delay_std: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Compute boolean mask indicating when monitoring should start for each EDAR.

    For each EDAR independently:
    1. Find first day where cumulative infections >= threshold
    2. Add stochastic delay: delay_days ± delay_std (truncated at 0)
    3. Set all days before that date to False (not monitoring)

    Args:
        infections_stratified: Array of shape (Time, EDARs, AgeGroups) with daily infections
        threshold: Cumulative infections per EDAR to trigger monitoring (0=disabled)
        delay_days: Mean delay in days after threshold before monitoring starts
        delay_std: Std dev of delay (stochastic variation per EDAR, prevents overfitting)
        rng: Random number generator for stochastic delay

    Returns:
        mask: (Time, EDARs) bool array, True = monitoring active

    Raises:
        ValueError: If threshold is negative
    """
    if threshold < 0:
        raise ValueError(f"Threshold must be non-negative, got {threshold}")

    if threshold == 0:
        # Feature disabled: monitoring active from day 0
        n_time, n_edars = infections_stratified.shape[:2]
        return np.ones((n_time, n_edars), dtype=bool)

    rng = rng or np.random.default_rng()

    n_time, n_edars = infections_stratified.shape[:2]

    # Sum over age groups to get total infections per EDAR per day
    # infections_stratified shape: (Time, EDARs, AgeGroups)
    infections_total = infections_stratified.sum(axis=2)  # (Time, EDARs)

    # Compute cumulative infections for each EDAR
    cumulative_inf = np.cumsum(infections_total, axis=0)  # (Time, EDARs)

    # Find first day where each EDAR reaches threshold
    # start_day[edar] = index of first day with cumulative >= threshold, or n_time if never reached
    start_day = np.full(n_edars, n_time, dtype=int)
    for edar_idx in range(n_edars):
        threshold_reached = cumulative_inf[:, edar_idx] >= threshold
        if np.any(threshold_reached):
            start_day[edar_idx] = np.argmax(threshold_reached)

    # Add stochastic delay per EDAR
    if delay_std > 0:
        # Random delay: delay_days ± delay_std, truncated at 0
        delays = rng.normal(delay_days, delay_std, size=n_edars)
        delays = np.maximum(0, delays).astype(int)
    else:
        delays = np.full(n_edars, delay_days, dtype=int)

    start_day_with_delay = start_day + delays

    # Build monitoring mask
    monitoring_mask = np.zeros((n_time, n_edars), dtype=bool)
    for edar_idx in range(n_edars):
        start = min(start_day_with_delay[edar_idx], n_time)
        monitoring_mask[start:, edar_idx] = True

    return monitoring_mask


def generate_wastewater_stratified(
    infections_by_age: np.ndarray,
    population: np.ndarray = None,
    age_weights: np.ndarray = None,
    config=None,
    rng=None,
    monitoring_mask: np.ndarray | None = None,
):
    """
    Generate wastewater signal using age-stratified shedding kinetics.

    Physical Model (see CONTEXT_SYNTHETIC_GEN.md):
        Concentration = Σ(Infections_g × Shedding_g) / (Population × FlowPerCapita)

    This formula models the physical reality of wastewater surveillance:
        - Σ(Infections_g × Shedding_g): Total viral load shed by infected individuals
        - Population × FlowPerCapita: Total wastewater flow (dilution factor)
        - Division by population: Models DILUTION, not per-capita normalization

    Key insight: Population division models DILUTION physics.
        - More population = more wastewater flow = lower concentration
        - Same infections in village (5k) vs metropolis (500k) = vastly different signals
        - Example: 30 cases detectable in village (5k) but invisible in metropolis (500k)

    Note: FlowPerCapita is absorbed into sensitivity_scale as a simplifying assumption.
        - Default sensitivity_scale = 1.0 implies normalized per-capita flow
        - Real-world factors (rainfall, sewer hydraulics) are not explicitly modeled
        - sensitivity_scale can be calibrated to match real-world datasets (LoD = 375 Copies/L)

    Literature Consensus:
    - Children (<18): High shedding probability, Long duration (tail), Flatter gamma.
    - Adults (18-65): Moderate shedding, Standard duration (acute phase), Sharper peak.
    - Elderly (>65): Moderate shedding, Standard duration.

    Args:
        infections_by_age: Array of shape (Time, Location, AgeGroups) or (Time, AgeGroups) if 1 loc.
        population: Array of population counts (Location,) or scalar.
        age_weights: Array of population fractions [Child, Adult, Elderly] for each location.
                     Shape (Location, 3). If None, assumes uniform distribution.
        config: Dict overriding defaults.
        rng: NumPy generator.
        monitoring_mask: Optional (Time, Location) bool array. When False, output is set to NaN
                         (monitoring not yet started). Useful for modeling delayed surveillance
                         deployment based on infection thresholds.

    Returns:
        observed_wastewater: Float array (Time, Location) in Copies/L.
    """
    cfg = {**DEFAULT_WASTEWATER_CONFIG, **(config or {})}
    rng = rng or np.random.default_rng()

    # Standardize Input Shapes
    # infections: (T, M, G)
    if infections_by_age.ndim == 2:  # (Time, Groups) -> Single Location
        infections_by_age = infections_by_age[:, None, :]
        squeeze_output = True
    else:
        squeeze_output = False

    n_time, n_locations, n_groups = infections_by_age.shape

    # Handle Population (Dilution Denominator)
    if population is None:
        population = np.ones(n_locations, dtype=float)
    else:
        population = np.asarray(population, dtype=float)
        if population.ndim == 0:
            population = np.full(n_locations, population)

    # Handle Age Weights (for normalization if needed, currently implicit in infections)
    # We assume 'infections_by_age' are absolute counts, so we sum their contributions.

    # Define Age-Specific Kernels (G=3 assumption: 0=Young, 1=Mature, 2=Old)
    # Young: Longer tail (shape=1.5, scale=10.0 => mean=15d, long skewed)
    # Mature/Old: Sharper acute phase (shape=2.5, scale=4.0 => mean=10d)
    kernels = [
        _build_shedding_kernel(1.5, 10.0, 0.999),  # Young
        _build_shedding_kernel(2.5, 4.0, 0.999),  # Mature
        _build_shedding_kernel(2.5, 4.0, 0.999),  # Old
    ]

    # Shedding Intensities (Relative Load)
    # Can be tuned. Currently assuming baseline equality but differences in kinetics.
    # If children shed less peak load but longer, scale < 1.0.
    load_scales = [1.0, 1.0, 1.0]

    ww_signal_total = np.zeros((n_time, n_locations), dtype=float)

    for g in range(n_groups):
        kernel = kernels[min(g, len(kernels) - 1)]
        scale = load_scales[min(g, len(load_scales) - 1)]

        for loc in range(n_locations):
            # Convolve this group's infections with their specific kernel
            inf_series = infections_by_age[:, loc, g]
            conv = np.convolve(inf_series, kernel, mode="full")[:n_time]

            # Add to total load (weighted by shedding intensity)
            ww_signal_total[:, loc] += conv * scale

    # Apply Sensitivity Scale & Dilution
    # Signal = (Total Load * Sensitivity) / Population
    # Use safe division to handle zero/NaN populations (result is NaN for those locations)
    for loc in range(n_locations):
        pop = population[loc]
        if pop > 0:
            ww_signal_total[:, loc] = (
                ww_signal_total[:, loc] * cfg["sensitivity_scale"]
            ) / pop
        else:
            # Zero or negative population: set signal to NaN
            ww_signal_total[:, loc] = np.nan

    # Apply Transport Loss (Physical Decay)
    transport_loss = cfg["transport_loss"]
    if transport_loss > 0.0:
        ww_signal_total = np.maximum(0.0, ww_signal_total - transport_loss)

    # Apply Noise
    epsilon = 1e-8
    noise = rng.normal(0.0, cfg["noise_sigma"], size=ww_signal_total.shape)
    ww_observed = (ww_signal_total + epsilon) * np.exp(noise)

    # Apply Limit of Detection
    lod = cfg["limit_of_detection"]
    if cfg["lod_probabilistic"] and lod > 0.0:
        k = cfg.get("lod_slope", 2.0)
        detection_prob = 1.0 / (1.0 + np.exp(-k * (ww_observed - lod)))
        is_detected = rng.random(size=ww_observed.shape) < detection_prob
        ww_observed[~is_detected] = 0.0
    elif lod > 0.0:
        ww_observed[ww_observed < lod] = 0.0

    # Apply monitoring mask: set pre-threshold values to NaN
    if monitoring_mask is not None:
        # monitoring_mask shape: (Time, Location)
        # ww_observed shape: (Time, Location) or (Time,) for single location
        if monitoring_mask.shape == ww_observed.shape:
            ww_observed[~monitoring_mask] = np.nan
        elif squeeze_output and monitoring_mask.shape[1] == 1:
            # Single location case: monitoring_mask is (Time, 1), ww_observed is (Time,)
            ww_observed[~monitoring_mask[:, 0]] = np.nan

    if squeeze_output:
        return ww_observed[:, 0]
    return ww_observed
