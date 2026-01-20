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


def generate_wastewater_stratified(
    infections_by_age: np.ndarray,
    population: np.ndarray = None,
    age_weights: np.ndarray = None,
    config=None,
    rng=None,
):
    """
    Generate wastewater signal using age-stratified shedding kinetics.

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
    for loc in range(n_locations):
        ww_signal_total[:, loc] = (
            ww_signal_total[:, loc] * cfg["sensitivity_scale"]
        ) / population[loc]

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

    if squeeze_output:
        return ww_observed[:, 0]
    return ww_observed
