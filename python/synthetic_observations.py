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
        ww_signal[:, loc] = conv[:n_time]

    epsilon = 1e-8
    noise = rng.normal(0.0, cfg["noise_sigma"], size=ww_signal.shape)
    ww_observed = (ww_signal + epsilon) * np.exp(noise)

    if squeeze_output:
        return ww_observed[:, 0]
    return ww_observed
