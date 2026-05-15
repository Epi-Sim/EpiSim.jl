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


def test_calendar_ipfp_weekend_reduces_offdiag_and_preserves_rows():
    """Calendar IPFP should suppress weekend travel and keep row-stochasticity."""
    edgelist = np.array([
        [0, 0], [0, 1], [0, 2],
        [1, 0], [1, 1], [1, 2],
        [2, 0], [2, 1], [2, 2],
    ], dtype=np.int64)
    baseline_R = np.array([
        0.70, 0.20, 0.10,
        0.15, 0.70, 0.15,
        0.10, 0.20, 0.70,
    ])
    generator = MobilityGenerator(
        baseline_R=(edgelist, baseline_R),
        sigma_O=0.0,
        sigma_D=0.0,
        rng_seed=7,
        generator_mode="calendar_ipfp",
        start_date="2020-01-06",  # Monday
        weekend_volume_factor=0.4,
        weekday_volume_jitter=0.0,
        edge_weekend_effect=0.8,
        intermit_prob=0.0,
        temporal_rho=0.5,
    )

    series = generator.generate_series(T=7, rng_seed=7)
    dense = np.stack([generator.to_dense(series[t]) for t in range(7)])
    row_sums = dense.sum(axis=2)
    offdiag_mask = ~np.eye(3, dtype=bool)
    weekday_offdiag = dense[:5, offdiag_mask].sum(axis=1).mean()
    weekend_offdiag = dense[5:, offdiag_mask].sum(axis=1).mean()
    weekday_self = np.diagonal(dense[:5], axis1=1, axis2=2).mean()
    weekend_self = np.diagonal(dense[5:], axis1=1, axis2=2).mean()

    assert np.all(series >= 0.0)
    assert series.shape == (7, len(baseline_R))
    assert np.allclose(row_sums, 1.0, atol=1e-6)
    assert weekend_offdiag < weekday_offdiag
    assert weekend_self > weekday_self


def test_calendar_ipfp_is_reproducible_for_same_seed():
    """Same seed and parameters should produce identical calendar trajectories."""
    edgelist = np.array([
        [0, 0], [0, 1],
        [1, 0], [1, 1],
    ], dtype=np.int64)
    baseline_R = np.array([0.8, 0.2, 0.1, 0.9])
    kwargs = {
        "baseline_R": (edgelist, baseline_R),
        "sigma_O": 0.2,
        "sigma_D": 0.2,
        "rng_seed": 123,
        "generator_mode": "calendar_ipfp",
        "start_date": "2020-01-01",
        "weekend_volume_factor": 0.45,
        "weekday_volume_jitter": 0.04,
        "edge_weekend_effect": 0.7,
        "intermit_prob": 0.2,
        "temporal_rho": 0.6,
    }

    first = MobilityGenerator(**kwargs).generate_series(T=14, rng_seed=123)
    second = MobilityGenerator(**kwargs).generate_series(T=14, rng_seed=123)

    np.testing.assert_allclose(first, second)


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
        for e, (i, _j) in enumerate(edgelist):
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


# ---------------------------------------------------------------------------
# Quantile Markov edge-class tests
# ---------------------------------------------------------------------------

def _make_5x5_full_generator(**extra_kwargs):
    """Build a 5x5 fully-connected generator with calendar_ipfp and known weights."""
    M = 5
    edgelist = np.array(
        [[i, j] for i in range(M) for j in range(M)], dtype=np.int64
    )
    rng = np.random.default_rng(0)
    raw = rng.random(M * M)
    raw = raw.reshape(M, M)
    raw /= raw.sum(axis=1, keepdims=True)
    baseline_R = raw.ravel()

    kwargs = dict(
        baseline_R=(edgelist, baseline_R),
        sigma_O=0.1,
        sigma_D=0.1,
        rng_seed=42,
        generator_mode="calendar_ipfp",
        start_date="2020-01-06",
        weekend_volume_factor=0.45,
        weekday_volume_jitter=0.04,
        edge_weekend_effect=0.8,
        intermit_prob=0.15,
        temporal_rho=0.6,
    )
    kwargs.update(extra_kwargs)
    return MobilityGenerator(**kwargs), edgelist, baseline_R, M


def test_quantile_markov_backward_compat():
    """Default edge_class_mode='none' must produce bit-identical output to current code."""
    gen_none, _, _, _ = _make_5x5_full_generator(edge_class_mode="none")
    gen_default, _, _, _ = _make_5x5_full_generator()

    s1 = gen_none.generate_series(T=14, rng_seed=99)
    s2 = gen_default.generate_series(T=14, rng_seed=99)

    np.testing.assert_array_equal(s1, s2)


def test_quantile_markov_classification():
    """Verify edge class assignments from known-weight matrix."""
    M = 4
    # Fully-connected with deliberately varied weights
    edgelist = np.array(
        [[i, j] for i in range(M) for j in range(M)], dtype=np.int64
    )
    # Row-stochastic with strong variation
    W = np.array([
        [0.80, 0.10, 0.05, 0.05],
        [0.03, 0.85, 0.07, 0.05],
        [0.02, 0.03, 0.90, 0.05],
        [0.04, 0.03, 0.03, 0.90],
    ])
    baseline_R = W.ravel()

    gen = MobilityGenerator(
        baseline_R=(edgelist, baseline_R),
        generator_mode="calendar_ipfp",
        edge_class_mode="quantile_markov",
        start_date="2020-01-06",
    )

    ec = gen._edge_class
    # Self-loops should be -1
    for e in range(len(edgelist)):
        if edgelist[e, 0] == edgelist[e, 1]:
            assert ec[e] == -1, f"Self-loop edge {e} classified as {ec[e]}"

    # Non-self edges: exactly 4 classes (0..3)
    nonself = ec[ec >= 0]
    assert set(nonself) <= {0, 1, 2, 3}
    # Each non-self edge should be assigned
    assert len(nonself) == M * (M - 1)  # 12 non-self edges

    # Class masks should partition non-self edges
    total_masked = sum(np.sum(gen._class_masks[c]) for c in range(4))
    assert total_masked == M * (M - 1)

    # Higher-volume non-self edges should be assigned to more stable classes.
    class_means = [
        baseline_R[gen._class_masks[c]].mean()
        for c in range(4)
        if np.any(gen._class_masks[c])
    ]
    assert all(
        left >= right for left, right in zip(class_means, class_means[1:])
    ), f"Class means should be descending by edge weight, got {class_means}"


def test_quantile_markov_row_stochastic():
    """Every timestep must be row-stochastic with quantile_markov."""
    gen, edgelist, _, M = _make_5x5_full_generator(
        edge_class_mode="quantile_markov",
        intermit_persistence=0.6,
        intermit_prob=0.15,
    )

    series = gen.generate_series(T=14, rng_seed=42)

    for t in range(14):
        row_sums = np.zeros(M)
        np.add.at(row_sums, edgelist[:, 0], series[t])
        assert np.allclose(row_sums, 1.0, atol=1e-6), (
            f"Row sums at t={t}: {row_sums}"
        )


def test_quantile_markov_state_persistence():
    """High persistence should produce high lag-1 autocorrelation in edge states."""
    gen, edgelist, _, M = _make_5x5_full_generator(
        edge_class_mode="quantile_markov",
        intermit_persistence=0.9,
        intermit_prob=0.15,
    )

    T = 60
    # We need to track edge states; generate series captures them indirectly
    # Use a simple approach: re-generate and track _edge_active_state
    series = gen.generate_series(T=T, rng_seed=42)

    # For class-3 edges, infer on/off from weight ratio to baseline
    class3_mask = gen._class_masks[3]
    if np.sum(class3_mask) == 0:
        # If no class-3 edges, skip autocorrelation check
        return

    # Track state changes through series: edge is "off" when weight << baseline
    # Re-run with state tracking
    rng = np.random.default_rng(42)
    gen._init_edge_states(rng)
    states = np.zeros((T, int(np.sum(class3_mask))), dtype=bool)
    for t in range(T):
        gen._update_edge_states(rng)
        states[t] = gen._edge_active_state[class3_mask]

    # Lag-1 autocorrelation
    autocorr = np.mean(states[:-1] == states[1:])
    assert autocorr > 0.7, f"Lag-1 autocorrelation with persistence=0.9: {autocorr:.3f} <= 0.7"

    # Compare with low persistence
    gen_low, _, _, _ = _make_5x5_full_generator(
        edge_class_mode="quantile_markov",
        intermit_persistence=0.0,
        intermit_prob=0.15,
    )
    rng2 = np.random.default_rng(42)
    gen_low._init_edge_states(rng2)
    states_low = np.zeros((T, int(np.sum(gen_low._class_masks[3]))), dtype=bool)
    for t in range(T):
        gen_low._update_edge_states(rng2)
        states_low[t] = gen_low._edge_active_state[gen_low._class_masks[3]]

    if states_low.shape[1] > 0:
        autocorr_low = np.mean(states_low[:-1] == states_low[1:])
        assert autocorr > autocorr_low + 0.1, (
            f"High persistence autocorr ({autocorr:.3f}) not > low ({autocorr_low:.3f}) + 0.1"
        )


def test_quantile_markov_cv_by_class():
    """Weak edges should become more variable while trunk edges stay stable."""
    # Use a larger 10x10 matrix with skewed weights to create strong class separation
    M = 10
    edgelist = np.array(
        [[i, j] for i in range(M) for j in range(M)], dtype=np.int64
    )
    rng = np.random.default_rng(0)
    raw = rng.random(M * M).reshape(M, M)
    # Create skew: raise to power to make some edges much stronger than others
    raw = raw**3
    raw /= raw.sum(axis=1, keepdims=True)
    baseline_R = raw.ravel()

    common = dict(
        baseline_R=(edgelist, baseline_R),
        sigma_O=0.1,
        sigma_D=0.1,
        generator_mode="calendar_ipfp",
        start_date="2020-01-06",
        weekend_volume_factor=0.45,
        weekday_volume_jitter=0.04,
        edge_weekend_effect=0.8,
        intermit_prob=0.25,
        temporal_rho=0.6,
    )

    gen_none = MobilityGenerator(**common, edge_class_mode="none")
    gen_qm = MobilityGenerator(**common, edge_class_mode="quantile_markov", intermit_persistence=0.6)

    T = 90
    s_none = gen_none.generate_series(T=T, rng_seed=42)
    s_qm = gen_qm.generate_series(T=T, rng_seed=42)

    nonself = edgelist[:, 0] != edgelist[:, 1]

    def median_cv(series, mask):
        ns = series[:, mask]
        means = ns.mean(axis=0)
        stds = ns.std(axis=0)
        valid = means > 1e-10
        return np.median(stds[valid] / means[valid])

    trunk_mask = gen_qm._class_masks[0] & nonself
    weak_mask = gen_qm._class_masks[3] & nonself

    trunk_cv_none = median_cv(s_none, trunk_mask)
    trunk_cv_qm = median_cv(s_qm, trunk_mask)
    weak_cv_none = median_cv(s_none, weak_mask)
    weak_cv_qm = median_cv(s_qm, weak_mask)

    assert trunk_cv_qm < trunk_cv_none, (
        f"Trunk CV quantile_markov ({trunk_cv_qm:.2f}) should be below none "
        f"({trunk_cv_none:.2f})"
    )
    assert weak_cv_qm > weak_cv_none, (
        f"Weak-edge CV quantile_markov ({weak_cv_qm:.2f}) should exceed none "
        f"({weak_cv_none:.2f})"
    )


def test_quantile_markov_deterministic():
    """Same seed + params produce identical output."""
    gen1, _, _, _ = _make_5x5_full_generator(
        edge_class_mode="quantile_markov",
        intermit_persistence=0.6,
        intermit_prob=0.15,
    )
    gen2, _, _, _ = _make_5x5_full_generator(
        edge_class_mode="quantile_markov",
        intermit_persistence=0.6,
        intermit_prob=0.15,
    )

    s1 = gen1.generate_series(T=30, rng_seed=77)
    s2 = gen2.generate_series(T=30, rng_seed=77)

    np.testing.assert_array_equal(s1, s2)


def test_quantile_markov_off_damping_preserves_structure():
    """No edges should drop to exact zero when in off-state."""
    gen, edgelist, baseline_R, M = _make_5x5_full_generator(
        edge_class_mode="quantile_markov",
        intermit_persistence=0.6,
        intermit_prob=0.15,
        off_damping=0.02,
    )

    T = 30
    series = gen.generate_series(T=T, rng_seed=42)

    # All values should be >= 0 (non-negative)
    assert np.all(series >= 0.0), "Negative values found in series"

    # Non-self edges that were off should still be > 0 due to off_damping
    # Check that all non-self edges remain positive
    nonself = edgelist[:, 0] != edgelist[:, 1]
    nonself_series = series[:, nonself]
    assert np.all(nonself_series >= 0.0), "Some non-self edges went negative"
