import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compare_mobility_artifacts import (
    MobilitySample,
    absolute_metrics,
    detect_flow_kind,
    normalize_rows,
    routing_metrics,
    summarize_sample,
)


def test_normalize_rows_keeps_zero_rows_zero():
    values = np.array(
        [
            [[2.0, 2.0], [0.0, 0.0]],
            [[1.0, 3.0], [4.0, 0.0]],
        ]
    )

    normalized = normalize_rows(values)

    np.testing.assert_allclose(normalized[0, 0], [0.5, 0.5])
    np.testing.assert_allclose(normalized[0, 1], [0.0, 0.0])
    np.testing.assert_allclose(normalized[1, 0], [0.25, 0.75])
    np.testing.assert_allclose(normalized[1, 1], [1.0, 0.0])


def test_detect_flow_kind_separates_count_like_and_row_stochastic():
    count_like = np.array([[[10.0, 5.0], [1.0, 8.0]]])
    stochastic = np.array([[[0.8, 0.2], [0.1, 0.9]]])

    assert detect_flow_kind(count_like)["flow_kind"] == "count_like"
    assert detect_flow_kind(stochastic)["flow_kind"] == "row_stochastic"


def test_absolute_metrics_capture_weekend_drop():
    dates = pd.date_range("2020-01-06", periods=7, freq="D")  # Monday start
    values = np.zeros((7, 2, 2), dtype=float)
    for t, date in enumerate(dates):
        cross_flow = 10.0 if date.weekday() < 5 else 2.0
        values[t] = [[100.0, cross_flow], [cross_flow, 80.0]]

    metrics = absolute_metrics(values, dates)

    assert metrics["absolute_weekend_to_weekday_total_ratio"] == 0.2
    assert metrics["absolute_dow_total_range_ratio"] == 5.0


def test_routing_metrics_capture_weekend_edge_effect():
    dates = pd.date_range("2020-01-06", periods=7, freq="D")  # Monday start
    values = np.zeros((7, 2, 2), dtype=float)
    for t, date in enumerate(dates):
        if date.weekday() < 5:
            values[t] = [[0.8, 0.2], [0.1, 0.9]]
        else:
            values[t] = [[0.95, 0.05], [0.02, 0.98]]

    metrics = routing_metrics(values, dates)

    assert metrics["routing_active_nonself_edges"] == 2
    assert metrics["routing_weekend_to_weekday_mean_ratio"] < 0.3
    assert metrics["routing_edge_weekend_effect_median"] > 0.7


def test_summarize_sample_reports_both_layers():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    values = np.array(
        [
            [[5.0, 1.0], [2.0, 4.0]],
            [[5.0, 2.0], [1.0, 4.0]],
            [[5.0, 0.5], [3.0, 4.0]],
        ]
    )
    sample = MobilitySample(
        name="toy",
        values=values,
        dates=dates,
        origins=["a", "b"],
        targets=["a", "b"],
    )

    summary = summarize_sample(sample)

    assert summary["sample"] == "toy"
    assert summary["flow_kind"] == "count_like"
    assert "absolute_daily_offdiag_total_mean" in summary
    assert "routing_edge_std_median" in summary
