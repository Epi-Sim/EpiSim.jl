#!/usr/bin/env python3
"""Compare real and synthetic mobility artifacts.

The script reports two separate views of mobility:
- absolute/count-like flows, preserving total-volume artifacts
- row-normalized routing probabilities, matching the simulator-facing layer
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

try:
    from episim_python.mobility import MobilityGenerator
except ImportError:  # pragma: no cover - supports running from repository root
    sys.path.append(str(Path(__file__).resolve().parent))
    from episim_python.mobility import MobilityGenerator


DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass
class MobilitySample:
    """In-memory mobility sample with date and region metadata."""

    name: str
    values: np.ndarray
    dates: pd.DatetimeIndex
    origins: list[str]
    targets: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_datetime_index(values: Any, n_dates: int) -> pd.DatetimeIndex:
    if values is None:
        return pd.date_range("2000-01-01", periods=n_dates, freq="D")
    dates = pd.to_datetime(values)
    if len(dates) != n_dates:
        raise ValueError(f"Expected {n_dates} dates, got {len(dates)}")
    return pd.DatetimeIndex(dates)


def _coord_values(ds: xr.Dataset, names: tuple[str, ...], size: int) -> list[str]:
    for name in names:
        if name in ds.coords:
            return [str(x) for x in ds.coords[name].values]
    return [str(i) for i in range(size)]


def _run_metadata(ds: xr.Dataset, run_idx: int) -> dict[str, Any]:
    metadata = {}
    for var_name in [
        "synthetic_mobility_generator",
        "synthetic_mobility_noise_sigma_O",
        "synthetic_mobility_noise_sigma_D",
        "synthetic_mobility_weekend_volume_factor",
        "synthetic_mobility_weekday_volume_jitter",
        "synthetic_mobility_edge_weekend_effect",
        "synthetic_mobility_intermit_prob",
        "synthetic_mobility_temporal_rho",
    ]:
        if var_name not in ds:
            continue
        value = ds[var_name].isel(run_id=run_idx).values
        metadata[var_name] = np.asarray(value).item()
    return metadata


def load_zarr_samples(path: str | Path, name: str, max_runs: int = 3) -> list[MobilitySample]:
    """Load mobility samples from a zarr store.

    Supported layouts:
    - mobility(date, origin, target)
    - mobility_time_varying(run_id, origin, target, date)
    - mobility_base(origin, target) + mobility_kappa0(run_id, date)
    """
    samples: list[MobilitySample] = []
    with xr.open_zarr(path, chunks=None) as ds:
        if "mobility" in ds:
            arr = ds["mobility"].transpose("date", ...).values.astype(np.float64)
            if arr.ndim != 3:
                raise ValueError("'mobility' must be 3D")
            dates = _as_datetime_index(ds.coords.get("date", None), arr.shape[0])
            origins = _coord_values(ds, ("origin",), arr.shape[1])
            targets = _coord_values(ds, ("target", "destination"), arr.shape[2])
            samples.append(MobilitySample(name, arr, dates, origins, targets))
            return samples

        if "mobility_time_varying" in ds:
            run_count = min(int(ds.sizes.get("run_id", 1)), max_runs)
            run_ids = (
                [str(x) for x in ds.coords["run_id"].values[:run_count]]
                if "run_id" in ds.coords
                else [str(i) for i in range(run_count)]
            )
            dates = _as_datetime_index(ds.coords.get("date", None), int(ds.sizes["date"]))
            origins = _coord_values(ds, ("origin",), int(ds.sizes["origin"]))
            targets = _coord_values(ds, ("target", "destination"), int(ds.sizes["target"]))
            for idx, run_id in enumerate(run_ids):
                arr = (
                    ds["mobility_time_varying"]
                    .isel(run_id=idx)
                    .transpose("date", "origin", "target")
                    .values.astype(np.float64)
                )
                samples.append(
                    MobilitySample(
                        f"{name}:{run_id}",
                        arr,
                        dates,
                        origins,
                        targets,
                        _run_metadata(ds, idx),
                    )
                )
            return samples

        if "mobility_base" in ds and "mobility_kappa0" in ds:
            base = ds["mobility_base"].values.astype(np.float64)
            run_count = min(int(ds.sizes.get("run_id", 1)), max_runs)
            run_ids = (
                [str(x) for x in ds.coords["run_id"].values[:run_count]]
                if "run_id" in ds.coords
                else [str(i) for i in range(run_count)]
            )
            dates = _as_datetime_index(ds.coords.get("date", None), int(ds.sizes["date"]))
            origins = _coord_values(ds, ("origin",), base.shape[0])
            targets = _coord_values(ds, ("target", "destination"), base.shape[1])
            for idx, run_id in enumerate(run_ids):
                kappa = ds["mobility_kappa0"].isel(run_id=idx).values.astype(np.float64)
                arr = base[np.newaxis, :, :] * (1.0 - kappa[:, np.newaxis, np.newaxis])
                samples.append(
                    MobilitySample(
                        f"{name}:{run_id}",
                        arr,
                        dates,
                        origins,
                        targets,
                        _run_metadata(ds, idx),
                    )
                )
            return samples

    raise ValueError(f"No supported mobility variables found in {path}")


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """Return row-normalized mobility with zero rows left as all-zero."""
    row_sums = values.sum(axis=2, keepdims=True)
    return np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)


def detect_flow_kind(values: np.ndarray, tolerance: float = 1e-4) -> dict[str, Any]:
    row_sums = values.sum(axis=2)
    positive = row_sums > 0
    if not np.any(positive):
        return {
            "flow_kind": "empty",
            "row_stochastic_fraction": 0.0,
            "row_sum_median": 0.0,
            "row_sum_min": 0.0,
            "row_sum_max": 0.0,
        }
    valid = np.abs(row_sums[positive] - 1.0) <= tolerance
    fraction = float(valid.mean())
    kind = "row_stochastic" if fraction >= 0.99 else "count_like"
    return {
        "flow_kind": kind,
        "row_stochastic_fraction": fraction,
        "row_sum_median": float(np.median(row_sums[positive])),
        "row_sum_min": float(row_sums[positive].min()),
        "row_sum_max": float(row_sums[positive].max()),
    }


def _nonself_mask(n_origins: int, n_targets: int) -> np.ndarray:
    mask = np.ones((n_origins, n_targets), dtype=bool)
    if n_origins == n_targets:
        np.fill_diagonal(mask, False)
    return mask


def _percentiles(prefix: str, values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_median": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p99": float("nan"),
        }
    return {
        f"{prefix}_median": float(np.nanmedian(finite)),
        f"{prefix}_p90": float(np.nanpercentile(finite, 90)),
        f"{prefix}_p99": float(np.nanpercentile(finite, 99)),
    }


def _weekend_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    return np.array([date.weekday() >= 5 for date in dates], dtype=bool)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return float("nan")
    return float(numerator / denominator)


def _vector_autocorrelation(values: np.ndarray, lag: int) -> float:
    if values.shape[0] <= lag:
        return float("nan")
    left = values[:-lag]
    right = values[lag:]
    left_centered = left - left.mean(axis=0)
    right_centered = right - right.mean(axis=0)
    denom = np.sqrt(
        np.sum(left_centered * left_centered, axis=0)
        * np.sum(right_centered * right_centered, axis=0)
    )
    corr = np.divide(
        np.sum(left_centered * right_centered, axis=0),
        denom,
        out=np.full(denom.shape, np.nan),
        where=denom > 1e-18,
    )
    return float(np.nanmedian(corr))


def _rankdata(values: np.ndarray) -> np.ndarray:
    sorter = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(sorter, dtype=np.float64)
    ranks[sorter] = np.arange(values.size, dtype=np.float64)
    return ranks


def _rank_stability(values: np.ndarray) -> float:
    if values.shape[0] < 2 or values.shape[1] < 2:
        return float("nan")
    correlations = []
    for t in range(values.shape[0] - 1):
        left = _rankdata(values[t])
        right = _rankdata(values[t + 1])
        left -= left.mean()
        right -= right.mean()
        denom = np.sqrt(np.sum(left * left) * np.sum(right * right))
        if denom > 0:
            correlations.append(float(np.sum(left * right) / denom))
    return float(np.nanmedian(correlations)) if correlations else float("nan")


def absolute_metrics(values: np.ndarray, dates: pd.DatetimeIndex) -> dict[str, Any]:
    """Compute count-like mobility metrics without row normalization."""
    nonself = _nonself_mask(values.shape[1], values.shape[2])
    weekend = _weekend_mask(dates)
    weekday = ~weekend
    offdiag = values[:, nonself]
    daily_total = offdiag.sum(axis=1)
    origin_totals = values.sum(axis=2)
    destination_totals = values.sum(axis=1)

    dow_means = {}
    dow_values = []
    for dow, name in enumerate(DOW_NAMES):
        selected = np.array([date.weekday() == dow for date in dates], dtype=bool)
        if np.any(selected):
            mean_value = float(daily_total[selected].mean())
            dow_values.append(mean_value)
            dow_means[f"absolute_dow_total_{name}"] = mean_value

    origin_mean = origin_totals.mean(axis=0)
    origin_std = origin_totals.std(axis=0)
    dest_mean = destination_totals.mean(axis=0)
    dest_std = destination_totals.std(axis=0)
    origin_cv = np.divide(
        origin_std,
        origin_mean,
        out=np.full_like(origin_std, np.nan),
        where=origin_mean > 1e-12,
    )
    destination_cv = np.divide(
        dest_std,
        dest_mean,
        out=np.full_like(dest_std, np.nan),
        where=dest_mean > 1e-12,
    )

    metrics = {
        "absolute_daily_offdiag_total_mean": float(daily_total.mean()),
        "absolute_daily_offdiag_total_std": float(daily_total.std()),
        "absolute_daily_offdiag_total_cv": _safe_ratio(
            float(daily_total.std()), float(daily_total.mean())
        ),
        "absolute_weekend_to_weekday_total_ratio": (
            _safe_ratio(float(daily_total[weekend].mean()), float(daily_total[weekday].mean()))
            if np.any(weekend) and np.any(weekday)
            else float("nan")
        ),
        "absolute_dow_total_range_ratio": (
            _safe_ratio(max(dow_values), min(dow_values)) if dow_values else float("nan")
        ),
    }
    metrics.update(dow_means)
    metrics.update(_percentiles("absolute_origin_marginal_cv", origin_cv))
    metrics.update(_percentiles("absolute_destination_marginal_cv", destination_cv))
    return metrics


def routing_metrics(values: np.ndarray, dates: pd.DatetimeIndex) -> dict[str, Any]:
    """Compute row-normalized routing metrics."""
    nonself = _nonself_mask(values.shape[1], values.shape[2])
    mean_matrix = values.mean(axis=0)
    active_edges = (mean_matrix > 1e-12) & nonself
    if not np.any(active_edges):
        return {"routing_active_nonself_edges": 0}

    edge_values = values[:, active_edges]
    edge_mean = edge_values.mean(axis=0)
    edge_std = edge_values.std(axis=0)
    edge_cv = np.divide(
        edge_std,
        edge_mean,
        out=np.full_like(edge_std, np.nan),
        where=edge_mean > 1e-12,
    )

    weekend = _weekend_mask(dates)
    weekday = ~weekend
    if np.any(weekend) and np.any(weekday):
        weekday_edge_mean = edge_values[weekday].mean(axis=0)
        weekend_edge_mean = edge_values[weekend].mean(axis=0)
        weekend_effect = np.abs(weekend_edge_mean - weekday_edge_mean) / np.maximum(
            weekday_edge_mean, 1e-12
        )
        weekend_ratio = _safe_ratio(
            float(edge_values[weekend].mean()), float(edge_values[weekday].mean())
        )
    else:
        weekend_effect = np.array([np.nan])
        weekend_ratio = float("nan")

    dow_edge_means = []
    for dow in range(7):
        selected = np.array([date.weekday() == dow for date in dates], dtype=bool)
        if np.any(selected):
            dow_edge_means.append(float(edge_values[selected].mean()))

    daily_offdiag_total = values[:, nonself].sum(axis=1)
    top_k = min(100, edge_values.shape[1])
    top_k_share = []
    for t in range(edge_values.shape[0]):
        total = float(edge_values[t].sum())
        if total > 0:
            top_k_share.append(float(np.partition(edge_values[t], -top_k)[-top_k:].sum() / total))

    metrics = {
        "routing_active_nonself_edges": int(active_edges.sum()),
        "routing_nonself_density": float(active_edges.sum() / nonself.sum()),
        "routing_zero_fraction": float((values[:, nonself] <= 1e-12).mean()),
        "routing_weekend_to_weekday_mean_ratio": weekend_ratio,
        "routing_dow_mean_range_ratio": (
            _safe_ratio(max(dow_edge_means), min(dow_edge_means))
            if dow_edge_means
            else float("nan")
        ),
        "routing_daily_offdiag_total_cv": _safe_ratio(
            float(daily_offdiag_total.std()), float(daily_offdiag_total.mean())
        ),
        "routing_lag1_corr_median": _vector_autocorrelation(edge_values, 1),
        "routing_lag7_corr_median": _vector_autocorrelation(edge_values, 7),
        "routing_edge_rank_stability_median": _rank_stability(edge_values),
        "routing_top100_edge_mass_share_mean": (
            float(np.mean(top_k_share)) if top_k_share else float("nan")
        ),
    }
    metrics.update(_percentiles("routing_edge_std", edge_std))
    metrics.update(_percentiles("routing_edge_cv", edge_cv))
    metrics.update(_percentiles("routing_edge_weekend_effect", weekend_effect))
    return metrics


def summarize_sample(sample: MobilitySample) -> dict[str, Any]:
    flow_info = detect_flow_kind(sample.values)
    normalized = normalize_rows(sample.values)
    summary = {
        "sample": sample.name,
        "n_dates": int(sample.values.shape[0]),
        "n_origins": int(sample.values.shape[1]),
        "n_targets": int(sample.values.shape[2]),
        **flow_info,
    }
    summary.update(sample.metadata)
    summary.update(absolute_metrics(sample.values, sample.dates))
    summary.update(routing_metrics(normalized, sample.dates))
    return summary


def generate_ipfp_samples(
    base_sample: MobilitySample, sigma_values: list[float], seed: int
) -> list[MobilitySample]:
    normalized = normalize_rows(base_sample.values)
    baseline = normalized.mean(axis=0)
    baseline = normalize_rows(baseline[np.newaxis, :, :])[0]
    edgelist = np.argwhere(baseline > 1e-12)
    weights = baseline[edgelist[:, 0], edgelist[:, 1]]
    samples = []
    for sigma in sigma_values:
        generator = MobilityGenerator(
            baseline_R=(edgelist, weights),
            sigma_O=sigma,
            sigma_D=sigma,
            rng_seed=seed,
        )
        series = generator.generate_series(T=base_sample.values.shape[0], rng_seed=seed)
        values = np.zeros_like(normalized)
        values[:, edgelist[:, 0], edgelist[:, 1]] = series
        samples.append(
            MobilitySample(
                name=f"ipfp_sigma_{sigma:g}",
                values=values,
                dates=base_sample.dates,
                origins=base_sample.origins,
                targets=base_sample.targets,
            )
        )
    return samples


def format_markdown(rows: list[dict[str, Any]]) -> str:
    columns = [
        "sample",
        "synthetic_mobility_generator",
        "flow_kind",
        "row_sum_median",
        "absolute_weekend_to_weekday_total_ratio",
        "absolute_daily_offdiag_total_cv",
        "routing_edge_std_median",
        "routing_edge_cv_median",
        "routing_edge_weekend_effect_median",
        "routing_zero_fraction",
        "routing_lag7_corr_median",
        "routing_edge_rank_stability_median",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append("nan" if np.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_plots(samples: list[MobilitySample], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for sample in samples:
        nonself = _nonself_mask(sample.values.shape[1], sample.values.shape[2])
        daily_total = sample.values[:, nonself].sum(axis=1)
        plt.plot(sample.dates, daily_total, marker="o", linewidth=1.5, label=sample.name)
    plt.title("Daily Total Off-Diagonal Mobility")
    plt.ylabel("flow")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "daily_total_offdiag.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    for sample in samples:
        nonself = _nonself_mask(sample.values.shape[1], sample.values.shape[2])
        daily_total = sample.values[:, nonself].sum(axis=1)
        means = []
        labels = []
        for dow, name in enumerate(DOW_NAMES):
            selected = np.array([date.weekday() == dow for date in sample.dates], dtype=bool)
            if np.any(selected):
                labels.append(name)
                means.append(float(daily_total[selected].mean()))
        plt.plot(labels, means, marker="o", linewidth=1.5, label=sample.name)
    plt.title("Mean Off-Diagonal Mobility by Day of Week")
    plt.ylabel("flow")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "day_of_week_means.png", dpi=150)
    plt.close()

    for metric_name, filename, xlabel in [
        ("std", "routing_edge_std_distribution.png", "routing edge std"),
        ("cv", "routing_edge_cv_distribution.png", "routing edge CV"),
        ("weekend", "routing_weekend_effect_distribution.png", "weekend effect"),
    ]:
        plt.figure(figsize=(9, 5))
        for sample in samples:
            normalized = normalize_rows(sample.values)
            nonself = _nonself_mask(normalized.shape[1], normalized.shape[2])
            active = (normalized.mean(axis=0) > 1e-12) & nonself
            if not np.any(active):
                continue
            vals = normalized[:, active]
            if metric_name == "std":
                data = vals.std(axis=0)
            elif metric_name == "cv":
                mean = vals.mean(axis=0)
                data = np.divide(
                    vals.std(axis=0),
                    mean,
                    out=np.full(mean.shape, np.nan),
                    where=mean > 1e-12,
                )
            else:
                weekend = _weekend_mask(sample.dates)
                weekday = ~weekend
                if not (np.any(weekend) and np.any(weekday)):
                    continue
                wd = vals[weekday].mean(axis=0)
                we = vals[weekend].mean(axis=0)
                data = np.abs(we - wd) / np.maximum(wd, 1e-12)
            data = data[np.isfinite(data)]
            if data.size:
                plt.hist(data, bins=80, alpha=0.35, density=True, label=sample.name)
        plt.title(xlabel)
        plt.xlabel(xlabel)
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()


def parse_sigma_values(raw: str | None) -> list[float]:
    if raw is None or raw.strip() == "":
        return []
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare real and synthetic mobility artifacts."
    )
    parser.add_argument("--real-zarr", required=True, help="Real mobility zarr path.")
    parser.add_argument(
        "--synthetic-zarr",
        help="Optional synthetic zarr path with mobility_time_varying or factorized mobility.",
    )
    parser.add_argument(
        "--sigma-values",
        default="",
        help="Comma-separated IPFP sigma values to generate from real topology.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Report format printed to stdout.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional directory for plots. No files are written unless this is set.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=3,
        help="Maximum synthetic runs to load from --synthetic-zarr.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Seed for IPFP samples.")
    args = parser.parse_args()

    samples = load_zarr_samples(args.real_zarr, "real", max_runs=1)
    samples.extend(generate_ipfp_samples(samples[0], parse_sigma_values(args.sigma_values), args.seed))
    if args.synthetic_zarr:
        samples.extend(load_zarr_samples(args.synthetic_zarr, "synthetic", args.max_runs))

    summaries = [summarize_sample(sample) for sample in samples]
    if args.output_dir:
        write_plots(samples, Path(args.output_dir))

    if args.format == "json":
        print(json.dumps(summaries, indent=2, allow_nan=True))
    else:
        print(format_markdown(summaries))


if __name__ == "__main__":
    main()
