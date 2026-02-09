"""
Benchmark mobility generation performance.

Tests different approaches to identify bottlenecks.
"""

import time
import numpy as np
import pandas as pd

from episim_python.mobility import MobilityGenerator, load_baseline_mobility


def benchmark_ipfp_iterations(mobility_csv, metapop_csv, T=180, n_repeats=10):
    """Benchmark the IPFP algorithm with different iteration counts."""
    print("=" * 60)
    print("IPFP Iteration Benchmark")
    print("=" * 60)

    # Load baseline mobility
    edgelist, R_baseline, M = load_baseline_mobility(mobility_csv, metapop_csv)
    E = len(R_baseline)

    print(f"Mobility stats: M={M}, E={E}, T={T}")

    # Test with different iteration counts
    for max_iter in [5, 10, 20, 50]:
        times = []
        for _ in range(n_repeats):
            generator = MobilityGenerator(
                baseline_R=(edgelist, R_baseline),
                sigma_O=0.1,
                sigma_D=0.1,
                rng_seed=int(time.time())
            )

            # Monkey-patch the max_iter
            import episim_python.mobility as mob_module
            original_ipfp = mob_module.ipfp_sparse

            def ipfp_with_iter(edgelist, B, O_target, D_target, max_iter=max_iter, tol=1e-6):
                return original_ipfp(edgelist, B, O_target, D_target, max_iter=max_iter, tol=tol)

            mob_module.ipfp_sparse = ipfp_with_iter

            start = time.perf_counter()
            R_series = generator.generate_series(T=T)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            mob_module.ipfp_sparse = original_ipfp

        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"max_iter={max_iter:3d}: {avg_time:.3f}s ± {std_time:.3f}s per {T} timesteps")


def benchmark_sigma_values(mobility_csv, metapop_csv, T=180):
    """Benchmark with different sigma values."""
    print("\n" + "=" * 60)
    print("Sigma Value Benchmark")
    print("=" * 60)

    edgelist, R_baseline, M = load_baseline_mobility(mobility_csv, metapop_csv)
    E = len(R_baseline)

    print(f"Mobility stats: M={M}, E={E}, T={T}")

    # Test with different sigma values
    test_cases = [
        (0.0, 0.0, "No noise (static)"),
        (0.05, 0.05, "Low noise"),
        (0.1, 0.1, "Medium noise"),
        (0.2, 0.2, "High noise"),
        (0.1, 0.0, "O-noise only"),
        (0.0, 0.1, "D-noise only"),
    ]

    for sigma_O, sigma_D, desc in test_cases:
        generator = MobilityGenerator(
            baseline_R=(edgelist, R_baseline),
            sigma_O=sigma_O,
            sigma_D=sigma_D,
            rng_seed=42
        )

        start = time.perf_counter()
        R_series = generator.generate_series(T=T)
        elapsed = time.perf_counter() - start

        print(f"{desc:20s} (O={sigma_O:.2f}, D={sigma_D:.2f}): {elapsed:.3f}s for {T} timesteps")


def benchmark_single_timestep(mobility_csv, metapop_csv, n_repeats=100):
    """Benchmark a single timestep generation."""
    print("\n" + "=" * 60)
    print("Single Timestep Benchmark")
    print("=" * 60)

    edgelist, R_baseline, M = load_baseline_mobility(mobility_csv, metapop_csv)
    E = len(R_baseline)

    print(f"Mobility stats: M={M}, E={E}")

    generator = MobilityGenerator(
        baseline_R=(edgelist, R_baseline),
        sigma_O=0.1,
        sigma_D=0.1,
        rng_seed=42
    )

    # Warm up
    for _ in range(10):
        _ = generator.generate_R_t(t=0)

    # Benchmark
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        _ = generator.generate_R_t(t=0)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    p50 = np.percentile(times, 50) * 1000
    p95 = np.percentile(times, 95) * 1000
    p99 = np.percentile(times, 99) * 1000

    print(f"Single timestep (n={n_repeats}):")
    print(f"  Mean: {avg_time:.2f}ms")
    print(f"  Std:  {std_time:.2f}ms")
    print(f"  P50:  {p50:.2f}ms")
    print(f"  P95:  {p95:.2f}ms")
    print(f"  P99:  {p99:.2f}ms")


def profile_components(mobility_csv, metapop_csv):
    """Profile individual components of the generation."""
    print("\n" + "=" * 60)
    print("Component Profiling")
    print("=" * 60)

    edgelist, R_baseline, M = load_baseline_mobility(mobility_csv, metapop_csv)
    E = len(R_baseline)

    print(f"Mobility stats: M={M}, E={E}")

    generator = MobilityGenerator(
        baseline_R=(edgelist, R_baseline),
        sigma_O=0.1,
        sigma_D=0.1,
        rng_seed=42
    )

    # Profile marginals computation
    times = []
    for _ in range(100):
        start = time.perf_counter()
        O, D = generator._compute_marginals(R_baseline)
        times.append(time.perf_counter() - start)
    print(f"Compute marginals: {np.mean(times)*1000:.2f}ms")

    # Profile noisy marginals generation
    times = []
    for _ in range(100):
        start = time.perf_counter()
        O_t, D_t = generator._generate_noisy_marginals(t=0)
        times.append(time.perf_counter() - start)
    print(f"Generate noisy marginals: {np.mean(times)*1000:.2f}ms")

    # Profile IPFP
    O_t, D_t = generator._generate_noisy_marginals(t=0)
    times = []
    for _ in range(100):
        start = time.perf_counter()
        R_new = generator._ipfp(R_baseline, O_t, D_t)
        times.append(time.perf_counter() - start)
    print(f"IPFP (20 iterations): {np.mean(times)*1000:.2f}ms")

    # Profile full timestep
    times = []
    for _ in range(100):
        start = time.perf_counter()
        R_t = generator.generate_R_t(t=0)
        times.append(time.perf_counter() - start)
    print(f"Full timestep: {np.mean(times)*1000:.2f}ms")


def benchmark_scales():
    """Benchmark with different graph sizes."""
    print("\n" + "=" * 60)
    print("Scale Benchmark")
    print("=" * 60)

    sizes = [
        (10, 50, "Small (10x10, 50 edges)"),
        (50, 500, "Medium (50x50, 500 edges)"),
        (100, 2000, "Large (100x100, 2000 edges)"),
        (2850, 50000, "Catalonia-like"),
    ]

    T = 180

    for M, E, desc in sizes:
        # Create synthetic mobility
        np.random.seed(42)
        edgelist = np.random.randint(0, M, size=(E, 2), dtype=np.int64)
        # Ensure no self-loops for this test
        mask = edgelist[:, 0] != edgelist[:, 1]
        edgelist = edgelist[mask]
        if len(edgelist) < E:
            # Add more edges if needed
            while len(edgelist) < E:
                new_edges = np.random.randint(0, M, size=(E - len(edgelist), 2), dtype=np.int64)
                mask = new_edges[:, 0] != new_edges[:, 1]
                new_edges = new_edges[mask]
                edgelist = np.vstack([edgelist, new_edges])
            edgelist = edgelist[:E]

        # Create row-stochastic weights
        R_baseline = np.random.rand(E)
        for i in range(M):
            mask = edgelist[:, 0] == i
            if np.sum(mask) > 0:
                row_sum = np.sum(R_baseline[mask])
                if row_sum > 0:
                    R_baseline[mask] /= row_sum

        generator = MobilityGenerator(
            baseline_R=(edgelist, R_baseline),
            sigma_O=0.1,
            sigma_D=0.1,
            rng_seed=42
        )

        start = time.perf_counter()
        try:
            R_series = generator.generate_series(T=T)
            elapsed = time.perf_counter() - start
            print(f"{desc:25s}: {elapsed:.3f}s for {T} timesteps ({elapsed/T*1000:.2f}ms per timestep)")
        except Exception as e:
            print(f"{desc:25s}: ERROR - {e}")


if __name__ == "__main__":
    import sys

    # Use Catalonia data by default
    mobility_csv = "../models/catalonia/data/Mobility_Network.csv"
    metapop_csv = "../models/catalonia/data/Metapopulation_data.csv"

    if len(sys.argv) > 2:
        mobility_csv = sys.argv[1]
        metapop_csv = sys.argv[2]

    print(f"Using mobility: {mobility_csv}")
    print(f"Using metapop: {metapop_csv}")

    # Run benchmarks
    benchmark_sigma_values(mobility_csv, metapop_csv, T=50)
    benchmark_single_timestep(mobility_csv, metapop_csv, n_repeats=50)
    benchmark_ipfp_iterations(mobility_csv, metapop_csv, T=50, n_repeats=5)
    profile_components(mobility_csv, metapop_csv)
    benchmark_scales()
