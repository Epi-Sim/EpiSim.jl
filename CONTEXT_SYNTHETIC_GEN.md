# Synthetic Mobility Intervention Generator

This document describes the synthetic data generation pipeline used to benchmark mobility intervention strategies in `EpiSim.jl`.

## Overview

The goal of this system is to scientifically compare different mobility reduction strategies (e.g., Global Lockdowns vs. Localized Restrictions) by generating **Counterfactual "Twin" Scenarios**.

Instead of randomizing all parameters for every run, we generate specific **Epidemiological Profiles** (fixed R0, incubation period, infectious period). For each profile, we run a set of "Twin" simulations that differ *only* in their intervention strategy. This allows for a "like-to-like" comparison of effectiveness.

## Architecture

The pipeline consists of three phases:

### Phase 1: Artifact Generation (`python/synthetic_generator.py`)

This Python script uses `scipy.stats.qmc` (Latin Hypercube Sampling) to generate parameter sets and creates all necessary input files *before* execution.

1.  **Sampling**: Generates $N$ **Epidemiological Profiles**.
    *   Parameters: `R0_scale`, `T_inf` (Infectious Period), `T_inc` (Incubation Period), `Event Start`, `Event Duration`.
2.  **Scenario Setup**: For each profile, it prepares files for 3 intervention types (+ Baseline), swept across multiple strengths (0.0 to 1.0):
    *   **Baseline**: No intervention.
    *   **Global_Const**: Constant reduction of global mobility ($\kappa_0$) by factor $S$.
    *   **Global_Timed**: Reduction of $\kappa_0$ only during the event window.
    *   **Local_Static**: Structural reduction of the Mobility Matrix (reducing weights of edges connected to $X$% of nodes).
3.  **File Creation**:
    *   `config_auto_py.json`: Injected with specific params.
    *   `kappa0_*.csv`: Time-series for global reductions.
    *   `mobility_matrix_*.csv`: Modified matrices for local reductions.
    *   `seeds_*.csv`: Fixed seed location/count per profile.

### Phase 2: Batch Execution (`src/batch_run.jl`)

To maximize performance and avoid thread-safety issues with the NetCDF library, we use a distributed Julia executor.

*   **Mechanism**: Uses `Distributed.jl` to launch worker processes (`-p auto`).
*   **Safety**: Each simulation runs in its own process memory space, preventing race conditions in the underlying NetCDF-C library.
*   **Performance**: Amortizes the Julia runtime startup cost by launching workers once and distributing the $N \times M$ simulation tasks via `pmap`.

### Phase 3: Analysis & Plotting (`python/plot_synthetic_results.py`)

Parses the simulation outputs to generate comparative visualizations.

*   **Metrics**: Calculates **Relative Effectiveness** = $1 - (\text{Total Infections}_{\text{Scenario}} / \text{Total Infections}_{\text{Baseline}})$.
*   **Outputs**:
    *   `synthetic_profile_epicurves.png`: Side-by-side epicurves (Infections, Hospitalizations, Deaths) for each Profile, showing all "Twin" trajectories.
    *   `synthetic_relative_effectiveness.png`: Box plots of relative reduction per scenario type.
    *   `synthetic_relative_scatter.png`: Scatter plot of Reduction Strength vs. Relative Effectiveness.

## Usage

1.  **Run the Generator & Batch Executor**:
    ```bash
    cd python
    uv run python synthetic_generator.py
    ```
    *This will generate inputs in `runs/synthetic_test/` and immediately invoke the Julia batch runner.*

2.  **Generate Plots**:
    ```bash
    uv run python plot_synthetic_results.py
    ```
    *This scans `runs/synthetic_test/` and produces PNG visualizations.*

## Directory Structure

Outputs are stored in `runs/synthetic_test/` with the following naming convention:

```text
runs/synthetic_test/
├── run_{ProfileID}_{ScenarioType}_s{StrengthPercent}/
│   └── {UUID}/
│       ├── config_auto_py.json
│       └── output/
│           └── observables.nc
├── kappa0_*.csv            # Generated time series inputs
├── mobility_matrix_*.csv   # Generated matrix inputs
└── seeds_*.csv             # Generated seed inputs
```

## Key Configuration Concepts

*   **Global Interventions**: Controlled via `NPI.κ₀s` (kappa0) in the config or an external CSV. Represents a scalar reduction in mobility for *all* patches.
*   **Local Interventions**: Controlled by modifying the input `mobility_matrix.csv`. Represents structural changes to the connectivity graph (e.g., cordoning off specific regions).
