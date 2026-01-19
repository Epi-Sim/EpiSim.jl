# Synthetic Mobility Intervention Generator

This document describes the synthetic data generation pipeline used to benchmark mobility intervention strategies in `EpiSim.jl`.

## Overview

The goal of this system is to scientifically compare different mobility reduction strategies (e.g., Global Lockdowns vs. Localized Restrictions) by generating **Counterfactual "Twin" Scenarios**.

Instead of randomizing all parameters for every run, we generate specific **Epidemiological Profiles** (fixed R0, incubation period, infectious period). For each profile, we run a set of "Twin" simulations that differ *only* in their intervention strategy. This allows for a "like-to-like" comparison of effectiveness.

## Architecture

The pipeline consists of five phases:

### Phase 1: Artifact Generation (`python/synthetic_generator.py`)

This Python script uses `scipy.stats.qmc` (Latin Hypercube Sampling) to generate parameter sets and creates all necessary input files *before* execution.

1. **Sampling**: Generates $N$ **Epidemiological Profiles**.
    * Parameters: `R0_scale`, `T_inf` (Infectious Period), `T_inc` (Incubation Period), `Event Start`, `Event Duration`.
2. **Scenario Setup**: For each profile, it prepares files for 2 intervention types (+ Baseline), swept across multiple strengths (0.05 to 0.8):
    * **Baseline**: No intervention.
    * **Global_Timed**: Reduction of $\kappa_0$ only during the event window.
    * **Local_Static**: Structural reduction of the Mobility Matrix (reducing weights of edges connected to $X$% of nodes).
3. **File Creation**:
    * `config_auto_py.json`: Injected with specific params.
    * `kappa0_*.csv`: Time-series for global reductions.
    * `mobility_matrix_*.csv`: Modified matrices for local reductions.
    * `seeds_*.csv`: Fixed seed location/count per profile.

### Phase 2: Batch Execution (`src/batch_run.jl`)

To maximize performance and avoid thread-safety issues with the NetCDF library, we use a distributed Julia executor.

* **Mechanism**: Uses `Distributed.jl` to launch worker processes (`-p auto`).
* **Safety**: Each simulation runs in its own process memory space, preventing race conditions in the underlying NetCDF-C library.
* **Performance**: Amortizes the Julia runtime startup cost by launching workers once and distributing the $N \times M$ simulation tasks via `pmap`.

### Phase 3: Mobility Zarr Conversion (`python/convert_mobility_to_zarr.py`)

The simulator expects a static mobility matrix, but downstream forecasters consume a time-indexed mobility series. This helper converts a static CSV matrix into the downstream zarr format by repeating the matrix across a date range.

* **Inputs**: `mobility_matrix_*.csv`, `metapopulation_data.csv`, and either a date range or a `kappa0_*.csv` for dates.
* **Output**: zarr dataset with dims `date`, `origin`, `target` and variable `mobility`.

### Phase 4: Observation Layer + Zarr Consolidation (`python/process_synthetic_outputs.py`)

Creates reported-case and wastewater observation layers from infections and consolidates results into a single zarr dataset.

* **Inputs**: `run_*` folders, `observables.nc`, `metapopulation_data.csv`, and `kappa0_*.csv`.
* **Outputs**: `synthetic_observations.zarr` with dims `(run_id, region_id, date)` and data vars:
  * `ground_truth_infections`: Daily infections (sum over ages).
  * `obs_cases`: Reported cases using a logistic testing ramp + binomial noise.
  * `obs_wastewater`: Gamma-shedding convolution + log-normal noise.
  * `ascertainment_rate`: Daily reporting probability.
  * `mobility_reduction`: κ₀ reductions from the kappa0 CSV.

### Phase 5: Analysis & Plotting (`python/plot_synthetic_results.py`)

Parses the simulation outputs to generate comparative visualizations.

* **Metrics**: Calculates **Relative Effectiveness** = $1 - (\text{Total Infections}_{\text{Scenario}} / \text{Total Infections}_{\text{Baseline}})$.
* **Outputs**:
  * `synthetic_profile_epicurves.png`: Side-by-side epicurves (Infections, Hospitalizations, Deaths) for each Profile, showing all "Twin" trajectories.
  * `synthetic_relative_effectiveness.png`: Box plots of relative reduction per scenario type.
  * `synthetic_relative_scatter.png`: Scatter plot of Reduction Strength vs. Relative Effectiveness.

## Usage

1. **Run the Generator & Batch Executor**:

    ```bash
    cd python
    uv run python synthetic_generator.py
    ```

    *This will generate inputs in `runs/synthetic_test/` and immediately invoke the Julia batch runner.*

2. **Convert Mobility to Zarr**:

    ```bash
    uv run python convert_mobility_to_zarr.py \
        --mobility-csv ../runs/synthetic_test/mobility_matrix_0_Baseline.csv \
        --metapop-csv ../models/mitma/metapopulation_data.csv \
        --kappa0-csv ../runs/synthetic_test/kappa0_0_Baseline.csv \
        --output ../runs/synthetic_test/mobility.zarr
    ```

    *This creates a time-indexed mobility series for downstream models.*

3. **Build Observation-Layer Zarr**:

    ```bash
    uv run python process_synthetic_outputs.py \
        --runs-dir ../runs/synthetic_test \
        --metapop-csv ../models/mitma/metapopulation_data.csv \
        --output ../runs/synthetic_test/synthetic_observations.zarr \
        --preview-plot
    ```

    *This generates `synthetic_observations.zarr` and optional preview plots.*

4. **Generate Plots**:

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
├── mobility.zarr                  # Optional time-indexed mobility output
├── synthetic_observations.zarr    # Consolidated observations + ground truth
├── observation_preview_*.png      # Optional wastewater preview plots
└── seeds_*.csv                    # Generated seed inputs
```

## Key Configuration Concepts

* **Global Interventions**: Controlled via `NPI.κ₀s` (kappa0) in the config or an external CSV. Represents a scalar reduction in mobility for *all* patches.
* **Local Interventions**: Controlled by modifying the input `mobility_matrix.csv`. Represents structural changes to the connectivity graph (e.g., cordoning off specific regions).
