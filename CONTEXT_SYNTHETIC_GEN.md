# Synthetic Mobility Intervention Generator

This document describes the synthetic data generation pipeline used to benchmark mobility intervention strategies in `EpiSim.jl`.

## Overview

The goal of this system is to scientifically compare different mobility reduction strategies (e.g., Global Lockdowns vs. Localized Restrictions) by generating **Counterfactual "Twin" Scenarios**.

Instead of randomizing all parameters for every run, we generate specific **Epidemiological Profiles** (fixed R0, incubation period, infectious period). For each profile, we run a set of "Twin" simulations that differ *only* in their intervention strategy. This allows for a "like-to-like" comparison of effectiveness.

### Default Pipeline: Two-Phase Spike-Based Generation

The default synthetic data generation pipeline uses a **two-phase approach** to ensure realistic intervention timing:

1. **Phase 1 (Baseline Generation)**: Generate and run all baseline scenarios (no interventions) to establish true epidemic dynamics
2. **Phase 2 (Spike-Based Interventions)**: Analyze baseline outputs to detect actual infection spike periods, then generate intervention scenarios with realistic timing based on observed data

**Why This Matters**: The previous heuristic-based approach (fixed detection threshold of 100 cases + reaction delay) resulted in unrealistic premature lockdowns, often on day 1-10. The new spike-based approach ensures interventions are timed to coincide with actual epidemic growth, producing more realistic counterfactual scenarios.

### Default Time Range

The standard simulation window spans **2020-03-01 to 2021-05-09** (435 days):

- **Start Date (2020-03-01)**: Earliest case data available in the MITMA dataset
- **End Date (2021-05-09)**: Latest mobility data available in the MITMA dataset

This range ensures complete coverage of both case observations and mobility patterns, covering the full first year of the COVID-19 pandemic in Spain including multiple waves of infection and intervention periods.

## Architecture

The default pipeline uses a **two-phase spike-based approach** for realistic intervention timing:

### Two-Phase Pipeline (Default)

**Phase 1: Baseline Generation**
- Generate epidemiological profiles using Latin Hypercube Sampling
- Run baseline simulations (no interventions) to establish true epidemic dynamics
- Process baseline outputs into zarr format for spike detection

**Phase 2: Spike-Based Intervention Generation**
- Detect infection spike periods in baseline outputs using `spike_detector.py`
- Generate intervention scenarios with timing based on detected spikes
- Run intervention scenarios and append to baseline zarr

**Key Components:**
- `python/spike_detector.py`: Spike detection using percentile or prominence-based methods
- `python/synthetic_generator.py`: Modified with `--baseline-only` and `--intervention-only` modes
- `python/run_synthetic_pipeline.py`: Orchestrates two-phase execution with `--two-phase` flag

### Standard Pipeline (Legacy)

The pipeline also supports the original five-phase approach:

### Phase 1: Artifact Generation (`python/synthetic_generator.py`)

This Python script uses `scipy.stats.qmc` (Latin Hypercube Sampling) to generate parameter sets and creates all necessary input files *before* execution.

1. **Sampling**: Generates $N$ **Epidemiological Profiles**.
    * Parameters: `R0_scale`, `T_inf` (Infectious Period), `T_inc` (Incubation Period), `Event Start`, `Event Duration`.
    * **Event Timing**: Sampled relative to the default simulation window (2020-03-01 to 2021-05-09); events should fall within this range for valid comparisons.
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

* **Inputs**: `mobility_matrix_*.csv`, `metapopulation_data.csv`, and either a date range or a `kappa0_*.csv` for dates (default: 2020-03-01 to 2021-05-09).
* **Output**: zarr dataset with dims `date`, `origin`, `target` and variable `mobility`.

### Phase 4: Raw-ish Observation Generation (`python/process_synthetic_outputs.py`)

Creates **raw-ish** observations that mimic real-world data sources, with configurable noise, gaps, and censoring patterns. The output format matches EpiForecaster's expected input, so it can be run through the same preprocessing pipeline (Tobit Kalman filter, alignment, etc.) as real data.

**Key Design Principle**: The output is "raw enough" that EpiForecaster's preprocessing pipeline does meaningful work, but "structured enough" to match the expected input format. This ensures models trained on synthetic data see the same data distribution as models trained on real data.

*   **Inputs**: `run_*` folders, `observables.nc`, `metapopulation_data.csv`, `mobility_matrix.csv` (or `R_mobility_matrix.csv`), `kappa0_*.csv`, and `edar_muni_edges.nc` (for EDAR-based wastewater aggregation).
*   **Outputs**: `raw_synthetic_observations.zarr` with the following structure:

#### Raw Observations (for EpiForecaster preprocessing pipeline)

| Variable | Shape | Description |
| --- | --- | --- |
| `cases` | `(run_id, date, region_id)` | Raw case observations with configurable noise and missing data (NaN for missing) |
| `hospitalizations` | `(run_id, date, region_id)` | Reported hospitalizations with underreporting and delay noise |
| `deaths` | `(run_id, date, region_id)` | Reported deaths with underreporting and delay noise |
| `mobility_base` | `(origin, target)` | Base mobility matrix (shared across all runs) - factorized format |
| `mobility_kappa0` | `(run_id, date)` | Mobility reduction factor per run and date (κ₀) - factorized format |
| `mobility_time_varying` | `(run_id, origin, target, date)` | Full time-varying mobility matrix per run (optional, large format) |
| `population` | `(run_id, region_id)` | Static population per region |
| `edar_biomarker_N1` | `(run_id, date, edar_id)` | Raw N1 wastewater concentration (NaN for censored/missing) |
| `edar_biomarker_N2` | `(run_id, date, edar_id)` | Raw N2 wastewater concentration (NaN for censored/missing) |
| `edar_biomarker_IP4` | `(run_id, date, edar_id)` | Raw IP4 wastewater concentration (NaN for censored/missing) |
| `edar_biomarker_*_censor_hints` | `(run_id, date, edar_id)` | Censoring hints: 0=observed, 1=censored, 2=missing (optional reference) |

**Note:** Mobility is stored in two possible formats:

**Factorized Format** (memory-efficient, recommended):
- `mobility_base`: Static OD matrix (shared across all runs)
- `mobility_kappa0`: Time-varying reduction factors per run
- **Reconstruction:** `mobility[run, date] = mobility_base * (1 - mobility_kappa0[run, date])`
- **Memory savings:** ~325GB → ~500MB for 100 runs × 100 days × 2850 regions (99.8% reduction)

**Time-Varying Format** (optional, direct access):
- `mobility_time_varying`: Full `(run_id, origin, target, date)` array
- **Use case:** When direct access to full mobility matrices is needed without reconstruction
- **Memory cost:** ~19GB for 23 runs × 114 days × 945 regions (substantial but manageable)

The `synthetic_mobility_type` metadata variable indicates which format is used for each run. EpiForecaster's `mobility_processor.py` automatically detects and handles both formats.

#### Ground Truth (for evaluation, separate from preprocessed data)

| Variable | Shape | Description |
| --- | --- | --- |
| `infections_true` | `(run_id, region_id, date)` | Daily infections (sum over ages) |
| `hospitalizations_true` | `(run_id, region_id, date)` | Daily hospitalizations |
| `deaths_true` | `(run_id, region_id, date)` | Daily deaths |
| `latent_S_true`, `latent_E_true`, `latent_A_true`, `latent_I_true`, `latent_R_true`, `latent_D_true` | `(run_id, region_id, date)` | Optional latent simulator states exported from `compartments_full.nc` for hybrid supervision |
| `latent_CH_true` | `(run_id, region_id, date)` | Optional confined-population latent target |
| `latent_hospitalized_true` | `(run_id, region_id, date)` | Optional latent hospital occupancy target (`HR + HD`) |
| `latent_active_true` | `(run_id, region_id, date)` | Optional latent active-burden target (`E + A + I + PH + PD + HR + HD`) |

Latent targets are synthetic-only. They are not available in real-data settings, but can be used as auxiliary training targets or regularizers for hybrid deep learning / mechanistic models. Enable them with `--include-latents` when running `python/process_synthetic_outputs.py` or `python/run_synthetic_pipeline.py`.

#### Synthetic Metadata (for reference, not used by preprocessor)

| Variable | Shape | Description |
| --- | --- | --- |
| `synthetic_scenario_type` | `(run_id,)` | Scenario type (Baseline, Global_Timed, Local_Static) |
| `synthetic_strength` | `(run_id,)` | Intervention strength (0.0 to 1.0) |
| `synthetic_sparsity_level` | `(run_id,)` | Missing data rate used |
| `synthetic_mobility_type` | `(run_id,)` | Mobility storage format ("factorized" or "time_varying") |

#### Cases Reporting Noise Metadata

| Variable | Shape | Description |
| --- | --- | --- |
| `synthetic_cases_report_rate_min` | `(run_id,)` | Minimum cases ascertainment rate (logistic ramp) |
| `synthetic_cases_report_rate_max` | `(run_id,)` | Maximum cases ascertainment rate (logistic ramp) |
| `synthetic_cases_report_delay_mean` | `(run_id,)` | Cases reporting delay in days (0 = not modeled) |

#### Hospitalizations Reporting Noise Metadata

| Variable | Shape | Description |
| --- | --- | --- |
| `synthetic_hosp_report_rate` | `(run_id,)` | Hospitalization reporting rate (0.0 to 1.0) |
| `synthetic_hosp_report_delay_mean` | `(run_id,)` | Mean hospitalization reporting delay in days |
| `synthetic_hosp_report_delay_std` | `(run_id,)` | Std dev of hospitalization reporting delay |

#### Deaths Reporting Noise Metadata

| Variable | Shape | Description |
| --- | --- | --- |
| `synthetic_deaths_report_rate` | `(run_id,)` | Deaths reporting rate (0.0 to 1.0) |
| `synthetic_deaths_report_delay_mean` | `(run_id,)` | Mean deaths reporting delay in days |
| `synthetic_deaths_report_delay_std` | `(run_id,)` | Std dev of deaths reporting delay |

#### Wastewater Noise Metadata

| Variable | Shape | Description |
| --- | --- | --- |
| `synthetic_ww_noise_sigma_N1` | `(run_id,)` | Log-normal noise sigma for N1 gene target |
| `synthetic_ww_noise_sigma_N2` | `(run_id,)` | Log-normal noise sigma for N2 gene target |
| `synthetic_ww_noise_sigma_IP4` | `(run_id,)` | Log-normal noise sigma for IP4 gene target |
| `synthetic_ww_transport_loss` | `(run_id,)` | Signal decay in sewer system (transport loss) |

#### Observation Layer Features

1.  **Configurable Missing Data Patterns**:
    *   **Random sparse missing**: Scattered individual missing values (NaN)
    *   **Gap-based missing**: Consecutive day gaps (e.g., system outages)
    *   Configurable via `--missing-rate` and `--missing-gap-length`

2.  **Wastewater Censoring**:
    *   **Probabilistic LoD**: Uses logistic probability curve for detection near limit
    *   **Missing measurements**: 2% of measurements marked as missing (NaN)
    *   **Censor hints**: Optional flags for reference (EpiForecaster may recompute)

3.  **Mobility Compression**:
    *   **Factorized Format**: Stored as `mobility_base` + `mobility_kappa0` (memory-efficient)
    *   **Time-Varying Format**: Full `mobility_time_varying` array (direct access, larger)
    *   Factorized format reduces memory from ~325GB to ~500MB for 100 runs
    *   Time-varying format uses ~19GB for 23 runs × 114 days × 945 regions
    *   The `synthetic_mobility_type` metadata indicates which format each run uses
    *   Both formats automatically detected by EpiForecaster's mobility processor
    *   Configurable compressor via `--compressor` (zstd, lz4, blosc, none)

4.  **Variable Naming Convention**:
    *   Matches EpiForecaster's expected input format
    *   `cases`, `mobility`, `population` use same names as real data
    *   `edar_biomarker_*` matches EpiForecaster's processed output format
    *   `*_censor_hints` suffix indicates optional reference data

#### Wastewater Physics Model

The wastewater generation uses a "High-Fidelity" physical model based on literature consensus (e.g., *Xu et al.*, *Ma et al.*):

1.  **Age-Stratified Shedding**:
    *   **Children (<18)**: Modeled with a "Long Tail" kernel ($\alpha=1.5, \beta=10.0$, mean 15d). Reflects prolonged fecal shedding despite rapid respiratory clearance.
    *   **Adults**: Modeled with an "Acute Phase" kernel ($\alpha=2.5, \beta=4.0$, mean 10d). Tightly correlated with symptoms.
    *   **Result**: "School Signatures" (residential areas) show lingering signals, while "Workforce Signatures" (business districts) are spikier.

2.  **Dilution & Transport**:
    *   **Concentration**: $C = \frac{\sum (I_g \times Shedding_g)}{Population \times FlowPerCapita}$
    *   **Detection**: Calibrated to real-world datasets (median LoD = 375 Copies/L).
    *   **Result**: Small clusters (30 cases) are detectable in villages (Pop 5k) but physically invisible in metropolises (Pop 500k) due to massive dilution.

3.  **Gene Targets**:
    *   **N1**: High sensitivity (Scale 500k), Robust (LoD 375).
    *   **N2**: Moderate sensitivity (Scale 400k), Noisier (LoD 500).
    *   **IP4**: Unstable/Low sensitivity (Scale 250k), High decay (LoD 800).

### EDAR-Level Biomarker Generation

When generating biomarkers at EDAR (wastewater treatment plant) level, the physical model applies as follows:

1. **Infection Aggregation**: Use EMAP to aggregate infections from municipalities to EDARs
   *   `infections_edar[EDAR] = sum(EMAP[EDAR, region] × infections[region])`
   *   EMAP `contribution_ratio` = wastewater flow fraction from each municipality to the EDAR
   *   This gives the **total infected people whose wastewater flows to each EDAR**

2. **Population Division**: Divide by EDAR catchment population (NOT per-capita normalization)
   *   `ww_signal = convolve(infections_edar, shedding_kernel) / population_edar`
   *   `population_edar` = total population in EDAR catchment (sum of contributing municipalities)
   *   This models **DILUTION**: more wastewater flow = lower concentration

3. **Why This Is Correct**
   *   The `contribution_ratio` represents **wastewater flow fractions**, not population distribution
   *   A municipality may send 50% of wastewater to EDAR1 and 50% to EDAR2 (flow-based)
   *   But its population might be distributed differently across catchment areas (geography-based)
   *   The division by population models the **physical dilution** in the sewer network

4. **Implementation Notes**
   *   `FlowPerCapita` is absorbed into `sensitivity_scale` as a simplifying assumption
   *   Default `sensitivity_scale = 1.0` implies normalized per-capita flow (unitless)
   *   Real-world factors (rainfall variation, sewer hydraulics) are not explicitly modeled
   *   `sensitivity_scale` can be tuned to calibrate to real-world datasets (LoD = 375 Copies/L)

### Common Pitfall

❌ **Incorrect Thinking**: "Infections are already the signal, don't scale by population"

✅ **Correct Understanding**: "Infection counts must be normalized by population to get concentration (copies/L),
     which is what wastewater surveillance actually measures. The division models **dilution physics**:
     more population = more wastewater flow = lower concentration per unit volume."

### Phase 5: Analysis & Plotting (`python/plot_synthetic_results.py`)

Parses the simulation outputs to generate comparative visualizations.

* **Metrics**: Calculates **Relative Effectiveness** = $1 - (\text{Total Infections}_{\text{Scenario}} / \text{Total Infections}_{\text{Baseline}})$.
* **Outputs**:
  * `synthetic_profile_epicurves.png`: Side-by-side epicurves (Infections, Hospitalizations, Deaths) for each Profile, showing all "Twin" trajectories.
  * `synthetic_relative_effectiveness.png`: Box plots of relative reduction per scenario type.
  * `synthetic_relative_scatter.png`: Scatter plot of Reduction Strength vs. Relative Effectiveness.

## Usage

### Recommended: Two-Phase Pipeline (Default)

The two-phase pipeline generates realistic intervention timing by analyzing baseline epidemic dynamics:

```bash
# Run the two-phase pipeline (default behavior)
uv run python/run_synthetic_pipeline.py --two-phase --n-profiles 15 --spike-threshold 0.1

# Output: runs/synthetic_two_phase/baselines/raw_synthetic_observations.zarr
```

**Two-Phase Pipeline Options:**
- `--n-profiles`: Number of epidemiological profiles (default: 15)
- `--spike-threshold`: Percentile threshold for spike detection (default: 0.1 = 10th percentile)
- `--dataset`: Dataset to use (catalonia or mitma, default: catalonia)
- `--edar-edges`: Path to EDAR-municipality edges file
- `--mobility-sigma-min`: Minimum mobility sigma for origin/destination noise (default: 0.0)
- `--mobility-sigma-max`: Maximum mobility sigma for origin/destination noise (default: 0.6)
- `--skip-sim`: Skip simulation stage (use existing runs)
- `--skip-process`: Skip processing stage (use existing zarr)
- `--skip-plot`: Skip plotting stage
- `--dry-run`: Show what would be run without executing

**Manual Two-Phase Execution:**

For more control over the pipeline, you can run each phase separately:

```bash
# Phase 1: Generate baselines
uv run python/synthetic_generator.py --n-profiles 15 --baseline-only --output-folder runs/two_phase/baselines --mobility-sigma-max 0.6
uv run python/process_synthetic_outputs.py --runs-dir runs/two_phase/baselines --baseline-only --output runs/two_phase/baselines/baseline.zarr

# Phase 2: Generate spike-based interventions
uv run python/synthetic_generator.py --intervention-only runs/two_phase/baselines --spike-threshold 0.1 --output-folder runs/two_phase/interventions --mobility-sigma-max 0.6
uv run python/process_synthetic_outputs.py --runs-dir runs/two_phase/interventions --append --output runs/two_phase/baselines/baseline.zarr
```

**Analyze Spikes in Existing Baselines:**

```bash
# Detect and display spikes in existing baseline zarr
uv run python/python/spike_detector.py runs/two_phase/baselines/baseline.zarr --spike-threshold 0.1
```

### Intervention Proportion Control

Control the fraction of profiles that receive intervention sweeps:

```bash
# Default: All profiles get interventions (86% of runs are interventions)
--n-profiles 15

# Mostly baselines: Only 30% of profiles get interventions
--n-profiles 15 --intervention-profile-fraction 0.3
# Result: 15 baselines + (5 × 6) = 30 interventions = 45 total runs (67% baselines)

# Very baseline-heavy: Only 20% get interventions
--n-profiles 15 --intervention-profile-fraction 0.2
# Result: 15 baselines + (3 × 6) = 18 interventions = 33 total runs (82% baselines)

# Baseline only
--n-profiles 15 --intervention-profile-fraction 0.0
# Result: 15 baselines + 0 interventions = 15 total runs (100% baselines)
```

**Key Points:**
- All profiles always generate a baseline scenario
- Selected profiles get the full 6-strength intervention sweep
- Profile selection uses a fixed seed for reproducibility
- Works in both Phase 1 (standard mode) and Phase 2 (spike-based mode)
- Default `--intervention-profile-fraction 1.0` preserves backward compatibility (all profiles get interventions)
- Use `--intervention-seed` to control the random seed for profile selection (default: 42)

### Legacy: Standard Pipeline

For compatibility with existing workflows, the original single-phase pipeline is still available:

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
        --output ../runs/synthetic_test/mobility.zarr \
        --start-date 2020-03-01 \
        --end-date 2021-05-09
    ```

    *This creates a time-indexed mobility series for downstream models.*

3. **Build Raw Observation Zarr**:

    ```bash
    uv run python process_synthetic_outputs.py \
        --runs-dir ../runs/synthetic_test \
        --metapop-csv ../models/mitma/metapopulation_data.csv \
        --edar-edges ../edar_muni_edges.nc \
        --output ../runs/synthetic_test/raw_synthetic_observations.zarr \
        --missing-rate 0.05 \
        --missing-gap-length 3 \
        --compressor zstd \
        --compressor-level 3
    ```

    **Noise Parameters** (for curriculum progression):
    *   `--missing-rate`: Fraction of data to make missing (default: 0.05)
    *   `--missing-gap-length`: Average length of missing data gaps (default: 3 days)

    **Cases Reporting Noise**:
    *   `--min-rate`: Minimum cases ascertainment rate (default: 0.05)
    *   `--max-rate`: Maximum cases ascertainment rate (default: 0.6)

    **Hospitalizations Reporting Noise**:
    *   `--hosp-report-rate`: Hospitalization reporting rate (default: 0.85)
    *   `--hosp-delay-mean`: Mean reporting delay in days (default: 3)
    *   `--hosp-delay-std`: Std dev of reporting delay (default: 1)

    **Deaths Reporting Noise**:
    *   `--deaths-report-rate`: Deaths reporting rate (default: 0.90)
    *   `--deaths-delay-mean`: Mean reporting delay in days (default: 7)
    *   `--deaths-delay-std`: Std dev of reporting delay (default: 2)

    **Wastewater Noise**:
    *   `--noise-sigma`: Log-normal noise sigma (default: 0.5)
    *   `--ww-noise-n1`: Override noise sigma for N1 target (default: 0.5)
    *   `--ww-noise-n2`: Override noise sigma for N2 target (default: 0.8)
    *   `--ww-noise-ip4`: Override noise sigma for IP4 target (default: 0.6)
    *   `--ww-transport-loss`: Signal decay in sewer (default: 0.0)

    **Other Options**:
    *   `--compressor`: Compressor for zarr (zstd, lz4, blosc, none)
    *   `--compressor-level`: Compression level (default: 3)
    *   `--append`: Append to existing zarr store for incremental generation

    **Data Type Control**:
    *   `--dtype`: Output dtype for floating-point arrays (`float16`, `float32`, `float64`, default: `float16`)
        *   `float16`: 75% memory reduction, ~3.3 decimal precision, fails if values exceed ±65504
        *   `float32`: 50% memory reduction, ~7 decimal precision
        *   `float64`: Full precision (legacy behavior)
        *   The default `float16` is safe for typical epidemic data (cases, hospitalizations, deaths all fit within range)

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
├── raw_synthetic_observations.zarr # Raw observations for preprocessing pipeline
└── seeds_*.csv                    # Generated seed inputs

# EDAR-municipality mapping (typically at project root)
edar_muni_edges.nc                 # For EDAR-based wastewater aggregation
```

## Key Configuration Concepts

* **Simulation Time Window**: Default range is 2020-03-01 to 2021-05-09 (435 days), covering the full first year of the COVID-19 pandemic in Spain with complete case and mobility data coverage.
* **Global Interventions**: Controlled via `NPI.κ₀s` (kappa0) in the config or an external CSV. Represents a scalar reduction in mobility for *all* patches.
* **Local Interventions**: Controlled by modifying the input `mobility_matrix.csv`. Represents structural changes to the connectivity graph (e.g., cordoning off specific regions).
