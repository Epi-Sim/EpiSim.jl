Fix DomainError in MMCAcovid19 Simulations

  ## Issue

  Synthetic data generation fails with `DomainError` when running simulations:
  ```
  DomainError with -0.22427440633245382:
  Exponentiation yielding a complex result requires a complex argument.
  ```

  **Root Cause:** In MMCAcovid19.jl's `markov.jl:100`, the calculation:
  ```julia
  CHᵢ = (1 - ϕ) * κ₀ * (CHᵢ / population.nᵢ[i]) ^ population.σ
  ```
  fails when `CHᵢ / population.nᵢ[i]` becomes negative due to numerical drift, then is raised to
  power σ (≈2.5).

  **Parameter Validity:** ✅ Parameters are VALID per NPI_PARAMETERS.md:
  - κ₀ = 0.8 (within [0, 1])
  - ϕ = 0.2 (within [0, 1])
  - δ = 0.8 (within [0, 1])

  This is a numerical instability issue, not invalid parameters.

  ---

  ## Solution: Two-Layer Fix

  ### Layer 1: Fix at Source (MMCAcovid19.jl) - PRIMARY

  **File:** `~/.julia/packages/MMCAcovid19/CkulC/src/markov.jl`
  **Line:** 100

  ```julia
  # BEFORE:
  CHᵢ = (1 - ϕ) * κ₀ * (CHᵢ / population.nᵢ[i]) ^ population.σ

  # AFTER:
  ratio_CH = max(0.0, CHᵢ / population.nᵢ[i])
  CHᵢ = (1 - ϕ) * κ₀ * ratio_CH ^ population.σ
  ```

  ### Layer 2: Add Negative Value Clamping (EpiSim.jl) - DEFENSIVE

  **File:** `/Volumes/HUBSSD/code/EpiSim.jl/src/engine.jl`

  Add negative value clamping alongside existing NaN handling (after line 518):

  ```julia
  # Existing NaN clamping (keep this)
  epi_params.ρˢᵍᵥ[isnan.(epi_params.ρˢᵍᵥ)]   .= 0
  # ... (keep all existing NaN clamps)

  # NEW: Also clamp negative values
  epi_params.ρˢᵍᵥ[epi_params.ρˢᵍᵥ .< 0]   .= 0
  epi_params.ρᴱᵍᵥ[epi_params.ρᴱᵍᵥ .< 0]   .= 0
  epi_params.ρᴬᵍᵥ[epi_params.ρᴬᵍᵥ .< 0]   .= 0
  epi_params.ρᴵᵍᵥ[epi_params.ρᴵᵍᵥ .< 0]   .= 0
  epi_params.ρᴾᴴᵍᵥ[epi_params.ρᴾᴴᵍᵥ .< 0] .= 0
  epi_params.ρᴾᴰᵍᵥ[epi_params.ρᴾᴰᵍᵥ .< 0] .= 0
  epi_params.ρᴴᴿᵍᵥ[epi_params.ρᴴᴿᵍᵥ .< 0] .= 0
  epi_params.ρᴴᴰᵍᵥ[epi_params.ρᴴᴰᵍᵥ .< 0] .= 0
  epi_params.ρᴿᵍᵥ[epi_params.ρᴿᵍᵥ .< 0]   .= 0
  epi_params.ρᴰᵍᵥ[epi_params.ρᴰᵍᵥ .< 0]   .= 0
  epi_params.CHᵢᵍᵥ[epi_params.CHᵢᵍᵥ .< 0]   .= 0
  ```

  Repeat the same for the non-vaccinated engine (after line 579).

  ---

  ## Files to Modify

  | File | Lines | Change |
  |------|-------|--------|
  | `~/.julia/packages/MMCAcovid19/CkulC/src/markov.jl` | 100 | Add `max(0.0, ...)` clamp |
  | `/Volumes/HUBSSD/code/EpiSim.jl/src/engine.jl` | ~519 | Add negative clamping (vaccinated) |
  | `/Volumes/HUBSSD/code/EpiSim.jl/src/engine.jl` | ~580 | Add negative clamping (non-vaccinated)
  |

  ---

  ## Verification Steps

  ### 1. Test with Single Simulation
  ```bash
  cd /Volumes/HUBSSD/code/EpiSim.jl/python
  # Run a small test batch
  uv run synthetic_generator.py --n-profiles 2 --start-index 0 --end-index 2
  ```

  ### 2. Verify No DomainErrors
  ```bash
  # Check for ERROR.json files (should be none or fewer)
  find runs/synthetic_catalonia -name "ERROR.json" -type f
  ```

  ### 3. Run Full Batch
  ```bash
  # Run the original failing scenario
  uv run synthetic_generator.py --n-profiles 15
  ```

  ---

  ## Summary

  - **2 files modified** (1 external package, 1 EpiSim.jl)
  - **~15 lines added** (negative clamping alongside existing NaN handling)
  - **Fixes root cause** (MMCAcovid19) + adds defensive layer (EpiSim.jl)

