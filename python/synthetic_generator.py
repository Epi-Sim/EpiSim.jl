import json
import logging
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import qmc

# Ensure we can import episim_python
sys.path.append(os.path.dirname(__file__))

from episim_python.episim_utils import EpiSimConfig
from episim_python.mobility import MobilityGenerator, load_baseline_mobility
from failed_profiles_logger import FailedProfilesLogger, scan_and_log_existing_failures

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SyntheticGenerator")


def generate_profile_standalone(
    profile: dict,
    base_config_path: str,
    data_folder: str,
    output_folder: str,
    baseline_only: bool = False,
    intervention_profiles: set = None,
    mobility_sigma_min: float = 0.0,
    mobility_sigma_max: float = 0.6,
) -> dict:
    """
    Standalone function to generate a single profile's configuration.
    Can be executed in parallel via multiprocessing.

    Returns:
        dict with keys: success (bool), profile_id (int), error (str or None)
    """
    try:
        # Create generator instance (loads data files)
        generator = SyntheticDataGenerator(
            base_config_path=base_config_path,
            data_folder=data_folder,
            output_folder=output_folder,
        )
        generator.mobility_sigma_min = mobility_sigma_min
        generator.mobility_sigma_max = mobility_sigma_max

        # Run sweep for this profile
        generator.run_profile_sweep(
            profile=profile,
            baseline_only=baseline_only,
            intervention_profiles=intervention_profiles
            if intervention_profiles
            else None,
        )

        return {
            "success": True,
            "profile_id": profile["profile_id"],
            "error": None,
        }

    except Exception as e:
        logger.error(f"Profile {profile['profile_id']} failed: {e}")
        return {
            "success": False,
            "profile_id": profile["profile_id"],
            "error": str(e),
        }


class SyntheticDataGenerator:
    def __init__(self, base_config_path, data_folder, output_folder):
        self.base_config_path = base_config_path
        self.data_folder = data_folder
        self.output_folder = output_folder

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Load base configuration
        self.base_config = EpiSimConfig.from_json(base_config_path)

        # Initialize failure logger
        self.failure_logger = FailedProfilesLogger(
            log_file=os.path.join(output_folder, "failed_profiles.jsonl")
        )

        # Get data file paths from config
        metapop_filename = self.base_config.get_param(
            "data.metapopulation_data_filename"
        )
        mobility_filename = self.base_config.get_param("data.mobility_matrix_filename")

        # Load Metapopulation Data for seeding (ensure ID is read as string)
        self.metapop_df = pd.read_csv(
            os.path.join(data_folder, metapop_filename), dtype={"id": str}
        )

        # Load Mobility Matrix
        self.mobility_df = pd.read_csv(os.path.join(data_folder, mobility_filename))

        # Normalize mobility matrix to be row-stochastic (routing probabilities)
        # Identify origin column (assume first column)
        origin_col = self.mobility_df.columns[0]
        ratio_col = (
            self.mobility_df.columns[2]
            if len(self.mobility_df.columns) >= 3
            else self.mobility_df.columns[-1]
        )

        # Calculate row sums
        row_sums = self.mobility_df.groupby(origin_col)[ratio_col].transform("sum")

        # Avoid division by zero
        row_sums = row_sums.replace(0, 1.0)

        # Normalize
        self.mobility_df[ratio_col] = self.mobility_df[ratio_col] / row_sums

        logger.info(f"Loaded and normalized mobility matrix from {mobility_filename}")

        self.mobility_sigma_min = 0.0
        self.mobility_sigma_max = 0.6

        # Load rosetta data for correct index mapping
        rosetta_filename = metapop_filename.replace("metapopulation_data", "rosetta")
        try:
            rosetta_path = os.path.join(data_folder, rosetta_filename)
            self.rosetta_df = pd.read_csv(rosetta_path, dtype={"id": str})
        except FileNotFoundError:
            # Fallback to default rosetta.csv
            self.rosetta_df = pd.read_csv(
                os.path.join(data_folder, "rosetta.csv"), dtype={"id": str}
            )

    def generate_parameter_grid(
        self,
        n_profiles=5,
        seed=42,
        mobility_sigma_min=None,
        mobility_sigma_max=None,
    ):
        """
        Generate Latin Hypercube Samples for Epidemiological Profiles:
        0: R0_scale (scale_β) [0.5, 3.0]
        1: T_inf (Infectious Period) [2.0, 10.0] -> μᵍ = 1/T (recovery rate)
        2: T_inc (Incubation Period) [2.0, 10.0] -> ηᵍ = 1/T (progression rate)
        3: Reaction Delay [0, 30] (Replaces absolute Event Start)
        4: Event Duration [7, 60]
        5: Affected Fraction [0.1, 0.6] (For Local Scenarios)
        6: Ratio Beta A (ratio_beta_a) [0.1, 1.0]
        7: Alpha Scale (alpha_scale) [0.5, 1.5]
        8: Mu Scale (mu_scale) [0.5, 1.5] (unused, kept for backward compatibility)
        9: Seed Size [10, 500] (Log-uniform sampled)
        10: Mobility Sigma O [mobility_sigma_min, mobility_sigma_max] - Origin outflow noise level
        11: Mobility Sigma D [mobility_sigma_min, mobility_sigma_max] - Destination inflow noise level

        NOTE: We enforce T_inf <= T_inc to ensure μᵍ >= ηᵍ for model stability.
        """
        # Use rng parameter for scipy >= 1.10, fallback to seed for older versions
        try:
            sampler = qmc.LatinHypercube(d=12, seed=int(seed))
        except TypeError:
            sampler = qmc.LatinHypercube(d=12, rng=np.random.default_rng(int(seed)))
        sample = sampler.random(n=n_profiles)

        if mobility_sigma_min is None:
            mobility_sigma_min = self.mobility_sigma_min
        if mobility_sigma_max is None:
            mobility_sigma_max = self.mobility_sigma_max

        # Scale samples
        l_bounds = [
            0.5,
            2.0,
            2.0,
            0.0,
            7.0,
            0.1,
            0.1,
            0.5,
            0.5,
            10.0,
            mobility_sigma_min,
            mobility_sigma_min,
        ]
        u_bounds = [
            3.0,
            10.0,
            10.0,
            30.0,
            60.0,
            0.6,
            1.0,
            1.5,
            1.5,
            500.0,
            mobility_sigma_max,
            mobility_sigma_max,
        ]

        scaled = qmc.scale(sample, l_bounds, u_bounds)

        profiles = []
        detection_threshold = 100.0  # Assumed detected cases

        for i, row in enumerate(scaled):
            (
                r0,
                t_inf,
                t_inc,
                delay,
                duration,
                fraction,
                ratio_beta_a,
                alpha_scale,
                mu_scale,
                seed,
                mobility_sigma_O,
                mobility_sigma_D,
            ) = row

            # Enforce T_inf <= T_inc to ensure μᵍ >= ηᵍ for model stability
            # If T_inf > T_inc, swap the values
            if t_inf > t_inc:
                t_inf, t_inc = t_inc, t_inf

            # Heuristic for Event Start
            # Time to detection ~ T_inf * ln(Threshold/Seed) / (R0 - 1)
            # If R0 <= 1, growth is negative or flat. Set a default late start or based on delay.
            if r0 > 1.05:
                # Basic SIR growth approximation
                # Doubling time Td = T_inf * ln(2) / (R0 - 1)
                # Time to grow from Seed to Threshold
                if seed < detection_threshold:
                    growth_rate = (r0 - 1.0) / t_inf
                    time_to_detect = np.log(detection_threshold / seed) / growth_rate
                else:
                    time_to_detect = 0.0
            else:
                # Low R0: Epidemic won't grow fast.
                # Intervention is less critical/realistic.
                # Set a baseline delay or "late" start.
                time_to_detect = 20.0

            # event_start is detection time + reaction delay
            event_start = max(1, int(time_to_detect + delay))

            profiles.append(
                {
                    "profile_id": i,
                    "r0_scale": r0,
                    "t_inf": t_inf,
                    "t_inc": t_inc,
                    "event_start": event_start,
                    "event_duration": int(duration),
                    "affected_fraction": fraction,
                    "ratio_beta_a": ratio_beta_a,
                    "alpha_scale": alpha_scale,
                    "mu_scale": mu_scale,
                    "seed_size": int(seed),
                    "mobility_sigma_O": mobility_sigma_O,
                    "mobility_sigma_D": mobility_sigma_D,
                }
            )

        return profiles

    def validate_profile_parameters(self, profile):
        """
        Validate a profile for model stability constraints.

        Additional validation rules based on MMCA model constraints:
        - All rates (μᵍ, ηᵍ) must be in valid range [0.01, 1.0]
        - Probabilities derived from rates must stay in [0, 1]
        - R0 * scale must produce reasonable infection probabilities

        Returns:
            bool: True if profile is valid, False otherwise
        """
        # Check μᵍ >= ηᵍ (i.e., T_inf <= T_inc)
        # If recovery rate is less than progression rate, the Markov chain becomes unstable
        mu = 1.0 / profile["t_inf"]
        eta = 1.0 / profile["t_inc"]

        if mu < eta:
            logger.warning(
                f"Profile {profile['profile_id']}: μ ({mu:.3f}) < η ({eta:.3f}), "
                f"T_inf={profile['t_inf']:.1f}, T_inc={profile['t_inc']:.1f}. "
                f"Skipping - this would cause DomainError in simulation."
            )
            return False

        # Additional validation: Rates must be in reasonable range
        # Rates > 1.0 can cause probabilities to exceed [0, 1]
        if mu > 1.0:
            logger.warning(
                f"Profile {profile['profile_id']}: μ ({mu:.3f}) > 1.0, "
                f"T_inf={profile['t_inf']:.1f} is too short. "
                f"Skipping - recovery rate exceeds probability bounds."
            )
            return False

        if eta > 1.0:
            logger.warning(
                f"Profile {profile['profile_id']}: η ({eta:.3f}) > 1.0, "
                f"T_inc={profile['t_inc']:.1f} is too short. "
                f"Skipping - progression rate exceeds probability bounds."
            )
            return False

        # Check that rates aren't too small (numerical stability)
        if mu < 0.01:
            logger.warning(
                f"Profile {profile['profile_id']}: μ ({mu:.4f}) < 0.01, "
                f"T_inf={profile['t_inf']:.1f} is too long. "
                f"Skipping - recovery rate too small for numerical stability."
            )
            return False

        if eta < 0.01:
            logger.warning(
                f"Profile {profile['profile_id']}: η ({eta:.4f}) < 0.01, "
                f"T_inc={profile['t_inc']:.1f} is too long. "
                f"Skipping - progression rate too small for numerical stability."
            )
            return False

        # Check R0 > 1 for valid epidemic growth (warn only)
        if profile["r0_scale"] < 1.0:
            logger.warning(
                f"Profile {profile['profile_id']}: R0_scale={profile['r0_scale']:.2f} < 1.0 "
                f"may not produce epidemic growth."
            )

        # Check alpha_scale doesn't produce probabilities > 1
        # alpha_scale is applied to base_alpha values around 0.1, so max should be ~10x
        alpha_scale = profile.get("alpha_scale", 1.0)
        if alpha_scale > 1.5:
            logger.warning(
                f"Profile {profile['profile_id']}: alpha_scale={alpha_scale:.2f} > 1.5 "
                f"exceeds LHS bounds and causes DomainError. Skipping."
            )
            return False

        return True

    def prepare_kappa0_file(
        self,
        run_id,
        scenario,
        strength,
        profile,
        start_date_str,
        end_date_str,
        output_dir=None,
    ):
        """Generate kappa0.csv file for ALL scenarios.

        Args:
            output_dir: If provided, write file to this directory. Otherwise use self.output_folder.
        """
        # Parse dates
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)

        # Create date range (inclusive)
        date_range = pd.date_range(start=start_date, end=end_date)
        n_days = len(date_range)

        # Initialize dataframe with default κ₀=0.0 (no confinement, full mobility)
        df = pd.DataFrame(
            {
                "date": date_range,
                "reduction": 0.0,
                "datetime": date_range,
                "time": range(n_days),
            }
        )

        kappa_val = strength

        if scenario == "Global_Timed":
            event_start = profile["event_start"]
            event_duration = profile["event_duration"]

            # Ensure indices are within bounds
            start_idx = min(event_start, n_days)
            end_idx = min(event_start + event_duration, n_days)

            # Apply reduction during event window
            if start_idx < end_idx:
                df.loc[start_idx : end_idx - 1, "reduction"] = kappa_val
        elif scenario == "Baseline":
            df["reduction"] = 0.0

        # Format dates as string for CSV
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d")

        filename = "kappa0.csv"
        output_path = output_dir or self.output_folder
        path = os.path.join(output_path, filename)
        df.to_csv(path, index=False)
        return filename, path

    def prepare_mobility_file(
        self,
        run_id,
        scenario,
        reduction_strength,
        affected_fraction,
        output_dir=None,
        profile=None,
        start_date=None,
        end_date=None,
        source_mobility_npz=None,
    ):
        """
        Prepare mobility matrix based on scenario.
        For Global scenarios, return original matrix (unscaled).

        If mobility sigma parameters are provided in the profile, generates
        time-varying mobility series using IPFP.

        Args:
            output_dir: If provided, write file to this directory. Otherwise use self.output_folder.
            profile: Profile dict containing mobility_sigma_O and mobility_sigma_D
            start_date: Simulation start date (for determining T)
            end_date: Simulation end date (for determining T)
            source_mobility_npz: Path to existing mobility_series.npz to reuse (for Twin scenarios)
        """
        output_path = output_dir or self.output_folder
        mobility_dir = os.path.join(output_path, "mobility")
        os.makedirs(mobility_dir, exist_ok=True)

        # 1. Reuse existing mobility series (for Twin scenarios)
        if source_mobility_npz and os.path.exists(source_mobility_npz):
            logger.info(f"Reusing baseline mobility series from {source_mobility_npz}")
            target_npz_path = os.path.join(mobility_dir, "mobility_series.npz")
            shutil.copy2(source_mobility_npz, target_npz_path)

            # Write the baseline CSV for compatibility (simulator needs it)
            filename = "mobility_matrix.csv"
            csv_path = os.path.join(output_path, filename)
            self.mobility_df.to_csv(csv_path, index=False)

            return filename, csv_path, target_npz_path

        # 2. Generate new time-varying mobility
        mobility_sigma_O = profile.get("mobility_sigma_O", 0.0) if profile else 0.0
        mobility_sigma_D = profile.get("mobility_sigma_D", 0.0) if profile else 0.0

        if mobility_sigma_O > 0 or mobility_sigma_D > 0:
            # Generate time-varying mobility series
            logger.info(
                f"Generating time-varying mobility for {run_id}: "
                f"sigma_O={mobility_sigma_O:.3f}, sigma_D={mobility_sigma_D:.3f}"
            )

            # Load baseline mobility
            mobility_csv = self.base_config.get_param("data.mobility_matrix_filename")
            metapop_csv = self.base_config.get_param(
                "data.metapopulation_data_filename"
            )
            edgelist, R_baseline, M = load_baseline_mobility(
                os.path.join(self.data_folder, mobility_csv),
                os.path.join(self.data_folder, metapop_csv),
            )

            # Calculate T from dates
            start = pd.to_datetime(str(start_date))
            end = pd.to_datetime(str(end_date))
            T = (end - start).days + 1

            # Create mobility generator
            generator = MobilityGenerator(
                baseline_R=(edgelist, R_baseline),
                sigma_O=mobility_sigma_O,
                sigma_D=mobility_sigma_D,
                rng_seed=42,
            )

            # Generate mobility series
            R_series = generator.generate_series(T=T, rng_seed=hash(run_id))

            # Save to NPZ file
            mobility_path = os.path.join(mobility_dir, "mobility_series.npz")
            np.savez_compressed(
                mobility_path,
                R_series=R_series,
                edgelist=edgelist,
                T=T,
                E=len(R_baseline),
                M=M,
                sigma_O=mobility_sigma_O,
                sigma_D=mobility_sigma_D,
            )

            # Still write the baseline CSV for compatibility
            filename = "mobility_matrix.csv"
            csv_path = os.path.join(output_path, filename)
            self.mobility_df.to_csv(csv_path, index=False)

            logger.info(f"Saved mobility series to {mobility_path}")
            return filename, csv_path, mobility_path
        else:
            # Static mobility - just copy baseline
            scaled_df = self.mobility_df.copy()

            filename = "mobility_matrix.csv"
            path = os.path.join(output_path, filename)
            scaled_df.to_csv(path, index=False)
            return filename, path, None

    def prepare_seed_file(self, run_id, seed_size=10.0):
        """Generate random seed file"""
        # Pick a random region (row index)
        random_idx = np.random.randint(0, len(self.metapop_df))
        seed_region = self.metapop_df.iloc[random_idx]
        region_id = seed_region["id"]

        # Cap seed size to 10% of region population to prevent stability issues
        # Use .get with default or check column existence
        if "total" in seed_region:
            total_pop = seed_region["total"]
            # Warn if seed size is large (>50%), but allow it (normalization fixes stability)
            if seed_size > 0.5 * total_pop:
                logger.warning(
                    f"Run {run_id}: Seed size {seed_size:.0f} is >50% of pop {total_pop:.0f} "
                    f"in region {region_id}. This is physically valid but unusual."
                )

        # Look up the correct 1-based index from rosetta
        idx_row = self.rosetta_df[self.rosetta_df["id"] == region_id]
        if len(idx_row) == 0:
            raise ValueError(f"Region ID {region_id} not found in rosetta")
        idx = idx_row["idx"].iloc[0]

        # Create seed dataframe
        # We start with `seed_size` Exposed/Asymptomatic individuals in Middle age group (M)
        seed_data = {
            "name": ["SeedRegion"],
            "id": [region_id],
            "idx": [idx],
            "Y": [0.0],
            "M": [float(seed_size)],
            "O": [0.0],
        }

        seed_df = pd.DataFrame(seed_data)

        filename = f"seeds_{run_id}.csv"
        path = os.path.join(self.output_folder, filename)
        seed_df.to_csv(path, index=False)
        return filename, path

    def run_single_scenario(
        self,
        pid,
        scen_name,
        strength,
        profile,
        seed_path,
        run_suffix="",
        source_mobility_npz=None,
    ):
        """Helper to prepare files for a single configuration."""
        r0_scale = profile["r0_scale"]
        t_inf = profile["t_inf"]
        t_inc = profile["t_inc"]
        ratio_beta_a = profile.get("ratio_beta_a", 0.5)
        alpha_scale = profile.get("alpha_scale", 1.0)
        mu_scale = profile.get("mu_scale", 1.0)

        # Construct Run ID
        # Format: {pid}_{Scenario}_s{strength_int} where strength_int is percent
        str_pct = int(round(strength * 100))
        run_id = f"{pid}_{scen_name}{run_suffix}_s{str_pct:02d}"
        if scen_name == "Baseline":
            run_id = f"{pid}_Baseline"

        logger.info(f"Preparing {run_id} (Str={strength:.2f})")

        # Prepare Directory Structure FIRST
        # Use run_id as the folder name (no UUID nesting when using name parameter)
        model_state_folder = os.path.join(self.output_folder, f"run_{run_id}")
        if not os.path.exists(model_state_folder):
            os.makedirs(model_state_folder)

        # Update Config
        import copy

        config_dict = copy.deepcopy(self.base_config.config)
        config = EpiSimConfig(config_dict)

        # Get start/end dates from config
        start_date = config.get_param("simulation.start_date")
        end_date = config.get_param("simulation.end_date")

        gamma = 1.0 / t_inf  # Recovery rate (μᵍ)
        eta = 1.0 / t_inc  # E→I/A progression rate (ηᵍ)

        # Calculate derived params
        base_beta_I = config.get_param("epidemic_params.βᴵ")
        base_alpha = config.get_param("epidemic_params.αᵍ")

        new_beta_A = base_beta_I * ratio_beta_a
        new_alpha = [min(1.0, x * alpha_scale) for x in base_alpha]

        # Prepare Mobility - Write to run directory
        # All scenarios use base mobility matrix (interventions via κ₀ only)
        # If mobility sigma parameters are non-zero, generates time-varying mobility series
        mob_filename, mob_path, mobility_npz_path = self.prepare_mobility_file(
            run_id,
            "Global",
            0.0,
            0.0,
            output_dir=model_state_folder,
            profile=profile,
            start_date=start_date,
            end_date=end_date,
            source_mobility_npz=source_mobility_npz,
        )

        # NPI Params Setup - Always create CSV for ALL scenarios
        # Write to run directory
        _, kappa0_path = self.prepare_kappa0_file(
            run_id,
            scen_name,
            strength,
            profile,
            start_date,
            end_date,
            output_dir=model_state_folder,
        )

        # Inject updates
        # NOTE: μᵍ is the recovery rate (gamma), NOT γᵍ (hospitalization probability)
        updates = {
            "simulation.end_date": end_date,
            "simulation.save_full_output": False,
            "simulation.save_observables": True,
            "epidemic_params.scale_β": r0_scale,
            "epidemic_params.βᴬ": new_beta_A,
            "epidemic_params.αᵍ": new_alpha,
            "epidemic_params.μᵍ": [gamma] * 3,  # Recovery rate = 1/T_inf
            "epidemic_params.ηᵍ": [eta] * 3,  # E→I/A progression rate = 1/T_inc
            "data.mobility_matrix_filename": mob_path,
            "data.initial_condition_filename": seed_path,
            "NPI.are_there_npi": True,
        }

        config.inject(updates)

        # Always set kappa0_filename - CSV is now single source of truth for κ₀
        config.update_param("data.kappa0_filename", kappa0_path)

        # Write Config File
        config_path = os.path.join(model_state_folder, "config_auto_py.json")
        with open(config_path, "w") as f:
            json.dump(config.config, f, indent=4)

        logger.info(f"Prepared config at {config_path}")

        # Return the mobility NPZ path so it can be reused by Twin scenarios
        return mobility_npz_path

    def run_profile_sweep(
        self, profile, baseline_only=False, intervention_profiles=None
    ):
        """Prepare files for Baseline (and optionally interventions) for a profile.

        Args:
            profile: Profile dict with epidemiological parameters
            baseline_only: If True, skip intervention sweep entirely
            intervention_profiles: Set of profile IDs selected for interventions, or None for all
        """
        pid = profile["profile_id"]

        # VALIDATE before generating any configs
        if not self.validate_profile_parameters(profile):
            logger.warning(f"Skipping Profile {pid} due to validation failure")
            return

        seed_size = profile.get("seed_size", 10.0)
        logger.info(f"--- Processing Profile {pid} (Seed={seed_size}) ---")

        # 1. Generate Seed ONCE for this profile
        seed_fname, seed_path = self.prepare_seed_file(pid, seed_size=seed_size)

        # 2. Prepare Baseline
        # Capture the generated mobility NPZ path to reuse for Twin scenarios
        baseline_mobility_npz = self.run_single_scenario(
            pid, "Baseline", 0.0, profile, seed_path
        )

        # Skip intervention sweep if baseline-only mode or profile not selected
        if baseline_only:
            logger.info(
                f"Baseline-only mode: skipping intervention sweep for Profile {pid}"
            )
            return

        if intervention_profiles is not None and pid not in intervention_profiles:
            logger.info(f"Profile {pid} not selected for interventions (baseline only)")
            return

        # 3. Sweep - Test valid range of intervention strengths (0.05 to 0.8)
        # Note: kappa0 (κ₀) must be in range [0, 1] for model stability; we use 0.8 as practical upper bound
        # Note: 0.05 floor ensures all intervention scenarios have a meaningful (non-zero) intervention effect
        #       since 0.0 is equivalent to the Baseline scenario
        strengths = np.linspace(0.05, 0.8, 6)
        scenarios = ["Global_Timed"]

        for scen in scenarios:
            for s in strengths:
                # Pass baseline_mobility_npz to reuse the same mobility noise realization
                self.run_single_scenario(
                    pid,
                    scen,
                    s,
                    profile,
                    seed_path,
                    source_mobility_npz=baseline_mobility_npz,
                )

    def run_profile_sweep_parallel(
        self,
        profiles: list,
        n_jobs: int = 1,
        baseline_only: bool = False,
        intervention_profiles: set | None = None,
    ) -> list:
        """
        Process multiple profiles in parallel using multiprocessing.

        Args:
            profiles: List of profile dictionaries
            n_jobs: Number of parallel workers (1 = sequential)
            baseline_only: If True, skip intervention sweep
            intervention_profiles: Set of profile IDs for interventions (optional, defaults to all)

        Returns:
            List of result dictionaries from each profile
        """
        if n_jobs == 1:
            # Sequential fallback
            results = []
            for profile in profiles:
                result = generate_profile_standalone(
                    profile=profile,
                    base_config_path=self.base_config_path,
                    data_folder=self.data_folder,
                    output_folder=self.output_folder,
                    baseline_only=baseline_only,
                    intervention_profiles=intervention_profiles,
                    mobility_sigma_min=self.mobility_sigma_min,
                    mobility_sigma_max=self.mobility_sigma_max,
                )
                results.append(result)
            return results

        # Parallel execution with ProcessPoolExecutor
        logger.info(
            f"Processing {len(profiles)} profiles with {n_jobs} parallel workers..."
        )

        results = []
        failed_profiles = []

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all jobs
            future_to_profile = {}
            for profile in profiles:
                future = executor.submit(
                    generate_profile_standalone,
                    profile=profile,
                    base_config_path=self.base_config_path,
                    data_folder=self.data_folder,
                    output_folder=self.output_folder,
                    baseline_only=baseline_only,
                    intervention_profiles=intervention_profiles,
                    mobility_sigma_min=self.mobility_sigma_min,
                    mobility_sigma_max=self.mobility_sigma_max,
                )
                future_to_profile[future] = profile

            # Wait for completion and collect results
            for future in as_completed(future_to_profile):
                profile = future_to_profile[future]
                try:
                    result = future.result()
                    results.append(result)

                    if not result["success"]:
                        failed_profiles.append(profile["profile_id"])
                        logger.error(
                            f"Profile {profile['profile_id']} failed: {result['error']}"
                        )

                except Exception as e:
                    logger.error(f"Profile {profile['profile_id']} job crashed: {e}")
                    results.append(
                        {
                            "success": False,
                            "profile_id": profile["profile_id"],
                            "error": str(e),
                        }
                    )
                    failed_profiles.append(profile["profile_id"])

        # Log summary
        n_total = len(profiles)
        n_failed = len(failed_profiles)
        n_success = n_total - n_failed

        logger.info("=" * 60)
        logger.info(f"Profile Generation Summary: {n_success}/{n_total} succeeded")
        if n_failed > 0:
            logger.warning(f"Failed profiles: {sorted(failed_profiles)}")
        logger.info("=" * 60)

        return results

    def check_run_success(self, run_id):
        """Check if a run completed successfully by looking for output files."""
        run_folder = os.path.join(self.output_folder, f"run_{run_id}")
        output_folder = os.path.join(run_folder, "output")

        # Check for ERROR.json
        error_file = os.path.join(run_folder, "ERROR.json")
        if os.path.exists(error_file):
            return False, "ERROR.json found"

        # Check for output observables
        observables_path = os.path.join(output_folder, "observables.nc")
        if not os.path.exists(observables_path):
            return False, "Missing observables.nc"

        return True, None

    def _collect_results(self):
        """Collect run results from output folder."""
        succeeded = []
        failed = []

        for item in os.listdir(self.output_folder):
            if item.startswith("run_"):
                run_id = item[4:]  # Remove "run_" prefix
                success, reason = self.check_run_success(run_id)
                if success:
                    succeeded.append(run_id)
                else:
                    failed.append(run_id)
                    logger.warning(f"Run {run_id} failed: {reason}")

        return {"succeeded": succeeded, "failed": failed}

    def _cleanup_run(self, run_id):
        """Clean up a failed run directory after logging the failure."""
        run_folder = os.path.join(self.output_folder, f"run_{run_id}")
        if not os.path.exists(run_folder):
            return

        # Log failure before cleanup
        self._log_run_failure(run_id, run_folder)

        shutil.rmtree(run_folder)
        logger.info(f"Cleaned up {run_folder}")

    def _log_run_failure(self, run_id, run_folder):
        """Extract and log failure details from a failed run."""
        # Check for ERROR.json
        error_file = os.path.join(run_folder, "ERROR.json")
        error_type = "Unknown"
        error_message = ""
        stacktrace = ""

        if os.path.exists(error_file):
            try:
                with open(error_file) as f:
                    error_data = json.load(f)
                error_type = error_data.get("error_type", "Unknown")
                error_message = error_data.get("error_message", "")
                stacktrace = error_data.get("stacktrace", "")
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to read ERROR.json for {run_id}: {e}")

        # Extract profile from config
        config_file = os.path.join(run_folder, "config_auto_py.json")
        profile = {}
        if os.path.exists(config_file):
            try:
                with open(config_file) as f:
                    config = json.load(f)

                # Extract relevant parameters
                epi_params = config.get("epidemic_params", {})
                profile["r0_scale"] = epi_params.get("scale_β", 1.0)

                # Extract rates
                mu_g = epi_params.get("μᵍ", [0.2])[0]
                eta_g = epi_params.get("ηᵍ", [0.2])[0]
                profile["t_inf"] = 1.0 / mu_g if mu_g > 0 else 5.0
                profile["t_inc"] = 1.0 / eta_g if eta_g > 0 else 5.0

                # Extract other params
                profile["ratio_beta_a"] = (
                    epi_params.get("βᴬ", 0.5) / epi_params.get("βᴵ", 0.5)
                    if epi_params.get("βᴵ")
                    else 0.5
                )
                profile["alpha_scale"] = (
                    epi_params.get("αᵍ", [0.1])[0] / 0.1
                    if epi_params.get("αᵍ")
                    else 1.0
                )
                profile["seed_size"] = self._extract_seed_size_from_config(config)

            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to read config for {run_id}: {e}")

        # Log the failure
        self.failure_logger.log_failure(
            run_id=run_id,
            profile=profile,
            error_type=error_type,
            error_message=error_message,
            config_path=config_file if os.path.exists(config_file) else None,
            stacktrace=stacktrace,
        )

    def _extract_seed_size_from_config(self, config):
        """Extract seed size from config."""
        seed_filename = config.get("data", {}).get("initial_condition_filename", "")
        if seed_filename and os.path.exists(seed_filename):
            try:
                seed_df = pd.read_csv(seed_filename)
                return int(seed_df["M"].sum())
            except Exception:
                pass
        return 10  # Default

    def _retry_failed_runs(self, failed_runs, seed_offset):
        """Retry failed runs with different random seeds."""
        # Extract profile IDs from failed run IDs
        # Run ID format: {pid}_{Scenario}_s{strength}
        profile_ids = set()
        for run_id in failed_runs:
            parts = run_id.split("_")
            if parts:
                try:
                    profile_ids.add(int(parts[0]))
                except ValueError:
                    logger.warning(
                        f"Could not extract profile ID from run_id: {run_id}"
                    )
                    continue

        if not profile_ids:
            logger.warning("No valid profile IDs found in failed runs")
            return

        # Re-generate profiles with new seed
        logger.info(
            f"Re-generating profiles {profile_ids} with seed offset {seed_offset}"
        )
        # Use different seed for LHS generation
        new_seed = 42 + seed_offset * 1000
        new_profiles = self.generate_parameter_grid(
            n_profiles=max(profile_ids) + 1, seed=new_seed
        )

        # Re-run failed profiles
        for pid in profile_ids:
            profile = new_profiles[pid]
            # Validate profile before retry
            if not self.validate_profile_parameters(profile):
                logger.warning(f"Profile {pid} failed validation, skipping retry")
                continue
            self.run_profile_sweep(profile)

    def _run_batch_single(self):
        """Execute a single batch run (invoke the Julia batch runner)."""
        logger.info("Starting Batch Execution...")

        julia_path = "julia"  # Assume in path
        script_path = os.path.join(
            self.base_config_path, "..", "..", "..", "src", "batch_run.jl"
        )
        script_path = os.path.abspath(script_path)

        project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        cmd = [
            julia_path,
            "--project=" + project_path,
            "-t",
            "1",
            script_path,
            "--batch-folder",
            self.output_folder,
            "--data-folder",
            self.data_folder,
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            logger.info("Batch execution finished successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Batch execution failed: {e}")
            return False

    def run_batch_with_retry(self, failure_tolerance=10):
        """Run batch with retry logic for failed simulations."""
        logger.info("Starting Batch Execution with Retry Logic...")

        consecutive_failures = 0
        max_attempts = 3  # Max retries per run

        for attempt in range(max_attempts):
            if consecutive_failures >= failure_tolerance:
                logger.error(
                    f"Exceeded failure tolerance ({failure_tolerance} consecutive failures). Aborting."
                )
                return False

            logger.info(f"Attempt {attempt + 1}/{max_attempts}")
            self._run_batch_single()

            # Check results
            results = self._collect_results()
            successful = results["succeeded"]
            failed_runs = results["failed"]

            logger.info(
                f"Results: {len(successful)} succeeded, {len(failed_runs)} failed"
            )

            if not failed_runs:
                logger.info("All runs succeeded!")
                return True

            # Retry failed runs with new random seeds
            consecutive_failures = len(failed_runs)
            logger.warning(
                f"{consecutive_failures} runs failed. Retrying with new seeds..."
            )

            # Clean up failed runs and retry
            for run_id in failed_runs:
                self._cleanup_run(run_id)

            # Generate new profiles for retry (different random seed)
            # Use attempt + 1 as seed offset
            self._retry_failed_runs(failed_runs, attempt + 1)

        # Final scan for any remaining failures (should be empty if all retries succeeded)
        self.failure_logger.load_failures()
        if self.failure_logger._failures:
            logger.info("=" * 60)
            logger.info(
                f"FAILURE SUMMARY: {len(self.failure_logger._failures)} failures logged"
            )
            logger.info(f"Failure log: {self.failure_logger.log_file}")
            logger.info("Run with --analyze-failures to see detailed analysis")
            logger.info("=" * 60)

        return False

    def print_failure_analysis(self):
        """Print analysis of logged failures."""
        self.failure_logger.load_failures()
        self.failure_logger.print_failure_summary()

        suggestions = self.failure_logger.suggest_validation_rules()
        if suggestions:
            print("\nSUGGESTED VALIDATION RULES:")
            for suggestion in suggestions:
                print(f"  - {suggestion}")

    def run_spike_based_interventions(
        self,
        baseline_dir,
        spike_threshold=0.1,
        min_duration=7,
        spike_method="percentile",
        growth_factor_threshold=1.5,
        min_growth_duration=3,
        min_cases_per_capita=1e-4,
        max_intervention_duration=90,
        intervention_profiles=None,
    ):
        """
        Generate intervention scenarios based on detected spikes in baseline outputs.

        This method implements Phase 2 of the two-phase pipeline:
        1. Detects infection spikes in processed baseline zarr file
        2. Generates intervention scenarios with realistic timing based on spikes

        Args:
            baseline_dir: Directory containing baseline run outputs and zarr
            spike_threshold: Percentile threshold for spike detection (default: 0.1)
            min_duration: Minimum days for spike period (default: 7)
            spike_method: Spike detection method (default: "percentile")
            growth_factor_threshold: Growth factor threshold for growth_rate (default: 1.5)
            min_growth_duration: Minimum consecutive days of growth for growth_rate (default: 3)
            min_cases_per_capita: Minimum cases per person for growth_rate (default: 1e-4)
            max_intervention_duration: Maximum intervention duration in days (default: 90)
            intervention_profiles: Set of profile IDs selected for interventions, or None for all

        Raises:
            FileNotFoundError: If baseline zarr or configs don't exist
            ValueError: If no baselines found or spike detection fails
        """
        from spike_detector import detect_spike_periods_from_zarr

        # Validate baseline directory exists
        # The zarr file is in the parent directory of baseline_dir (output_base)
        # because it contains both baselines AND interventions (interventions are appended)
        baseline_zarr = os.path.join(
            os.path.dirname(baseline_dir), "raw_synthetic_observations.zarr"
        )
        if not os.path.exists(baseline_dir):
            raise FileNotFoundError(
                f"Baseline directory not found: {baseline_dir}\n"
                f"Please run baselines first with --baseline-only"
            )
        if not os.path.exists(baseline_zarr):
            raise FileNotFoundError(
                f"Baseline zarr not found: {baseline_zarr}\n"
                f"Please process baselines first:\n"
                f"  uv run python synthetic_generator.py --baseline-only --output-folder {os.path.dirname(baseline_dir)}\n"
                f"  uv run python process_synthetic_outputs.py --runs-dir {baseline_dir} --output {baseline_zarr}"
            )

        logger.info("=" * 60)
        logger.info("PHASE 2: Spike-Based Intervention Generation")
        logger.info("=" * 60)
        logger.info(f"Baseline directory: {baseline_dir}")
        logger.info(f"Spike method: {spike_method}")
        logger.info(f"Spike threshold: {spike_threshold} (percentile)")
        logger.info(f"Min spike duration: {min_duration} days")
        logger.info(f"Max intervention duration: {max_intervention_duration} days")
        if spike_method == "growth_rate":
            logger.info(f"Growth factor threshold: {growth_factor_threshold}")
            logger.info(f"Min growth duration: {min_growth_duration}")
            logger.info(f"Min cases per capita: {min_cases_per_capita}")

        # Load baseline zarr and detect spikes per profile
        spikes = detect_spike_periods_from_zarr(
            baseline_zarr,
            threshold_pct=spike_threshold,
            min_duration=min_duration,
            method=spike_method,
            baseline_filter=True,  # Only process Baseline scenarios
            growth_factor_threshold=growth_factor_threshold,
            min_growth_duration=min_growth_duration,
            min_cases_per_capita=min_cases_per_capita,
        )

        logger.info(f"Detected spikes in {len(spikes)} baseline runs")

        # Track generated scenarios
        scenarios_generated = 0

        # Generate intervention scenarios using detected spike windows
        for run_id, spike_windows in spikes.items():
            # Strip whitespace from run_id (zarr may have trailing spaces)
            run_id = run_id.strip()
            # Extract profile ID from run_ID_Baseline format
            # run_id format: "0_Baseline", "1_Baseline", etc.
            parts = run_id.split("_")
            if len(parts) < 2 or parts[1] != "Baseline":
                logger.warning(f"Skipping non-baseline run_id: {run_id}")
                continue

            try:
                profile_id = int(parts[0])
            except ValueError:
                logger.warning(f"Could not extract profile ID from run_id: {run_id}")
                continue

            # Skip if profile not selected for interventions
            if (
                intervention_profiles is not None
                and profile_id not in intervention_profiles
            ):
                logger.info(
                    f"Profile {profile_id} not selected for spike-based interventions"
                )
                continue

            # Load original profile parameters from baseline config
            baseline_run_dir = os.path.join(baseline_dir, f"run_{run_id}")
            baseline_config_path = os.path.join(baseline_run_dir, "config_auto_py.json")

            if not os.path.exists(baseline_config_path):
                logger.warning(f"Baseline config not found: {baseline_config_path}")
                continue

            # Check for existing mobility series in Baseline
            # If found, we will COPY it to the intervention runs to ensure "Twin" scenarios
            # have the exact same mobility noise realization.
            baseline_mobility_npz = os.path.join(
                baseline_run_dir, "mobility", "mobility_series.npz"
            )
            source_npz = None
            if os.path.exists(baseline_mobility_npz):
                source_npz = baseline_mobility_npz
                logger.info(f"Detected baseline mobility series: {source_npz}")

            with open(baseline_config_path) as f:
                baseline_config = json.load(f)

            # Generate profile dict from baseline config
            # Extract parameters from config to reconstruct profile
            profile = self._extract_profile_from_config(baseline_config, profile_id)

            # Regenerate seed for this profile (same random seed for reproducibility)
            seed_size = profile.get("seed_size", 10.0)
            _, seed_path = self.prepare_seed_file(profile_id, seed_size=seed_size)

            logger.info(
                f"--- Profile {profile_id}: Generating {len(spike_windows)} spike-based interventions ---"
            )

            # For each spike window, generate intervention scenarios at different strengths
            strengths = np.linspace(0.05, 0.8, 6)
            for spike_idx, (spike_start, spike_end) in enumerate(spike_windows):
                spike_duration = spike_end - spike_start

                # Cap the intervention duration to max_intervention_duration
                capped_duration = min(spike_duration, max_intervention_duration)
                if spike_duration != capped_duration:
                    logger.warning(
                        f"  Spike {spike_idx + 1}: Duration capped from {spike_duration}d to {capped_duration}d"
                    )

                logger.info(
                    f"  Spike {spike_idx + 1}: Day {spike_start} -> {spike_end} (duration={spike_duration}d, capped={capped_duration}d)"
                )

                # Add suffix if multiple spikes to avoid overwriting folders
                suffix = ""
                if len(spike_windows) > 1:
                    suffix = f"_spike{spike_idx + 1}"

                for strength in strengths:
                    # Create modified profile with spike-based timing
                    spike_profile = profile.copy()
                    spike_profile["event_start"] = spike_start
                    spike_profile["event_duration"] = capped_duration

                    # Generate scenario
                    self.run_single_scenario(
                        profile_id,
                        "Global_Timed",
                        strength,
                        spike_profile,
                        seed_path,
                        run_suffix=suffix,
                        source_mobility_npz=source_npz,
                    )
                    scenarios_generated += 1

        logger.info(f"Generated {scenarios_generated} intervention scenarios")
        return scenarios_generated

    def _extract_profile_from_config(self, config, profile_id):
        """Extract profile parameters from a generated config file."""
        # Extract epidemic parameters
        epidemic_params = config.get("epidemic_params", {})
        npi_params = config.get("NPI", {})

        # Calculate derived parameters
        beta_I = epidemic_params.get("βᴵ", 0.5)
        scale_beta = epidemic_params.get("scale_β", 1.0)
        r0_scale = scale_beta  # Approximation

        # Extract recovery rate (μᵍ) and progression rate (ηᵍ)
        mu_g = epidemic_params.get("μᵍ", [0.2])[0]  # Recovery rate
        eta_g = epidemic_params.get("ηᵍ", [0.2])[0]  # Progression rate

        t_inf = 1.0 / mu_g if mu_g > 0 else 5.0
        t_inc = 1.0 / eta_g if eta_g > 0 else 5.0

        # Extract other parameters
        ratio_beta_a = (
            epidemic_params.get("βᴬ", beta_I * 0.5) / beta_I if beta_I > 0 else 0.5
        )
        alpha_scale = (
            epidemic_params.get("αᵍ", [0.1])[0] / 0.1
            if epidemic_params.get("αᵍ")
            else 1.0
        )

        # Load seed file to get seed_size
        seed_filename = config.get("data", {}).get("initial_condition_filename", "")
        seed_size = 10.0  # Default

        if seed_filename and os.path.exists(seed_filename):
            try:
                seed_df = pd.read_csv(seed_filename)
                # Seed size is sum of M (middle age group) column
                seed_size = seed_df["M"].sum()
            except Exception:
                pass

        return {
            "profile_id": profile_id,
            "r0_scale": r0_scale,
            "t_inf": t_inf,
            "t_inc": t_inc,
            "event_start": 20,  # Placeholder, will be overridden by spike timing
            "event_duration": 30,  # Placeholder
            "affected_fraction": 0.3,
            "ratio_beta_a": ratio_beta_a,
            "alpha_scale": alpha_scale,
            "mu_scale": 1.0,
            "seed_size": int(seed_size),
        }

    def _sample_intervention_profiles(self, n_profiles, fraction, seed=42):
        """Sample profile IDs to receive interventions.

        Args:
            n_profiles: Total number of profiles
            fraction: Fraction of profiles to select (0.0 to 1.0)
            seed: Random seed for reproducibility

        Returns:
            Set of selected profile IDs
        """
        if fraction >= 1.0:
            return set(range(n_profiles))  # All profiles

        if fraction <= 0.0:
            return set()  # No interventions

        rng = np.random.default_rng(seed)
        n_selected = max(1, int(n_profiles * fraction))  # At least 1 if fraction > 0
        selected = rng.choice(n_profiles, n_selected, replace=False)
        return set(selected)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic data for EpiSim")
    parser.add_argument("--config", default=None, help="Path to base config json")
    parser.add_argument("--data-folder", default=None, help="Path to data folder")
    parser.add_argument("--output-folder", default=None, help="Path to output folder")
    parser.add_argument(
        "--n-profiles",
        type=int,
        default=15,
        help="Total number of profiles to generate",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel workers for profile generation (default: 1)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index for processing (inclusive)",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End index for processing (exclusive)",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean output folder before running"
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running the simulation (generation only)",
    )
    parser.add_argument(
        "--failure-tolerance",
        type=int,
        default=10,
        help="Number of consecutive failures before aborting (default: 10)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Generate baseline scenarios only (no intervention sweep)",
    )
    parser.add_argument(
        "--intervention-only",
        type=str,
        metavar="BASELINE_DIR",
        help="Generate intervention scenarios based on spike analysis of baseline runs in BASELINE_DIR",
    )
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=0.1,
        help="Spike detection threshold percentile for --intervention-only (default: 0.1)",
    )
    parser.add_argument(
        "--min-spike-duration",
        type=int,
        default=7,
        help="Minimum days for spike period for --intervention-only (default: 7)",
    )
    parser.add_argument(
        "--spike-method",
        type=str,
        default="percentile",
        choices=["percentile", "prominence", "growth_rate"],
        help="Spike detection method for --intervention-only (default: percentile)",
    )
    parser.add_argument(
        "--growth-factor-threshold",
        type=float,
        default=1.5,
        help="Growth factor threshold for growth_rate method (default: 1.5 = 50%% growth)",
    )
    parser.add_argument(
        "--min-growth-duration",
        type=int,
        default=3,
        help="Minimum consecutive days of growth for growth_rate method (default: 3)",
    )
    parser.add_argument(
        "--min-cases-per-capita",
        type=float,
        default=1e-4,
        help="Minimum cases per person for growth_rate method (default: 1e-4 = 1 per 10K)",
    )
    parser.add_argument(
        "--max-intervention-duration",
        type=int,
        default=90,
        help="Maximum intervention duration in days for spike-based interventions (default: 90)",
    )
    parser.add_argument(
        "--analyze-failures",
        action="store_true",
        help="Analyze logged failures and exit (don't run simulations)",
    )
    parser.add_argument(
        "--scan-failures",
        type=str,
        metavar="BATCH_FOLDER",
        help="Scan batch folder for ERROR.json files and log them",
    )
    parser.add_argument(
        "--intervention-profile-fraction",
        type=float,
        default=1.0,
        help="Fraction of profiles that receive full intervention sweep (default: 1.0 = all). "
        "All profiles always generate baselines. Selected profiles get 6 intervention strengths. "
        "Uses fixed seed for reproducible profile selection.",
    )
    parser.add_argument(
        "--intervention-seed",
        type=int,
        default=42,
        help="Random seed for profile sampling (default: 42).",
    )
    parser.add_argument(
        "--mobility-sigma-min",
        type=float,
        default=0.0,
        help="Minimum mobility sigma for origin/destination noise (default: 0.0).",
    )
    parser.add_argument(
        "--mobility-sigma-max",
        type=float,
        default=0.6,
        help="Maximum mobility sigma for origin/destination noise (default: 0.6).",
    )

    args = parser.parse_args()

    # Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    DATA_FOLDER = args.data_folder
    if not DATA_FOLDER:
        DATA_FOLDER = os.path.join(PROJECT_ROOT, "models", "mitma")

    CONFIG_PATH = args.config
    if not CONFIG_PATH:
        CONFIG_PATH = os.path.join(DATA_FOLDER, "config_MMCACovid19.json")

    OUTPUT_FOLDER = args.output_folder
    if not OUTPUT_FOLDER:
        OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "runs", "synthetic_test")

    logger.info(f"Project Root: {PROJECT_ROOT}")

    # Clean previous runs if requested
    if args.clean and os.path.exists(OUTPUT_FOLDER):
        logger.info(f"Cleaning output folder {OUTPUT_FOLDER}")
        shutil.rmtree(OUTPUT_FOLDER)

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    generator = SyntheticDataGenerator(CONFIG_PATH, DATA_FOLDER, OUTPUT_FOLDER)
    generator.mobility_sigma_min = args.mobility_sigma_min
    generator.mobility_sigma_max = args.mobility_sigma_max

    # Handle failure scanning mode (standalone utility)
    if args.scan_failures:
        batch_folder = args.scan_failures
        logger.info(f"Scanning {batch_folder} for failures...")
        count = scan_and_log_existing_failures(batch_folder, generator.failure_logger)
        logger.info(f"Logged {count} failures")
        generator.print_failure_analysis()
        sys.exit(0)

    # Handle failure analysis mode (standalone utility)
    if args.analyze_failures:
        logger.info(f"Analyzing failures from {OUTPUT_FOLDER}")
        generator.print_failure_analysis()
        sys.exit(0)

    # Handle intervention-only mode (Phase 2 of two-phase pipeline)
    if args.intervention_only:
        baseline_dir = args.intervention_only

        logger.info("=" * 60)
        logger.info("INTERVENTION-ONLY MODE: Generating spike-based interventions")
        logger.info("=" * 60)
        logger.info(f"Baseline directory: {baseline_dir}")

        # Sample profiles for interventions (for consistency with Phase 1)
        intervention_profiles = None
        if args.intervention_profile_fraction < 1.0:
            intervention_profiles = generator._sample_intervention_profiles(
                args.n_profiles,
                args.intervention_profile_fraction,
                args.intervention_seed,
            )
            logger.info(
                f"Selected {len(intervention_profiles)}/{args.n_profiles} profiles for interventions"
            )

        # Create marker only if interventions will actually be generated
        # This prevents false positives when intervention_profile_fraction = 0
        if intervention_profiles is None or len(intervention_profiles) > 0:
            marker_path = os.path.join(OUTPUT_FOLDER, ".interventions_pending")
            with open(marker_path, "w") as f:
                f.write("")

        # Generate spike-based intervention scenarios
        generator.run_spike_based_interventions(
            baseline_dir=baseline_dir,
            spike_threshold=args.spike_threshold,
            min_duration=args.min_spike_duration,
            spike_method=args.spike_method,
            growth_factor_threshold=args.growth_factor_threshold,
            min_growth_duration=args.min_growth_duration,
            min_cases_per_capita=args.min_cases_per_capita,
            max_intervention_duration=args.max_intervention_duration,
            intervention_profiles=intervention_profiles,
        )

        # Execute batch if not skipped
        if not args.skip_run:
            generator.run_batch_with_retry(failure_tolerance=args.failure_tolerance)

        logger.info("Intervention-only generation complete!")
        sys.exit(0)

    # Standard mode or baseline-only mode (Phase 1 of two-phase pipeline)
    mode = (
        "BASELINE-ONLY" if args.baseline_only else "STANDARD (Baseline + Interventions)"
    )
    logger.info(f"Mode: {mode}")

    # Sample profiles for interventions
    intervention_profiles = None
    if args.intervention_profile_fraction < 1.0:
        intervention_profiles = generator._sample_intervention_profiles(
            args.n_profiles, args.intervention_profile_fraction, args.intervention_seed
        )
        logger.info(
            f"Selected {len(intervention_profiles)}/{args.n_profiles} profiles for interventions"
        )

    # Generate profiles for intervention timing analysis
    # We always generate the FULL set to maintain Latin Hypercube properties
    profiles = generator.generate_parameter_grid(
        n_profiles=args.n_profiles,
        mobility_sigma_min=args.mobility_sigma_min,
        mobility_sigma_max=args.mobility_sigma_max,
    )

    # Determine subset to process
    start_idx = args.start_index
    end_idx = args.end_index if args.end_index is not None else args.n_profiles

    # Clip to bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(profiles), end_idx)

    logger.info(
        f"Processing profiles {start_idx} to {end_idx} (Total: {len(profiles)})"
    )

    # Determine n_jobs
    n_jobs = args.n_jobs
    if n_jobs == -1:
        n_jobs = min(len(profiles) - start_idx, end_idx - start_idx, 45)

    # Prepare files for selected profiles
    selected_profiles = profiles[start_idx:end_idx]

    if n_jobs > 1:
        # Parallel execution
        results = generator.run_profile_sweep_parallel(
            profiles=selected_profiles,
            n_jobs=n_jobs,
            baseline_only=args.baseline_only,
            intervention_profiles=intervention_profiles,
        )

        # Check for failures
        failed_count = sum(1 for r in results if not r["success"])
        if failed_count > args.failure_tolerance:
            logger.error(
                f"Too many profile generation failures ({failed_count}), aborting"
            )
            sys.exit(1)
    else:
        # Sequential execution (original behavior)
        for i in range(len(selected_profiles)):
            profile = selected_profiles[i]
            generator.run_profile_sweep(
                profile,
                baseline_only=args.baseline_only,
                intervention_profiles=intervention_profiles,
            )

    # Execute batch if not skipped
    if not args.skip_run and start_idx < end_idx:
        generator.run_batch_with_retry(failure_tolerance=args.failure_tolerance)
