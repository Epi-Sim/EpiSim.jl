"""
Time-Varying Mobility Test for EpiSim.jl

This test validates that time-varying mobility using external NetCDF files
works correctly by demonstrating that:
1. Infections seeded in region A spread to connected regions via mobility
2. When mobility drops to zero at a specific timestep, infections are contained
3. Static mobility allows continued spread to all connected regions

Test Scenario:
- Chain Topology: Region A -> Region B -> Region C -> Region D
- Phase 1 (Days 1-9): Normal mobility (90% self-loop, 10% flow to next region)
- Phase 2 (Day 10+): Zero mobility (100% self-loop, diagonal matrix)

Note: The full simulation tests are skipped by default due to Julia launcher
network requirements. To run them, ensure you have network access and run:
    pytest python/tests/test_time_varying_mobility.py -v --run-slow
"""

import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

# Add python directory to path to import episim_python
sys_path_append = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if sys_path_append not in __import__("sys").path:
    __import__("sys").path.append(sys_path_append)

from episim_python.epi_sim import EpiSim


class TestTimeVaryingMobility:
    """Test class for time-varying mobility using external NetCDF files"""

    # Test constants
    NUM_REGIONS = 4
    SIMULATION_DAYS = 30
    MOBILITY_DROP_DAY = 10
    SEED_REGION = "A"
    SEED_SIZE = 50.0

    # Region names and IDs
    REGION_NAMES = ["Region_A", "Region_B", "Region_C", "Region_D"]
    REGION_IDS = ["00000", "00001", "00002", "00003"]
    AGE_GROUPS = ["Y", "M", "O"]

    # Path to actual Julia binary (not launcher) - avoid network access issues
    JULIA_BINARY = "/Users/lewis/.julia/juliaup/julia-1.11.5+0.aarch64.apple.darwin14/bin/julia"

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for test data"""
        temp_dir = tempfile.mkdtemp(prefix="time_varying_mobility_")
        yield temp_dir
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    def _create_mobility_netcdf(self, workspace, mode="containment"):
        """
        Create NetCDF file with time-varying mobility data.

        Dimensions: (date, origin, destination)
        Variable: mobility (probability)

        Args:
            workspace: Path to workspace directory
            mode: Mobility mode - "containment", "static", "shutdown", "gradual"

        Returns:
            Path to the created NetCDF file
        """
        M = self.NUM_REGIONS
        T = self.SIMULATION_DAYS

        # Create chain topology: A->B->C->D
        # Edges: (origin, destination) pairs
        edgelist = []
        for i in range(M):
            for j in range(M):
                edgelist.append([i, j])
        edgelist = np.array(edgelist, dtype=np.int64)

        # Generate mobility series for each day
        mobility_data = np.zeros((T, M, M), dtype=np.float64)

        for t in range(T):
            if mode == "containment":
                # Normal mobility until day 10, then zero off-diagonal
                if t < self.MOBILITY_DROP_DAY:
                    # 90% stay, 10% move forward in chain
                    # Last region in chain has 100% self-loop (no forward destination)
                    for i in range(M):
                        for j in range(M):
                            if i == j:
                                # Last region stays 100%, others 90%
                                mobility_data[t, i, j] = 1.0 if i == M - 1 else 0.90
                            elif j == i + 1:  # Forward in chain
                                mobility_data[t, i, j] = 0.10
                            else:
                                mobility_data[t, i, j] = 0.0
                else:
                    # Zero mobility: diagonal only
                    for i in range(M):
                        for j in range(M):
                            mobility_data[t, i, j] = 1.0 if i == j else 0.0

            elif mode == "static":
                # Constant mobility throughout
                # Last region in chain has 100% self-loop (no forward destination)
                for i in range(M):
                    for j in range(M):
                        if i == j:
                            mobility_data[t, i, j] = 1.0 if i == M - 1 else 0.90
                        elif j == i + 1:
                            mobility_data[t, i, j] = 0.10
                        else:
                            mobility_data[t, i, j] = 0.0

            elif mode == "shutdown":
                # Zero mobility from day 1
                for i in range(M):
                    for j in range(M):
                        mobility_data[t, i, j] = 1.0 if i == j else 0.0

            elif mode == "gradual":
                # Gradually reducing mobility (100% -> 0%)
                factor = max(0.0, 1.0 - (t / T))
                for i in range(M):
                    for j in range(M):
                        if i == j:
                            # Last region always stays 100%, others gradually increase to 100%
                            mobility_data[t, i, j] = 1.0 if i == M - 1 else 0.90 + 0.10 * (1 - factor)
                        elif j == i + 1:
                            mobility_data[t, i, j] = 0.10 * factor
                        else:
                            mobility_data[t, i, j] = 0.0

        # Create date range
        start_date = datetime(2020, 3, 1)
        dates = [start_date + timedelta(days=t) for t in range(T)]
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]

        # Create xarray Dataset
        # NOTE: Julia's NCDatasets.jl reads NetCDF dimensions in REVERSE order.
        # When we write (date, origin, destination), Julia reads (destination, origin, date).
        # Workaround: Write data in reverse order so Julia reads it correctly.
        # Transpose (T, M, M) -> (M, M, T) and use reversed dimension names.
        mobility_transposed = mobility_data.transpose(2, 1, 0)  # (dest, origin, date)

        ds = xr.Dataset(
            {
                "mobility": (["destination", "origin", "date"], mobility_transposed)
            },
            coords={
                "date": (["date"], date_strs),
                "origin": (["origin"], np.array(self.REGION_IDS)),
                "destination": (["destination"], np.array(self.REGION_IDS)),
            }
        )

        # Add attributes
        ds["mobility"].attrs = {
            "description": "Mobility flow probability",
            "units": "probability"
        }

        # Save to NetCDF
        output_path = os.path.join(workspace, f"mobility_{mode}.nc")
        ds.to_netcdf(output_path)

        return output_path

    def _create_seed_file(self, workspace):
        """
        Create seed CSV file with infections only in region A.

        Format: name,id,idx,Y,M,O
        """
        seed_path = os.path.join(workspace, "seeds.csv")

        with open(seed_path, "w") as f:
            f.write("name,id,idx,Y,M,O\n")
            # Seed infections in region A (middle age group)
            f.write(f"{self.REGION_NAMES[0]},{self.REGION_IDS[0]},1,0.0,{self.SEED_SIZE},0.0\n")

        return seed_path

    def _create_metapopulation_csv(self, workspace):
        """
        Create minimal metapopulation CSV file.

        Format: id,area,Y,M,O,total
        """
        meta_path = os.path.join(workspace, "metapopulation_data.csv")

        with open(meta_path, "w") as f:
            f.write("id,area,Y,M,O,total\n")
            for i, (name, region_id) in enumerate(zip(self.REGION_NAMES, self.REGION_IDS)):
                # Equal population in all regions
                y = 5000
                m = 8000
                o = 3000
                total = y + m + o
                area = 100.0 + i * 10.0
                f.write(f"{region_id},{area},{y},{m},{o},{total}\n")

        return meta_path

    def _create_contact_matrix_csv(self, workspace):
        """
        Create contact matrix CSV file.

        Format: G_labels are rows, columns are implicit
        """
        contact_path = os.path.join(workspace, "contact_matrices_data.csv")

        G = len(self.AGE_GROUPS)
        # Simple contact matrix
        C = np.array([
            [0.6, 0.3, 0.1],
            [0.3, 0.5, 0.2],
            [0.1, 0.2, 0.7],
        ])

        with open(contact_path, "w") as f:
            f.write("G_labels," + ",".join(self.AGE_GROUPS) + "\n")
            for i, g in enumerate(self.AGE_GROUPS):
                f.write(g + "," + ",".join(map(str, C[i])) + "\n")

        return contact_path

    def _create_mobility_csv(self, workspace):
        """
        Create baseline mobility CSV file (required for compatibility).

        Format: source_idx,target_idx,ratio
        """
        mobility_path = os.path.join(workspace, "mobility_matrix.csv")

        M = self.NUM_REGIONS

        with open(mobility_path, "w") as f:
            f.write("source_idx,target_idx,ratio\n")
            for i in range(M):
                for j in range(M):
                    # Julia uses 1-based indexing, so convert Python's 0-based to 1-based
                    if i == j:
                        f.write(f"{i+1},{j+1},100.0\n")
                    elif j == i + 1:
                        f.write(f"{i+1},{j+1},10.0\n")
                    else:
                        f.write(f"{i+1},{j+1},0.0\n")

        return mobility_path

    def _create_config(self, workspace, mobility_netcdf_path, run_name):
        """
        Create EpiSim configuration dictionary.

        Args:
            workspace: Path to workspace directory
            mobility_netcdf_path: Path to external mobility NetCDF file
            run_name: Name for this run
        """
        config = {
            "simulation": {
                "engine": "MMCACovid19Vac",  # Required for external_netcdf support
                "start_date": "2020-03-01",
                "end_date": f"2020-03-{self.SIMULATION_DAYS:02d}",
                "save_full_output": True,  # Use full output to avoid NetCDF observables bug
                "save_observables": True,
                "save_time_step": -1,
                "input_format": "csv",
                "output_folder": "output",
                "output_format": "netcdf"
            },
            "data": {
                "initial_condition_filename": "seeds.csv",
                "metapopulation_data_filename": "metapopulation_data.csv",
                "mobility_matrix_filename": "mobility_matrix.csv",
            },
            "epidemic_params": {
                # High R0 for visible spread within 30 days
                "scale_β": 2.0,
                "βᴵ": 0.09,
                "βᴬ": 0.045,
                "ηᵍ": [0.27, 0.27, 0.27],
                "αᵍ": [0.26, 0.64, 0.64],
                "μᵍ": [1.0, 0.31, 0.31],
                "θᵍ": [0.0, 0.0, 0.0],
                "γᵍ": [0.003, 0.01, 0.08],
                "ζᵍ": [0.128, 0.128, 0.128],
                "λᵍ": [1.0, 1.0, 1.0],
                "ωᵍ": [0.0, 0.04, 0.3],
                "ψᵍ": [0.143, 0.143, 0.143],
                "χᵍ": [0.048, 0.048, 0.048],
                "Λ": 0.02,
                "Γ": 0.01,
                "risk_reduction_dd": 0.0,
                "risk_reduction_h": 0.1,
                "risk_reduction_d": 0.05,
                # Vaccination parameters (required even if not used)
                "rᵥ": [0.0, 0.6, 0.0],
                "kᵥ": [0.0, 0.4, 0.0],
            },
            "population_params": {
                "G_labels": self.AGE_GROUPS,
                "C": [
                    [0.6, 0.3, 0.1],
                    [0.3, 0.5, 0.2],
                    [0.1, 0.2, 0.7],
                ],
                "kᵍ": [12.0, 13.0, 7.0],
                "kᵍ_h": [3.0, 3.0, 3.0],
                "kᵍ_w": [2.0, 5.0, 0.0],
                "pᵍ": [0.0, 1.0, 0.0],
                "ξ": 0.01,
                "σ": 2.5,
                # External NetCDF mobility configuration
                "mobility_variation_type": "external_netcdf",
                "mobility_external_file": os.path.abspath(mobility_netcdf_path),
                "mobility_external_variable": "mobility",
                "mobility_validate": True,
                "mobility_fill_missing": False,
                "mobility_time_alignment": "start",
            },
            "NPI": {
                "κ₀s": [0.0],  # No NPI to isolate mobility effects
                "ϕs": [0.2],
                "δs": [0.8],
                "tᶜs": [100],
                "are_there_npi": True,
            },
            "vaccination": {
                "ϵᵍ": [0.1, 0.4, 0.5],
                "percentage_of_vacc_per_day": 0.0,
                "start_vacc": 0,
                "dur_vacc": 0,
                "are_there_vaccines": False  # No actual vaccination for this test
            }
        }

        return config

    def _run_simulation(self, config, workspace, run_name):
        """
        Run EpiSim simulation using Python wrapper.

        Args:
            config: Configuration dictionary
            workspace: Path to workspace directory
            run_name: Name for this run

        Returns:
            Dictionary with simulation results
        """
        # Create instance folder
        instance_folder = os.path.join(workspace, "instances")
        os.makedirs(instance_folder, exist_ok=True)

        # JULIA_PROJECT: Point to the EpiSim.jl project
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        os.environ["JULIA_PROJECT"] = project_root

        # Initialize EpiSim model
        model = EpiSim(
            config=config,
            data_folder=workspace,
            instance_folder=instance_folder,
            initial_conditions=None,
            name=run_name,
        )

        # Setup model with actual Julia binary (not launcher) to avoid network access
        model.setup(executable_type="interpreter")

        # Override with actual Julia binary path to bypass julialauncher
        script_path = os.path.join(project_root, "src", "run.jl")
        model.executable_path = [self.JULIA_BINARY, script_path]

        # Run simulation
        start_date = config["simulation"]["start_date"]
        final_state, next_date = model.step(start_date, length_days=self.SIMULATION_DAYS)

        # Read observables NetCDF to get infections by region
        output_path = os.path.join(os.path.dirname(final_state), "observables.nc")

        ds = xr.open_dataset(output_path)

        # new_infected dimensions: (G, M, T) - sum over age groups and time
        infections_by_region = ds["new_infected"].sum(dim="G").sum(dim="T").values
        total_infections = ds["new_infected"].sum().values

        ds.close()

        return {
            "infections_by_region": infections_by_region,
            "total_infections": float(total_infections),
            "output_path": output_path,
        }

    def test_mobility_containment(self, temp_workspace):
        """
        Main test: Compare time-varying mobility vs static mobility.

        Expected behavior:
        1. Region A (seed) has infections in both scenarios
        2. Region B has infections in both (connected before cutoff)
        3. Region C has MORE infections in static case
        4. Region D has infections ONLY in static case
        5. Total infections: time-varying <= static (containment effect)
        """
        # Create test data files
        mobility_containment_path = self._create_mobility_netcdf(temp_workspace, mode="containment")
        mobility_static_path = self._create_mobility_netcdf(temp_workspace, mode="static")
        seed_path = self._create_seed_file(temp_workspace)
        metapop_path = self._create_metapopulation_csv(temp_workspace)
        mobility_csv_path = self._create_mobility_csv(temp_workspace)

        # Create configs
        config_containment = self._create_config(temp_workspace, mobility_containment_path, "time_varying")
        config_static = self._create_config(temp_workspace, mobility_static_path, "static")

        # Run simulations
        print("\n=== Running Time-Varying Mobility Simulation ===")
        results_tv = self._run_simulation(config_containment, temp_workspace, "time_varying")

        print("\n=== Running Static Mobility Simulation ===")
        results_static = self._run_simulation(config_static, temp_workspace, "static")

        infections_tv = results_tv["infections_by_region"]
        infections_static = results_static["infections_by_region"]

        # Print results
        print("\n=== Results ===")
        print("Region | Time-Varying | Static")
        print("-" * 35)
        for i, name in enumerate(self.REGION_NAMES):
            print(f"{name:8} | {infections_tv[i]:12.1f} | {infections_static[i]:7.1f}")
        print("-" * 35)
        print(f"Total    | {results_tv['total_infections']:12.1f} | {results_static['total_infections']:7.1f}")

        # Test assertions

        # 1. Region A (seed) has infections in both scenarios
        assert infections_tv[0] > 0, f"Region A should have infections in time-varying case, got {infections_tv[0]}"
        assert infections_static[0] > 0, f"Region A should have infections in static case, got {infections_static[0]}"

        # 2. Region B has infections in both (connected before cutoff)
        assert infections_tv[1] > 0, f"Region B should have infections in time-varying case, got {infections_tv[1]}"
        assert infections_static[1] > 0, f"Region B should have infections in static case, got {infections_static[1]}"

        # 3. Region C has MORE infections in static case
        assert infections_static[2] > infections_tv[2], (
            f"Region C should have MORE infections in static case. "
            f"Got static={infections_static[2]}, time_varying={infections_tv[2]}"
        )

        # 4. Region D has infections ONLY in static case (or significantly more)
        assert infections_static[3] > 0, f"Region D should have infections in static case, got {infections_static[3]}"
        # Allow for some leakage in time-varying case, but should be much less
        assert infections_tv[3] < infections_static[3] * 0.5, (
            f"Region D should have FEW infections in time-varying case. "
            f"Got time_varying={infections_tv[3]}, static={infections_static[3]}"
        )

        # 5. Total infections: time-varying <= static (containment effect)
        assert results_tv["total_infections"] <= results_static["total_infections"], (
            f"Total infections should be less or equal in time-varying case (containment effect). "
            f"Got time_varying={results_tv['total_infections']}, static={results_static['total_infections']}"
        )

        print("\n=== All assertions passed! ===")

    def test_mobility_shutdown(self, temp_workspace):
        """
        Test mobility shutdown from day 1.

        Expected: infections stay only in region A
        """
        mobility_shutdown_path = self._create_mobility_netcdf(temp_workspace, mode="shutdown")
        seed_path = self._create_seed_file(temp_workspace)
        metapop_path = self._create_metapopulation_csv(temp_workspace)
        mobility_csv_path = self._create_mobility_csv(temp_workspace)

        config = self._create_config(temp_workspace, mobility_shutdown_path, "shutdown")
        results = self._run_simulation(config, temp_workspace, "shutdown")

        infections = results["infections_by_region"]

        print("\n=== Shutdown Test Results ===")
        print("Region | Infections")
        print("-" * 25)
        for i, name in enumerate(self.REGION_NAMES):
            print(f"{name:8} | {infections[i]:10.1f}")

        # Only region A should have significant infections
        assert infections[0] > 0, f"Region A should have infections, got {infections[0]}"
        for i in range(1, self.NUM_REGIONS):
            assert infections[i] < infections[0] * 0.01, (
                f"Region {self.REGION_NAMES[i]} should have negligible infections with shutdown, "
                f"got {infections[i]} vs region A: {infections[0]}"
            )

        print("\n=== Shutdown test passed! ===")

    def test_mobility_netcdf_generation(self, temp_workspace):
        """
        Unit test: Validate NetCDF file generation for time-varying mobility.

        This test validates that:
        1. NetCDF file is created with correct dimensions
        2. Mobility data has correct values for containment mode
        3. Date strings are properly formatted
        """
        # Create NetCDF file
        netcdf_path = self._create_mobility_netcdf(temp_workspace, mode="containment")

        # Verify file exists
        assert os.path.exists(netcdf_path), f"NetCDF file should exist at {netcdf_path}"

        # Open and validate structure
        ds = xr.open_dataset(netcdf_path)

        # Check dimensions
        assert "date" in ds.sizes, "NetCDF should have 'date' dimension"
        assert "origin" in ds.sizes, "NetCDF should have 'origin' dimension"
        assert "destination" in ds.sizes, "NetCDF should have 'destination' dimension"

        # Check variable
        assert "mobility" in ds.variables, "NetCDF should have 'mobility' variable"

        # Check dimension sizes
        assert ds.sizes["date"] == self.SIMULATION_DAYS, f"Should have {self.SIMULATION_DAYS} dates"
        assert ds.sizes["origin"] == self.NUM_REGIONS, f"Should have {self.NUM_REGIONS} origins"
        assert ds.sizes["destination"] == self.NUM_REGIONS, f"Should have {self.NUM_REGIONS} destinations"

        # Check mobility data shape
        # NOTE: Data is written transposed (dest, origin, date) for Julia, so xarray reads it as (M, M, T)
        mobility = ds["mobility"].values
        expected_shape = (self.NUM_REGIONS, self.NUM_REGIONS, self.SIMULATION_DAYS)
        assert mobility.shape == expected_shape, \
            f"Mobility data should have shape {expected_shape} (dest, origin, date for Julia)"

        # Validate containment mode behavior
        # Days 0-9: normal mobility (90% diagonal, 10% forward)
        # Day 10+: zero mobility (100% diagonal)
        # Need to index as [dest, origin, date] due to transpose
        for t in range(self.SIMULATION_DAYS):
            if t < self.MOBILITY_DROP_DAY:
                # Should have off-diagonal mobility (origin -> dest)
                assert mobility[1, 0, t] == 0.10, f"Day {t}: Should have 10% forward mobility (region 0 -> 1)"
                assert mobility[2, 1, t] == 0.10, f"Day {t}: Should have 10% forward mobility (region 1 -> 2)"
            else:
                # Should be diagonal only
                for i in range(self.NUM_REGIONS):
                    for j in range(self.NUM_REGIONS):
                        if i != j:
                            assert mobility[j, i, t] == 0.0, f"Day {t}: Should have zero off-diagonal mobility (region {i} -> {j})"

        # Check date format
        dates = ds.coords["date"].values
        for date_str in dates:
            assert isinstance(date_str, (str, bytes)), f"Date should be string, got {type(date_str)}"
            # Verify YYYY-MM-DD format
            if isinstance(date_str, str):
                assert len(date_str) == 10, f"Date string should be 10 characters, got {date_str}"
                assert date_str[4] == "-" and date_str[7] == "-", f"Date should have YYYY-MM-DD format, got {date_str}"

        ds.close()

        print("\n=== NetCDF generation test passed! ===")

    def test_mobility_config_validation(self, temp_workspace):
        """
        Unit test: Validate configuration with external NetCDF mobility.

        This test validates that:
        1. Configuration passes JSON schema validation
        2. Mobility parameters are correctly set
        """
        # Create test data files
        mobility_netcdf_path = self._create_mobility_netcdf(temp_workspace, mode="containment")
        config = self._create_config(temp_workspace, mobility_netcdf_path, "test")

        # Validate configuration
        from episim_python.schema_validator import EpiSimSchemaValidator

        validator = EpiSimSchemaValidator()
        is_valid = validator.validate_config(config, verbose=False)

        assert is_valid, "Configuration should pass schema validation"

        # Check mobility-specific parameters
        assert config["population_params"]["mobility_variation_type"] == "external_netcdf"
        assert config["population_params"]["mobility_external_file"] == os.path.abspath(mobility_netcdf_path)
        assert config["population_params"]["mobility_external_variable"] == "mobility"
        assert config["population_params"]["mobility_validate"] is True
        assert config["simulation"]["engine"] == "MMCACovid19Vac"  # Required for external_netcdf support

        print("\n=== Configuration validation test passed! ===")

    def test_static_mobility_netcdf(self, temp_workspace):
        """
        Unit test: Validate static mobility NetCDF generation.

        Static mode should have constant mobility throughout all timesteps.
        """
        netcdf_path = self._create_mobility_netcdf(temp_workspace, mode="static")

        ds = xr.open_dataset(netcdf_path)
        mobility = ds["mobility"].values

        # All days should have same mobility pattern
        # NOTE: Data is (dest, origin, date) due to transpose for Julia
        first_day = mobility[:, :, 0]
        for t in range(1, self.SIMULATION_DAYS):
            assert np.allclose(mobility[:, :, t], first_day), f"Day {t} should have same mobility as day 0"

        # Check chain topology: 90% diagonal, 10% forward
        # Indexing is [dest, origin, date], so check mobility[dest, origin]
        assert first_day[0, 0] == 0.90, "Should have 90% self-loop"
        assert first_day[1, 0] == 0.10, "Should have 10% forward mobility (0 -> 1)"
        assert first_day[2, 0] == 0.0, "Should have 0% non-forward mobility (0 -> 2)"

        ds.close()

        print("\n=== Static mobility NetCDF test passed! ===")
