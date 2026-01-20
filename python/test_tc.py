import json
import os
import sys
import tempfile

sys.path.append(os.path.dirname(__file__))
import xarray as xr

from episim_python.epi_sim import EpiSim

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_folder = os.path.join(project_root, "models", "mitma")
config_path = os.path.join(data_folder, "config_MMCACovid19.json")


def run_scenario(name, kappa_val, tc_val):
    with tempfile.TemporaryDirectory() as temp_dir:
        with open(config_path) as f:
            config_dict = json.load(f)

        if "kappa0_filename" in config_dict["data"]:
            del config_dict["data"]["kappa0_filename"]

        config_dict["simulation"]["end_date"] = "2020-02-28"
        config_dict["NPI"]["κ₀s"] = [kappa_val]
        config_dict["NPI"]["tᶜs"] = [tc_val]
        config_dict["simulation"]["save_full_output"] = False
        config_dict["simulation"]["save_observables"] = True

        output_folder = os.path.join(temp_dir, name)
        os.makedirs(output_folder)

        model = EpiSim(
            config=config_dict,
            data_folder=data_folder,
            instance_folder=output_folder,
            initial_conditions=None,
        )
        model.setup(executable_type="interpreter")
        final_state, next_date = model.step("2020-02-09", length_days=20)

        ds = xr.open_dataset(final_state)
        S_final = ds["S"].sum().values
        print(f"{name:35s} κ₀={kappa_val:4.2f} tᶜ={tc_val:3d}: {S_final:12.0f}")


print("Testing JSON mode with different tᶜ values:")
print("-" * 80)

# Test with tᶜ=1 (intervention starts at day 1)
run_scenario("Baseline (κ₀=1.0, tᶜ=1)", 1.0, 1)
run_scenario("Intervention (κ₀=0.2, tᶜ=1)", 0.2, 1)
run_scenario("No intervention (κ₀=0.0, tᶜ=1)", 0.0, 1)
