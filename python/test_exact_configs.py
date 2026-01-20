import json
import os
import sys
import tempfile

import pandas as pd

sys.path.append(os.path.dirname(__file__))
import xarray as xr

from episim_python.epi_sim import EpiSim

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_folder = os.path.join(project_root, "models", "mitma")
config_path = os.path.join(data_folder, "config_MMCACovid19.json")


def run_scenario(name, kappa_val, tc_val, kappa0_csv=None):
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

        if kappa0_csv:
            config_dict["data"]["kappa0_filename"] = kappa0_csv

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


print("Testing exact configurations:")
print("-" * 80)

# Create CSV with kappa0=0.2
with tempfile.TemporaryDirectory() as temp_dir:
    date_range = pd.date_range(start="2020-02-09", end="2020-02-28")
    df = pd.DataFrame(
        {
            "date": date_range.strftime("%Y-%m-%d"),
            "reduction": [0.2] * len(date_range),
            "datetime": date_range.strftime("%Y-%m-%d"),
            "time": range(len(date_range)),
        }
    )
    csv_path = os.path.join(temp_dir, "kappa0_test.csv")
    df.to_csv(csv_path, index=False)

    print("Baseline config uses κ₀=[1.0], tᶜ=[0] (full mobility)")
    print("Intervention uses CSV with κ₀=0.2 (20% transmission)")
    print("-" * 80)

    run_scenario("Baseline (κ₀=1.0, tᶜ=0)", 1.0, 0)
    run_scenario("Intervention (CSV κ₀=0.2)", 1.0, 0, csv_path)
    run_scenario("No intervention (κ₀=0.0)", 0.0, 0)
    run_scenario("Base config (κ₀=0.8, tᶜ=100)", 0.8, 100)
