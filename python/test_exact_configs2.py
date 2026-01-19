import os
import sys
import tempfile
import json
import pandas as pd
sys.path.append(os.path.dirname(__file__))
from episim_python.epi_sim import EpiSim
import xarray as xr

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_folder = os.path.join(project_root, "models", "mitma")
config_path = os.path.join(data_folder, "config_MMCACovid19.json")

def run_scenario(name, kappa_s, tc_s, kappa0_csv=None):
    with tempfile.TemporaryDirectory() as temp_dir:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        if "kappa0_filename" in config_dict["data"]:
            del config_dict["data"]["kappa0_filename"]
        
        config_dict["simulation"]["end_date"] = "2020-02-28"
        config_dict["NPI"]["κ₀s"] = kappa_s
        config_dict["NPI"]["tᶜs"] = tc_s
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
            initial_conditions=None
        )
        model.setup(executable_type='interpreter')
        final_state, next_date = model.step("2020-02-09", length_days=20)
        
        ds = xr.open_dataset(final_state)
        S_final = ds['S'].sum().values
        kappa_str = str(kappa_s)
        print(f"{name:30s} κ₀={kappa_str:8s} tᶜ={tc_s:3d}: {S_final:12.0f} susceptibles")

print("Testing exact configurations:")
print("-" * 70)

# Create CSV with kappa0=0.2
with tempfile.TemporaryDirectory() as temp_dir:
    date_range = pd.date_range(start="2020-02-09", end="2020-02-28")
    df = pd.DataFrame({
        "date": date_range.strftime("%Y-%m-%d"),
        "reduction": [0.2] * len(date_range),
        "datetime": date_range.strftime("%Y-%m-%d"),
        "time": range(len(date_range)),
    })
    csv_path = os.path.join(temp_dir, "kappa0_test.csv")
    df.to_csv(csv_path, index=False)
    
    run_scenario("Baseline (κ₀=1.0, tᶜ=0)", [1.0], [0])
    run_scenario("Intervention (CSV κ₀=0.2)", [1.0], [0], csv_path)
    run_scenario("No κ₀ value (κ₀=0.0)", [0.0], [0])
    run_scenario("Base config (κ₀=0.8, tᶜ=100)", [0.8], [100])
