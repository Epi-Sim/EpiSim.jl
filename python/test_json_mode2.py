import os
import sys
import tempfile
import json
sys.path.append(os.path.dirname(__file__))
from synthetic_generator import SyntheticDataGenerator
from episim_python.epi_sim import EpiSim

with tempfile.TemporaryDirectory() as temp_dir:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(project_root, "models", "mitma")
    config_path = os.path.join(data_folder, "config_MMCACovid19.json")
    output_folder = os.path.join(temp_dir, "test_json")
    os.makedirs(output_folder)
    
    # Load and modify config for JSON-only mode
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Remove kappa0_filename to use JSON-only mode
    if "kappa0_filename" in config_dict["data"]:
        del config_dict["data"]["kappa0_filename"]
    
    # Modify for JSON mode (intervention from day 1)
    config_dict["simulation"]["end_date"] = "2020-02-28"
    config_dict["NPI"]["κ₀s"] = [0.8]  # 80% reduction
    config_dict["NPI"]["tᶜs"] = [1]  # Start from day 1
    config_dict["simulation"]["save_full_output"] = False
    config_dict["simulation"]["save_observables"] = True
    
    # Save config
    config_file = os.path.join(output_folder, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    print("=== JSON Mode Config ===")
    print(f"κ₀s: {config_dict['NPI']['κ₀s']}")
    print(f"tᶜs: {config_dict['NPI']['tᶜs']}")
    print(f"kappa0_filename exists: {'kappa0_filename' in config_dict['data']}")
    
    # Run simulation
    print("\nRunning JSON mode simulation...")
    model = EpiSim(
        config=config_dict,
        data_folder=data_folder,
        instance_folder=output_folder,
        initial_conditions=None
    )
    model.setup(executable_type='interpreter')
    
    final_state, next_date = model.step("2020-02-09", length_days=20)
    
    # Read output
    import xarray as xr
    ds = xr.open_dataset(final_state)
    S_final = ds['S'].sum().values
    print(f"Remaining Susceptibles: {S_final}")
