import os
import sys
import json
import tempfile
sys.path.append(os.path.dirname(__file__))
from synthetic_generator import SyntheticDataGenerator

with tempfile.TemporaryDirectory() as temp_dir:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(project_root, "models", "mitma")
    config_path = os.path.join(data_folder, "config_MMCACovid19.json")
    output_folder = os.path.join(temp_dir, "synthetic_test_output")
    
    generator = SyntheticDataGenerator(config_path, data_folder, output_folder)
    
    profile = {
        "profile_id": 999,
        "r0_scale": 2.5,
        "t_inf": 3.0,
        "t_inc": 5.0,
        "event_start": 5,
        "event_duration": 40,
        "affected_fraction": 0.5,
    }
    
    seed_fname, seed_path = generator.prepare_seed_file("test")
    
    # Run intervention
    generator.run_single_scenario(
        pid="test",
        scen_name="Global_Const",
        strength=0.8,
        profile=profile,
        seed_path=seed_path
    )
    
    # Check the config that was written
    config_file = os.path.join(output_folder, "run_test_Global_Const_s80")
    run_uuid = os.listdir(config_file)[0]
    config_path = os.path.join(config_file, run_uuid, "config_auto_py.json")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("=== Config JSON ===")
    print(f"kappa0_filename: {config['data'].get('kappa0_filename')}")
    print(f"NPI.κ₀s: {config['NPI']['κ₀s']}")
    print(f"NPI.ϕs: {config['NPI']['ϕs']}")
    print(f"NPI.δs: {config['NPI']['δs']}")
    print(f"NPI.tᶜs: {config['NPI']['tᶜs']}")
    
    # Check CSV
    csv_path = os.path.join(output_folder, "kappa0_test_Global_Const_s80.csv")
    print(f"\n=== CSV File ===")
    with open(csv_path, 'r') as f:
        lines = f.readlines()[:5]
        print(''.join(lines))
