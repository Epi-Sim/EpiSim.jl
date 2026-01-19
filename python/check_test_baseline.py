import os
import sys
import tempfile
import json
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
    
    # Baseline (strength=0.0)
    generator.run_single_scenario("test", "Baseline", 0.0, profile, seed_path)
    
    # Check config
    baseline_config_dir = os.path.join(output_folder, "run_test_Baseline")
    baseline_uuid = os.listdir(baseline_config_dir)[0]
    baseline_config_path = os.path.join(baseline_config_dir, baseline_uuid, "config_auto_py.json")
    
    with open(baseline_config_path, 'r') as f:
        config = json.load(f)
    
    print("=== Baseline Config ===")
    print(f"κ₀s: {config['NPI']['κ₀s']}")
    print(f"tᶜs: {config['NPI']['tᶜs']}")
    print(f"kappa0_filename: {config['data'].get('kappa0_filename', 'Not present')}")
    
    # Also check the base config to see what value it has
    with open(config_path, 'r') as f:
        base_config = json.load(f)
    print("\n=== Base Config (from file) ===")
    print(f"κ₀s: {base_config['NPI']['κ₀s']}")
    print(f"tᶜs: {base_config['NPI']['tᶜs']}")
