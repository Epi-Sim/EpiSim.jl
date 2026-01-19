import os
import sys
sys.path.append(os.path.dirname(__file__))
from synthetic_generator import SyntheticDataGenerator
import tempfile

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
    
    seed_fname, seed_path = generator.prepare_seed_file("test_debug")
    
    # Test Global_Const with strength=0.8
    run_id = f"test_debug_Global_Const_s80"
    print(f"Calling run_single_scenario with run_id={run_id}, strength=0.8")
    generator.run_single_scenario(
        pid="test_debug",
        scen_name="Global_Const",
        strength=0.8,
        profile=profile,
        seed_path=seed_path
    )
    
    # Check the generated CSV
    kappa_path = os.path.join(output_folder, f"kappa0_{run_id}.csv")
    if os.path.exists(kappa_path):
        print(f"\nGenerated CSV content:")
        with open(kappa_path, 'r') as f:
            print(f.read()[:500])
    else:
        print(f"ERROR: CSV not found at {kappa_path}")
