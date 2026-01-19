import os
import sys
import tempfile
import shutil

sys.path.append(os.path.dirname(__file__))
from synthetic_generator import SyntheticDataGenerator

temp_dir = tempfile.mkdtemp(prefix="episim_test_")
print(f"Temp dir: {temp_dir}")

try:
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
    
    seed_fname, seed_path = generator.prepare_seed_file("test_check")
    
    # Run baseline (strength=0.0)
    print("\n=== BASELINE ===")
    generator.run_single_scenario(
        pid="test_check",
        scen_name="Baseline",
        strength=0.0,
        profile=profile,
        seed_path=seed_path
    )
    
    # Run intervention (strength=0.8)
    print("\n=== INTERVENTION (Global_Const, strength=0.8) ===")
    generator.run_single_scenario(
        pid="test_check",
        scen_name="Global_Const",
        strength=0.8,
        profile=profile,
        seed_path=seed_path
    )
    
    # Check the CSVs
    kappa_baseline = os.path.join(output_folder, "kappa0_test_check_Baseline.csv")
    kappa_intervention = os.path.join(output_folder, "kappa0_test_check_Global_Const_s80.csv")
    
    print(f"\nChecking if files exist:")
    print(f"Baseline kappa0 exists: {os.path.exists(kappa_baseline)}")
    print(f"Intervention kappa0 exists: {os.path.exists(kappa_intervention)}")
    
    if os.path.exists(kappa_intervention):
        print(f"\nIntervention CSV first 5 lines:")
        with open(kappa_intervention, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:
                    print(line.rstrip())
                else:
                    break
    
    print(f"\n\nTemp dir preserved at: {temp_dir}")
    print("Press Enter to clean up...")
    input()

finally:
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Cleaned up {temp_dir}")
