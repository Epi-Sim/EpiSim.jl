import os
import sys
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

    # Baseline
    generator.run_single_scenario("test", "Baseline", 0.0, profile, seed_path)
    baseline_config = os.path.join(output_folder, "run_test_Baseline")
    baseline_uuid = os.listdir(baseline_config)[0]
    baseline_config_path = os.path.join(
        baseline_config, baseline_uuid, "config_auto_py.json"
    )

    # Intervention
    generator.run_single_scenario("test", "Global_Const", 0.8, profile, seed_path)
    intervention_csv = os.path.join(output_folder, "kappa0_test_Global_Const_s80.csv")

    print("=== Baseline Config (JSON mode) ===")
    import json

    with open(baseline_config_path) as f:
        cfg = json.load(f)
    print(f"κ₀s: {cfg['NPI']['κ₀s']}")
    print(f"tᶜs: {cfg['NPI']['tᶜs']}")
    print(f"kappa0_filename: {'kappa0_filename' in cfg['data']}")

    print("\n=== Intervention CSV ===")
    with open(intervention_csv) as f:
        lines = f.readlines()[:5]
        print("".join(lines))
