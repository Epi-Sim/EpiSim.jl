import json
import os
import sys
import tempfile

sys.path.append(os.path.dirname(__file__))
from episim_python.epi_sim import EpiSim


def run_with_kappa(kappa, label, temp_dir):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(project_root, "models", "mitma")
    config_path = os.path.join(data_folder, "config_MMCACovid19.json")
    output_folder = os.path.join(temp_dir, label)
    os.makedirs(output_folder)

    with open(config_path) as f:
        config_dict = json.load(f)

    if "kappa0_filename" in config_dict["data"]:
        del config_dict["data"]["kappa0_filename"]

    config_dict["simulation"]["end_date"] = "2020-02-28"
    config_dict["NPI"]["κ₀s"] = [kappa]
    config_dict["NPI"]["tᶜs"] = [1]
    config_dict["simulation"]["save_full_output"] = False
    config_dict["simulation"]["save_observables"] = True

    model = EpiSim(
        config=config_dict,
        data_folder=data_folder,
        instance_folder=output_folder,
        initial_conditions=None,
    )
    model.setup(executable_type="interpreter")
    final_state, next_date = model.step("2020-02-09", length_days=20)

    import xarray as xr

    ds = xr.open_dataset(final_state)
    S_final = ds["S"].sum().values
    print(f"κ₀={kappa:4.2f}: {S_final:15.2f} susceptibles")


with tempfile.TemporaryDirectory() as temp_dir:
    print("Testing κ₀ values:")
    for kappa in [0.0, 0.2, 0.5, 0.8, 1.0]:
        run_with_kappa(kappa, f"kappa_{kappa}", temp_dir)
