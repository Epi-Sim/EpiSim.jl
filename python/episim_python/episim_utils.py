import copy
import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import xarray as xr

from .schema_validator import EpiSimSchemaValidator

SCHEMA_VALIDATION_AVAILABLE = True


class Metapopulation:
    def __init__(self, metapop_csv, rosetta_csv=None):
        self._metapop_csv = metapop_csv
        self._rosetta_csv = rosetta_csv
        self._region_ids = []
        self._region_areas = {}
        self._agent_types = []
        self._levels = []

        self._metapop_csv = metapop_csv
        pop_df = pd.read_csv(self._metapop_csv, index_col="id")

        self._region_ids = pop_df.index.tolist()
        self._region_areas = pop_df["area"].to_dict()

        # Drop column with region areas
        pop_df = pop_df.drop("area", axis=1)

        if pop_df.shape[1] > 1:
            pop_df = pop_df.drop("total", axis=1)

        self._agent_types = pop_df.columns.tolist()
        self._populations = pop_df

        if self._rosetta_csv is not None:
            self._rosetta = pd.read_csv(self._rosetta_csv, dtype=object)
            self._levels = self._rosetta.columns
            self._rosetta = self._rosetta.set_index("level_1")

            assert set(self._rosetta.index) == set(self._region_ids)

    def as_datarray(self):
        da = xr.DataArray(
            data=self._populations.values,
            name="population",
            coords={"M": self._region_ids, "G": self._agent_types},
            dims=["M", "G"],
        )
        return da

    def aggregate_to_level(self, level, as_array=True):
        if level == "level_1":
            df = self._populations
        else:
            df = pd.merge(
                self._populations,
                self._rosetta[[level]],
                left_index=True,
                right_index=True,
            )
            df = df.groupby(level).sum()
        if as_array:
            da = xr.DataArray(
                data=df.values,
                name="population",
                coords={"M": df.index.tolist(), "G": self._agent_types},
                dims=["M", "G"],
            )
            return da

        return df


class EpiSimConfig:
    def __init__(self, template: dict, group_suffix="ᵍ"):
        self._base_template = copy.deepcopy(template)
        self.config = copy.deepcopy(template)

        self._group_suffix = group_suffix
        # Extract demographic group labels (e.g., Y, M, O)
        self.group_labels = self._get_nested(["population_params", "G_labels"])
        self.group_size = len(self.group_labels)

        # Automatically detect all parameters that are group-dependent
        self.group_params = self._detect_group_params()

    @classmethod
    def from_json(cls, json_path: str) -> "EpiSimConfig":
        with open(json_path, encoding="utf-8") as f:
            template = json.load(f)
        return cls(template)

    def validate(self, verbose: bool = True, use_schema: bool = True):
        """
        Validate the configuration using both JSON schema and custom validation.

        Args:
            verbose: Whether to print validation messages
            use_schema: Whether to use JSON schema validation (requires jsonschema package)

        Raises:
            ValueError: If validation fails
        """
        errors = []

        # JSON Schema validation (if available and enabled)
        if use_schema and SCHEMA_VALIDATION_AVAILABLE:
            try:
                schema_validator = EpiSimSchemaValidator()
                is_valid, schema_errors = schema_validator.validate_config_safe(
                    self.config,
                    verbose=False,
                )
                if not is_valid:
                    errors.extend([f"Schema: {err}" for err in schema_errors])
                elif verbose:
                    print(
                        f"JSON Schema validation passed (group size: {self.group_size})",
                    )
            except Exception as e:
                errors.append(f"Schema validation error: {e!s}")
        elif use_schema and not SCHEMA_VALIDATION_AVAILABLE and verbose:
            print(
                "Warning: JSON schema validation requested but jsonschema package not available",
            )

        # Custom validation (backward compatibility)
        required_keys = {
            "simulation": ["engine", "start_date", "end_date"],
            "data": ["initial_condition_filename", "metapopulation_data_filename"],
            "epidemic_params": [],
            "population_params": ["G_labels", "C", "kᵍ", "kᵍ_h", "kᵍ_w", "pᵍ"],
            "NPI": ["κ₀s", "ϕs", "δs", "tᶜs", "are_there_npi"],
        }

        # Check required top-level and nested keys
        for section, subkeys in required_keys.items():
            if section not in self.config:
                errors.append(f"Missing section: '{section}'")
                continue
            for key in subkeys:
                if key not in self.config[section]:
                    errors.append(f"Missing key: '{section}.{key}'")

        # Check group-dependent parameters (redundant with schema but kept for robustness)
        for key_path, is_grouped in self.group_params.items():
            if is_grouped:
                try:
                    value = self.get_param(key_path)
                    if len(value) != self.group_size:
                        errors.append(
                            f"Length mismatch in '{key_path}': expected {self.group_size}, got {len(value)}",
                        )
                except Exception as e:
                    errors.append(f"Could not access group parameter '{key_path}': {e}")

        # Additional custom validations can be added here
        # For example: date format validation, parameter value constraints, etc.

        if errors:
            if verbose:
                print("Validation errors:")
                for err in errors:
                    print("  -", err)
            raise ValueError(
                f"Configuration validation failed with {len(errors)} error(s).",
            )
        if verbose:
            print("Configuration is valid.")

    def to_json(self, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

    def reset(self):
        self.config = copy.deepcopy(self._base_template)

    def _get_nested(self, keys: list) -> Any:
        d = self.config
        for k in keys:
            d = d[k]
        return d

    def _set_nested(self, keys: list, value: Any):
        d = self.config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def _detect_group_params(self) -> Dict[str, bool]:
        def recurse(d, prefix=""):
            result = {}
            for k, v in d.items():
                path = f"{prefix}.{k}" if prefix else k
                if (
                    isinstance(v, list)
                    and len(v) == self.group_size
                    and all(isinstance(x, (float, int)) for x in v)
                ):
                    result[path] = True
                elif isinstance(v, dict):
                    result.update(recurse(v, path))
            return result

        return recurse(self.config)

    def is_group_param(self, key_path: str) -> bool:
        return self.group_params.get(key_path, False)

    def update_param(self, key_path: str, value: Any):
        """
        Update a parameter, automatically handling group/single values based on detection
        """
        keys = key_path.split(".")
        is_grouped = self.is_group_param(key_path)

        if is_grouped:
            if not (isinstance(value, list) and len(value) == self.group_size):
                raise ValueError(
                    f"Expected a list of length {self.group_size} for group-dependent parameter '{key_path}'",
                )
        elif isinstance(value, list):
            raise ValueError(f"Expected a scalar for scalar parameter '{key_path}'")

        self._set_nested(keys, value)

    def get_param(self, key_path: str) -> Any:
        keys = key_path.split(".")
        return self._get_nested(keys)

    def inject(self, updates: Dict[str, Any]):
        for key_path, value in updates.items():
            self.update_param(key_path, value)

    def update_group_param(self, key_path: str, group_label: str, value: float):
        """
        Update a single group-specific value inside a vector (e.g. change γᵍ for group 'M')
        """
        if not self.is_group_param(key_path):
            raise ValueError(
                f"Parameter '{key_path}' is not detected as group-dependent.",
            )

        if group_label not in self.group_labels:
            raise ValueError(
                f"Group label '{group_label}' not in G_labels: {self.group_labels}",
            )

        param_vector = self.get_param(key_path)
        group_index = self.group_labels.index(group_label)
        param_vector[group_index] = value
        self.update_param(key_path, param_vector)

    def get_group_param(self, key_path: str, group_label: str) -> float:
        if not self.is_group_param(key_path):
            raise ValueError(
                f"Parameter '{key_path}' is not detected as group-dependent.",
            )

        if group_label not in self.group_labels:
            raise ValueError(
                f"Group label '{group_label}' not in G_labels: {self.group_labels}",
            )

        param_vector = self.get_param(key_path)
        group_index = self.group_labels.index(group_label)
        return param_vector[group_index]

    def inject_group_vector(self, key_path: str, values_by_group: Dict[str, float]):
        if not self.is_group_param(key_path):
            raise ValueError(f"Parameter '{key_path}' is not group-dependent.")

        new_vector = self.get_param(key_path)
        for group_label, value in values_by_group.items():
            if group_label not in self.group_labels:
                raise ValueError(
                    f"Group label '{group_label}' not in G_labels: {self.group_labels}",
                )
            index = self.group_labels.index(group_label)
            new_vector[index] = value
        self.update_param(key_path, new_vector)

    def update_params_from_flat_dict(self, param_set: Dict[str, float]):
        for param, value in param_set.items():
            if self._group_suffix in param:
                # e.g., 'γᵍ1' → base='γᵍ', group='1'
                try:
                    base, group_id = param.rsplit(self._group_suffix, 1)
                    base += self._group_suffix  # put back the suffix
                    if group_id.isdigit():
                        group_index = int(group_id)
                        self.update_group_param(base, group_index, value)
                    elif group_id in self.get_param("population_params.G_labels"):
                        self.update_group_param(base, group_id, value)
                    else:
                        raise ValueError(
                            f"Unrecognized group label/index '{group_id}' in param '{param}'",
                        )
                except Exception as e:
                    raise ValueError(f"Failed to parse grouped param '{param}': {e}")
            else:
                self.update_param(param, value)


def update_params(params_dict, update_dict, G=3):
    # Mapping epi_param βᴵ
    if "βᴵ" in update_dict:
        params_dict["epidemic_params"]["βᴵ"] = update_dict["βᴵ"]
    elif "β" in update_dict:
        params_dict["epidemic_params"]["βᴵ"] = update_dict["β"]
    # Mapping epi_param βᴬ
    if "βᴬ" in update_dict:
        params_dict["epidemic_params"]["βᴬ"] = update_dict["βᴬ"]
    elif "scale_β" in update_dict:
        params_dict["epidemic_params"]["βᴬ"] = (
            params_dict["epidemic_params"]["βᴵ"] * update_dict["scale_β"]
        )
    # Mapping epi_param ηᵍ
    if "ηᵍ" in update_dict:
        x = update_dict["ηᵍ"]
        params_dict["epidemic_params"]["ηᵍ"] = [x, x, x]
    elif ("τ_inc" in update_dict) and ("scale_ea" in update_dict):
        t_inc = update_dict["τ_inc"]
        s_ea = update_dict["scale_ea"]
        nu_g = 1.0 / (t_inc * (1.0 - s_ea))
        params_dict["epidemic_params"]["ηᵍ"] = [nu_g] * G
    if "ηᵍY" in update_dict:
        params_dict["epidemic_params"]["ηᵍ"] = [
            update_dict["ηᵍY"],
            update_dict["ηᵍM"],
            update_dict["ηᵍO"],
        ]
    # Mapping epi_param αᵍ
    if "αᵍ" in update_dict:
        a = update_dict["αᵍ"]
        a1 = a / 2.5
        params_dict["epidemic_params"]["αᵍ"] = [a1, a, a]
    elif (
        ("τ_inc" in update_dict)
        and ("scale_ea" in update_dict)
        and ("τᵢ" in update_dict)
    ):
        t_inc = update_dict["τ_inc"]
        s_ea = update_dict["scale_ea"]
        ti = update_dict["τᵢ"]
        n1 = 1.0 / (ti - 1 + t_inc * s_ea)
        n2 = 1.0 / (t_inc * s_ea)
        n3 = 1.0 / (t_inc * s_ea)
        params_dict["epidemic_params"]["αᵍ"] = [n1, n2, n3]
    if "αᵍY" in update_dict:
        params_dict["epidemic_params"]["αᵍ"] = [
            update_dict["αᵍY"],
            update_dict["αᵍM"],
            update_dict["αᵍO"],
        ]

    # Mapping epi_param μᵍ
    if "μᵍ" in update_dict:
        u = update_dict["μᵍ"]
        params_dict["epidemic_params"]["μᵍ"] = [1, u, u]
    elif "τᵢ" in update_dict:
        ti = update_dict["τᵢ"]
        params_dict["epidemic_params"]["μᵍ"] = [1.0, 1.0 / ti, 1.0 / ti]
    if "μᵍY" in update_dict:
        params_dict["epidemic_params"]["μᵍ"] = [
            update_dict["μᵍY"],
            update_dict["μᵍM"],
            update_dict["μᵍO"],
        ]

    if "γᵍY" in update_dict:
        params_dict["epidemic_params"]["γᵍ"][0] = update_dict["γᵍY"]
    if "γᵍM" in update_dict:
        params_dict["epidemic_params"]["γᵍ"][1] = update_dict["γᵍM"]
    if "γᵍO" in update_dict:
        params_dict["epidemic_params"]["γᵍ"][2] = update_dict["γᵍO"]
    # arreglar esto
    if "ϕs" in update_dict:
        if isinstance(update_dict["ϕs"], list):
            params_dict["NPI"]["ϕs"] = update_dict["ϕs"]
        else:
            params_dict["NPI"]["ϕs"] = [update_dict["ϕs"]]

    if "ϕs1" in update_dict:
        params_dict["NPI"]["ϕs"] = [
            update_dict["ϕs1"],
            update_dict["ϕs2"],
            update_dict["ϕs3"],
            update_dict["ϕs4"],
        ]

    if "δs" in update_dict:
        if isinstance(update_dict["δs"], list):
            params_dict["NPI"]["δs"] = update_dict["δs"]
        else:
            params_dict["NPI"]["δs"] = [update_dict["δs"]]

    if "δs1" in update_dict:
        params_dict["NPI"]["δs"] = [
            update_dict["δs1"],
            update_dict["δs2"],
            update_dict["δs3"],
            update_dict["δs4"],
        ]

    if "initial_condition_filename" in update_dict:
        params_dict["data"]["initial_condition_filename"] = update_dict[
            "initial_condition_filename"
        ]

    if "ϵᵍ" in update_dict:
        params_dict["vaccination"]["ϵᵍ"] = update_dict["ϵᵍ"]

    if "percentage_of_vacc_per_day" in update_dict:
        params_dict["vaccination"]["percentage_of_vacc_per_day"] = update_dict[
            "percentage_of_vacc_per_day"
        ]

    if "start_vacc" in update_dict:
        params_dict["vaccination"]["start_vacc"] = update_dict["start_vacc"]

    if "dur_vacc" in update_dict:
        params_dict["vaccination"]["dur_vacc"] = update_dict["dur_vacc"]

    return params_dict


def compute_observables(sim_xa, instance_folder, data_folder, **kwargs):
    """
    new_infectᵍ(t+1) = A(t) * αᵍ

    hosp_rateᵍ = μᵍ * (1 - θᵍ) * γᵍ
    new_hosptᵍ(t+1)  = I(t) * hosp_rateᵍ
    """

    observable_labels = ["A", "I", "D"]

    config_fname = os.path.join(instance_folder, "episim_config.json")

    with open(config_fname) as fh:
        config_dict = json.load(fh)

    G_labels = config_dict["population_params"]["G_labels"]
    epi_params = config_dict["epidemic_params"]

    sim_observables_xa = sim_xa.loc[:, :, :, observable_labels]
    sim_observables_xa = sim_observables_xa.transpose("epi_states", "M", "T", "G")

    alphas = np.zeros(len(G_labels))
    hosp_rates = np.zeros(len(G_labels))
    for i, _g in enumerate(G_labels):
        alphas[i] = epi_params["αᵍ"][i]
        hosp_rates[i] = (epi_params["μᵍ"][i] * (1 - epi_params["θᵍ"][i])) * epi_params[
            "γᵍ"
        ][i]

    # Computing daily new asymptomatic
    # alphas_vec = np.array([alphas[g] for g in sim_xa.coords['G'].values])
    sim_observables_xa.loc["A", :, :, :] = np.multiply(
        sim_observables_xa.loc["A", :, :, :],
        alphas,
    )

    # hosp_rates_vec = np.array([hosp_rates[g] for g in sim_xa.coords['G'].values])
    sim_observables_xa.loc["I", :, :, :] = np.multiply(
        sim_observables_xa.loc["I", :, :, :],
        hosp_rates,
    )

    dims = sim_observables_xa.dims
    data = sim_observables_xa.values
    coords = sim_observables_xa.coords

    coords["epi_states"] = ["I", "H", "D"]
    sim_observables_xa = xr.DataArray(data, coords=coords, dims=dims)
    sim_observables_xa = sim_observables_xa.transpose("epi_states", "G", "M", "T")

    sim_observables_xa.loc["D", :] = np.diff(sim_observables_xa.loc["D", :], prepend=0)

    return sim_observables_xa
