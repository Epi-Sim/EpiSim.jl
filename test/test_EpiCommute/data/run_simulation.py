#!/usr/bin/env python3
"""
Simple script to run an example simulation using the files in the data/ folder.
"""
import json
import os
import numpy as np

# Ensure package import path
try:
    from EpiCommute import SIRModel
except Exception as e:
    raise ImportError('Could not import EpiCommute. Make sure the package is installed or the repo root is on PYTHONPATH.')

DATA_DIR = os.path.dirname(__file__)
PARAMS_FILE = os.path.join(DATA_DIR, 'params.json')
RESULTS_FILE = os.path.join(DATA_DIR, 'results.json')


def load_params(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_csv_matrix(path):
    return np.loadtxt(path, delimiter=',')


def load_subpop_sizes(path):
    # Accept single-column CSV with optional header named 'size'
    try:
        data = np.loadtxt(path, delimiter=',', skiprows=1)
    except Exception:
        data = np.loadtxt(path, delimiter=',')
    # Ensure 1D
    return np.array(data).astype(int).reshape(-1)


def make_serializable(results):
    serial = {}
    for k, v in results.items():
        # Lists of numpy arrays (e.g., 'S', 'I', 'R')
        if isinstance(v, list):
            serial[k] = []
            for item in v:
                if hasattr(item, 'tolist'):
                    serial[k].append(np.array(item).tolist())
                else:
                    serial[k].append(item)
        else:
            if hasattr(v, 'tolist'):
                serial[k] = np.array(v).tolist()
            else:
                serial[k] = v
    return serial


if __name__ == '__main__':
    params = load_params(PARAMS_FILE)

    mobility_path = os.path.join(DATA_DIR, params['mobility_file'])
    mobility = load_csv_matrix(mobility_path)

    subpop_path = os.path.join(DATA_DIR, params['subpopulation_sizes_file'])
    subpopulation_sizes = load_subpop_sizes(subpop_path)

    mobility_baseline = None
    if params.get('mobility_baseline_file'):
        base_path = os.path.join(DATA_DIR, params['mobility_baseline_file'])
        mobility_baseline = load_csv_matrix(base_path)

    # Convert outbreak_source if necessary
    outbreak = params.get('outbreak_source', 'random')
    if isinstance(outbreak, str) and outbreak != 'random':
        try:
            outbreak = int(outbreak)
        except ValueError:
            outbreak = 'random'

    # Create model
    model = SIRModel(
        mobility=mobility,
        subpopulation_sizes=subpopulation_sizes,
        mobility_baseline=mobility_baseline,
        quarantine_mode=params.get('quarantine_mode', None),
        outbreak_source=outbreak,
        T_max=params.get('T_max', 100),
        dt=params.get('dt', 0.1),
        dt_save=params.get('dt_save', 1),
        mu=params.get('mu', 1/8),
        R0=params.get('R0', 3.0),
        I0=params.get('I0', 10),
        save_observables=params.get('save_observables', ['epi_subpopulations','epi_total','arrival_times']),
        VERBOSE=params.get('VERBOSE', False)
    )

    results = model.run_simulation()

    serial = make_serializable(results)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(serial, f, indent=2)

    print(f'Results written to {RESULTS_FILE}')
