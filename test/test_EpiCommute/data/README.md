Data folder for EpiCommute

This folder contains example input files and a small helper script to run a simulation.

Files and formats
-----------------
- params.json
  - JSON file containing model parameters and paths to the input files used by the script.
  - Keys used by the example:
    - mobility_file: filename of the mobility CSV (MxM matrix)
    - mobility_baseline_file: filename of the baseline mobility CSV (MxM matrix), can be null
    - subpopulation_sizes_file: filename of the subpopulation sizes CSV (one column named 'size')
    - quarantine_mode: null, 'isolation' or 'distancing'
    - outbreak_source: 'random' or an integer index of the seed subpopulation (0-based)
    - T_max, dt, dt_save, mu, R0, I0, save_observables, VERBOSE

- mobility.csv
  - Comma-separated values without header. Contains an M x M mobility matrix. Each row corresponds to an origin subpopulation and each column to a destination.
  - Example: 5x5 matrix for 5 subpopulations.

- mobility_baseline.csv
  - Same format as mobility.csv; represents pre-change mobility used for quarantine calculations. If you do not use quarantine, set mobility_baseline_file to null in params.json.

- subpopulation_sizes.csv
  - A single column with header 'size' and M rows, listing the population size of each subpopulation.

- run_simulation.py
  - Small helper script that reads params.json and the CSV files, runs the EpiCommute SIRModel, and writes results to data/results.json.

How to run the example
----------------------
1. Install requirements (from repository root):
   python -m pip install -r requirements.txt

2. From the repository root run:
   python data/run_simulation.py

This script loads data/params.json, the mobility and size files, runs the simulation and writes data/results.json with the saved observables.

Notes
-----
- The mobility matrix should be square (M x M) and should match the length of the subpopulation sizes array.
- If you set outbreak_source to an integer, it must be in range [0, M-1].
- If mobility_baseline_file is null but quarantine_mode is non-null, the model will raise an error. Ensure consistency between these settings.
