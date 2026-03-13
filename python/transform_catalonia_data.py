#!/usr/bin/env python3
"""
Transform Catalonia data from dasymetric_mob output to EpiSim format.

Input files (from python/data/):
- censph540mun.csv - Census population data
- mobility.zarr - Time series OD matrix
- edar_muni_edges.nc - EDAR mapping
- ../municipality_id_reference.csv - Municipality code reference

Output files (to models/catalonia/):
- metapopulation_data.csv - Population by age group (Y, M, O)
- rosetta.csv - ID to 1-based index mapping
- R_mobility_matrix.csv - Mobility matrix (single time slice)
- A0_initial_conditions_seeds.csv - Initial conditions
- config_MMCACovid19.json - Config file

Format matches models/mitma/ template.
"""

import json
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import zarr


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'python' / 'data'
    output_dir = project_root / 'models' / 'catalonia'

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Catalonia Data Transformation for EpiSim")
    print("=" * 60)
    print(f"Input: {data_dir}")
    print(f"Output: {output_dir}")

    # ========================================================================
    # Step 1: Load mobility data to get the list of municipalities
    # ========================================================================
    print("\n[Step 1] Loading mobility data...")
    z = zarr.open(data_dir / 'mobility.zarr')
    mobility_array = z['mobility']
    origin_array = z['origin']
    target_array = z['target']

    mobility_muni = [str(x) for x in origin_array[:]]
    n_muni = len(mobility_muni)
    n_timesteps = mobility_array.shape[0]
    print(f"  Found {n_muni} municipalities, {n_timesteps} timesteps")

    # ========================================================================
    # Step 2: Load census data and map to municipality codes
    # ========================================================================
    print("\n[Step 2] Loading census data...")
    census_df = pd.read_csv(data_dir / 'censph540mun.csv')

    # Filter to most recent year (2024) and total sex
    census_2024 = census_df[
        (census_df['year'] == 2024) &
        (census_df['sex'] == 'total') &
        (census_df['municipality'] != 'Catalunya')  # Exclude region total
    ].copy()

    # Load municipality reference to get codes
    muni_ref = pd.read_csv(project_root / 'municipality_id_reference.csv')
    muni_ref['muni_code_5digit'] = muni_ref['muni_code_5digit'].astype(str).str.zfill(5)

    # Create name to code mapping
    name_to_code = dict(zip(muni_ref['muni_name'], muni_ref['muni_code_5digit']))

    # Map census municipalities to codes
    census_2024['muni_code'] = census_2024['municipality'].map(name_to_code)

    # Check for unmatched municipalities
    unmatched = census_2024[census_2024['muni_code'].isna()]['municipality'].unique()
    if len(unmatched) > 0:
        print(f"  Warning: {len(unmatched)} census municipalities not found in reference:")
        print(f"    {sorted(unmatched)[:10]}")

    # Pivot census data to get Y, M, O populations
    age_mapping = {
        'from 0 to 15 years': 'Y',
        'from 16 to 64 years': 'M',
        '65 years and over': 'O',
        'total': 'total'
    }

    census_pivot = census_2024.pivot_table(
        index='muni_code',
        columns='age',
        values='value',
        aggfunc='sum'
    ).reset_index()

    census_pivot.columns.name = None
    census_pivot = census_pivot.rename(columns=age_mapping)

    # Ensure required columns exist
    for col in ['Y', 'M', 'O', 'total']:
        if col not in census_pivot.columns:
            census_pivot[col] = 0

    # Keep only municipalities that exist in mobility data
    census_pivot = census_pivot[census_pivot['muni_code'].isin(mobility_muni)]

    print(f"  Mapped {len(census_pivot)} municipalities to population data")

    # ========================================================================
    # Step 3: Create metapopulation data file
    # ========================================================================
    print("\n[Step 3] Creating metapopulation data file...")

    # Sort by municipality code
    census_pivot = census_pivot.sort_values('muni_code')

    # Calculate area (use placeholder - should come from shapefile)
    # For now, estimate from population (rough approximation)
    census_pivot['area'] = np.sqrt(census_pivot['total']) * 0.5

    # Create metapopulation dataframe
    metapop_df = census_pivot[['muni_code', 'area', 'Y', 'M', 'O', 'total']].copy()
    metapop_df = metapop_df.rename(columns={'muni_code': 'id'})
    metapop_df['id'] = metapop_df['id'].astype(str)

    # Save metapopulation data
    metapop_output = output_dir / 'metapopulation_data.csv'
    metapop_df.to_csv(metapop_output, index=False)
    print(f"  Saved {len(metapop_df)} municipalities to {metapop_output}")

    # ========================================================================
    # Step 4: Create rosetta file (1-based indices)
    # ========================================================================
    print("\n[Step 4] Creating rosetta file...")
    rosetta_df = pd.DataFrame({
        'id': metapop_df['id'],
        'idx': range(1, len(metapop_df) + 1)
    })

    # Create ID to index mapping
    id_to_idx = dict(zip(rosetta_df['id'], rosetta_df['idx']))
    idx_to_id = dict(zip(rosetta_df['idx'], rosetta_df['id']))

    rosetta_output = output_dir / 'rosetta.csv'
    rosetta_df.to_csv(rosetta_output, index=False)
    print(f"  Saved rosetta to {rosetta_output}")

    # ========================================================================
    # Step 5: Extract mobility matrix (use first time slice)
    # ========================================================================
    print("\n[Step 5] Extracting mobility matrix...")

    # Load mobility data - use first time slice
    mobility_slice = mobility_array[0]  # Shape: (963, 963)

    # Convert to sparse format (source_idx, target_idx, ratio)
    mobility_rows = []
    threshold = 1e-10  # Threshold for non-zero mobility

    for i in range(n_muni):
        source_id = mobility_muni[i]
        if source_id not in id_to_idx:
            continue

        source_idx = id_to_idx[source_id]

        for j in range(n_muni):
            target_id = mobility_muni[j]
            if target_id not in id_to_idx:
                continue

            ratio = float(mobility_slice[i, j])
            if ratio > threshold:
                target_idx = id_to_idx[target_id]
                mobility_rows.append({
                    'source_idx': source_idx,
                    'target_idx': target_idx,
                    'ratio': ratio
                })

    mobility_df = pd.DataFrame(mobility_rows)

    mobility_output = output_dir / 'R_mobility_matrix.csv'
    mobility_df.to_csv(mobility_output, index=False)
    print(f"  Saved {len(mobility_df)} mobility entries to {mobility_output}")

    # ========================================================================
    # Step 6: Create initial conditions file
    # ========================================================================
    print("\n[Step 6] Creating initial conditions file...")

    init_conditions = []
    for _, row in metapop_df.iterrows():
        # Use small initial infections proportional to population
        y_init = max(0.12, row['Y'] * 0.0001)
        m_init = max(0.16, row['M'] * 0.0001)
        o_init = max(0.72, row['O'] * 0.0001)

        idx = id_to_idx[row['id']]

        init_conditions.append({
            'name': row['id'],
            'id': row['id'],
            'idx': idx,
            'Y': round(y_init, 2),
            'M': round(m_init, 2),
            'O': round(o_init, 2)
        })

    init_df = pd.DataFrame(init_conditions)

    init_output = output_dir / 'A0_initial_conditions_seeds.csv'
    init_df.to_csv(init_output, index=False)
    print(f"  Saved initial conditions to {init_output}")

    # ========================================================================
    # Step 7: Create config file
    # ========================================================================
    print("\n[Step 7] Creating config file...")

    # Load base config template from models/mitma
    base_config_path = project_root / 'models' / 'mitma' / 'config_MMCACovid19.json'
    with open(base_config_path) as f:
        config = json.load(f)

    # Update data filenames
    config['data'] = {
        'initial_condition_filename': 'A0_initial_conditions_seeds.csv',
        'metapopulation_data_filename': 'metapopulation_data.csv',
        'mobility_matrix_filename': 'R_mobility_matrix.csv',
        'kappa0_filename': config['data'].get('kappa0_filename', 'kappa0_from_mitma.csv')
    }

    config_output = output_dir / 'config_MMCACovid19.json'
    with open(config_output, 'w') as f:
        json.dump(config, f, indent='\t')
    print(f"  Saved config to {config_output}")

    # ========================================================================
    # Step 8: Create summary file
    # ========================================================================
    print("\n[Step 8] Creating summary file...")

    # Load EDAR data to get EDAR municipality count
    nc = netCDF4.Dataset(data_dir / 'edar_muni_edges.nc')
    edar_home = [str(x) for x in nc.variables['home'][:]]
    nc.close()

    edar_in_dataset = set(metapop_df['id']) & set(edar_home)

    summary = {
        'n_municipalities': len(metapop_df),
        'n_mobility_timesteps': n_timesteps,
        'n_mobility_entries': len(mobility_df),
        'n_edar_municipalities': len(edar_home),
        'n_edar_in_dataset': len(edar_in_dataset),
        'municipality_codes': metapop_df['id'].tolist(),
        'population_summary': {
            'total_Y': int(metapop_df['Y'].sum()),
            'total_M': int(metapop_df['M'].sum()),
            'total_O': int(metapop_df['O'].sum()),
            'total_population': int(metapop_df['total'].sum())
        }
    }

    summary_output = output_dir / 'catalonia_dataset_summary.json'
    with open(summary_output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to {summary_output}")

    print("\n" + "=" * 60)
    print("Transformation complete!")
    print("=" * 60)
    print("\nSummary:")
    print(f"  Municipalities: {len(metapop_df)}")
    print(f"  Population: Y={summary['population_summary']['total_Y']:,}, "
          f"M={summary['population_summary']['total_M']:,}, "
          f"O={summary['population_summary']['total_O']:,}")
    print(f"  EDAR municipalities: {summary['n_edar_in_dataset']}/{summary['n_edar_municipalities']}")
    print("\nOutput files:")
    print(f"  - {metapop_output}")
    print(f"  - {rosetta_output}")
    print(f"  - {mobility_output}")
    print(f"  - {init_output}")
    print(f"  - {config_output}")
    print(f"  - {summary_output}")


if __name__ == '__main__':
    main()
