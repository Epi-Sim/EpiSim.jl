#!/usr/bin/env python3
"""
Generate initial conditions file for EDAR 159 regions.

This script creates an A0_initial_conditions_seeds_edar159.csv file
compatible with the filtered 159-region dataset.
"""

from pathlib import Path

import pandas as pd


def generate_initial_conditions_edar159():
    """Generate initial conditions for EDAR 159 regions."""

    # Paths
    project_root = Path('/Volumes/HUBSSD/code/EpiSim.jl')
    rosetta_path = project_root / 'models/mitma/rosetta_edar159.csv'
    metapop_path = project_root / 'models/mitma/metapopulation_data_edar159.csv'
    output_path = project_root / 'models/mitma/A0_initial_conditions_seeds_edar159.csv'

    # Load data (ensure ID is read as string to preserve leading zeros)
    rosetta_df = pd.read_csv(rosetta_path, dtype={'id': str})
    metapop_df = pd.read_csv(metapop_path, dtype={'id': str})

    # Merge rosetta with metapopulation data
    merged = rosetta_df.merge(metapop_df, on='id')

    # Calculate initial conditions (small fraction of each age group infected)
    # Using a fixed small number per municipality (same pattern as original)
    initial_conditions = []
    for _, row in merged.iterrows():
        # Use small initial infections proportional to population
        # Original pattern: Y=0.12-2.4, M=0.16-3.2, O=0.72-14.4
        y_init = max(0.12, row['Y'] * 0.0001)
        m_init = max(0.16, row['M'] * 0.0001)
        o_init = max(0.72, row['O'] * 0.0001)

        # Use the ID directly (already a string with leading zeros from rosetta)
        region_id = row['id']

        initial_conditions.append({
            'name': region_id,  # Use municipality code as name
            'id': region_id,
            'idx': row['idx'],
            'Y': round(y_init, 2),
            'M': round(m_init, 2),
            'O': round(o_init, 2)
        })

    result_df = pd.DataFrame(initial_conditions)
    result_df.to_csv(output_path, index=False)

    print(f"Created initial conditions file with {len(result_df)} regions")
    print(f"Output: {output_path}")
    print("\nSummary:")
    print(f"  Total Y infections: {result_df['Y'].sum():.2f}")
    print(f"  Total M infections: {result_df['M'].sum():.2f}")
    print(f"  Total O infections: {result_df['O'].sum():.2f}")
    print(f"  Total infections: {result_df[['Y', 'M', 'O']].sum().sum():.2f}")

    return result_df


if __name__ == '__main__':
    generate_initial_conditions_edar159()
