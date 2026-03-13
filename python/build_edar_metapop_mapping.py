#!/usr/bin/env python3
"""
Build EDAR-to-Metapopulation ID mapping for synthetic data generation.

This script creates a mapping from EDAR municipality 5-digit codes to
metapopulation region IDs, handling various ID format mismatches.
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_edar_municipalities(edar_nc_path: str) -> set:
    """Load EDAR municipality IDs from NetCDF file."""
    import netCDF4
    nc = netCDF4.Dataset(edar_nc_path)
    home_ids = nc.variables['home'][:]
    nc.close()
    return set(home_ids)


def load_metapop_data(metapop_csv_path: str) -> pd.DataFrame:
    """Load metapopulation data."""
    return pd.read_csv(metapop_csv_path)


def build_edar_metapop_mapping(edar_ids: set, metapop_df: pd.DataFrame) -> dict:
    """
    Build mapping from EDAR 5-digit codes to metapopulation IDs.

    Returns:
        dict with keys:
        - 'exact_match': EDAR ID -> list of metapop IDs (exact match)
        - 'with_suffix': EDAR ID -> list of metapop IDs (with _AM suffix removed)
        - 'multi_part': EDAR ID -> list of metapop IDs (multi-part codes aggregated)
        - 'no_match': list of EDAR IDs with no metapopulation data
    """
    metapop_ids = set(metapop_df['id'].values)

    exact_match = {}
    with_suffix = {}
    multi_part = defaultdict(list)
    no_match = []

    for edar_id in sorted(edar_ids):
        # Try exact match
        if edar_id in metapop_ids:
            exact_match[edar_id] = [edar_id]
            continue

        # Try with _AM suffix
        with_am_id = f"{edar_id}_AM"
        if with_am_id in metapop_ids:
            with_suffix[edar_id] = [with_am_id]
            continue

        # Try multi-part codes (e.g., 08015 -> 0801501, 0801502, ...)
        multi_part_ids = [mid for mid in metapop_ids if mid.startswith(edar_id) and len(mid) == len(edar_id) + 2]
        if multi_part_ids:
            multi_part[edar_id] = sorted(multi_part_ids)
            continue

        # No match found
        no_match.append(edar_id)

    return {
        'exact_match': exact_match,
        'with_suffix': with_suffix,
        'multi_part': dict(multi_part),
        'no_match': sorted(no_match),
        'summary': {
            'total_edar': len(edar_ids),
            'exact_match_count': len(exact_match),
            'with_suffix_count': len(with_suffix),
            'multi_part_count': len(multi_part),
            'no_match_count': len(no_match),
            'total_matched': len(exact_match) + len(with_suffix) + len(multi_part)
        }
    }


def aggregate_metapop_row(metapop_df: pd.DataFrame, metapop_ids: list) -> dict:
    """Aggregate multiple metapop rows into a single entry."""
    rows = metapop_df[metapop_df['id'].isin(metapop_ids)]

    return {
        'id': metapop_ids[0][:5],  # Use 5-digit EDAR code
        'area': rows['area'].sum(),
        'Y': rows['Y'].sum(),
        'M': rows['M'].sum(),
        'O': rows['O'].sum(),
        'total': rows['total'].sum(),
        'constituent_ids': metapop_ids
    }


def create_filtered_metapop_data(mapping: dict, metapop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create filtered metapopulation data with 159 EDAR-matched regions.

    For multi-part codes, aggregate population into single entries.
    """
    rows = []

    # Process exact matches
    for edar_id, metapop_ids in mapping['exact_match'].items():
        row = metapop_df[metapop_df['id'] == metapop_ids[0]].iloc[0].to_dict()
        row['id'] = edar_id  # Normalize to 5-digit code
        row['edar_match_type'] = 'exact'
        rows.append(row)

    # Process with suffix matches
    for edar_id, metapop_ids in mapping['with_suffix'].items():
        row = metapop_df[metapop_df['id'] == metapop_ids[0]].iloc[0].to_dict()
        row['id'] = edar_id  # Normalize to 5-digit code
        row['edar_match_type'] = 'suffix'
        rows.append(row)

    # Process multi-part codes (aggregate)
    for edar_id, metapop_ids in mapping['multi_part'].items():
        row = aggregate_metapop_row(metapop_df, metapop_ids)
        row['edar_match_type'] = 'multi_part'
        rows.append(row)

    return pd.DataFrame(rows)


def create_filtered_mobility_matrix(
    mapping: dict,
    metapop_df: pd.DataFrame,
    mobility_df: pd.DataFrame,
    rosetta_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create filtered mobility matrix with 1-based indices for 159 regions.

    For multi-part codes, sum outbound mobility from all constituent subregions.
    """
    # Build mapping from metapop ID to index in rosetta (1-based)
    metapop_to_idx = dict(zip(rosetta_df['id'], rosetta_df['idx']))

    # Create mapping from EDAR ID to metapop IDs
    edar_to_metapop = {}
    edar_to_metapop.update(mapping['exact_match'])
    edar_to_metapop.update(mapping['with_suffix'])
    edar_to_metapop.update(mapping['multi_part'])

    # Create new index mapping for filtered regions (1-based)
    edar_ids = sorted(edar_to_metapop.keys())
    new_idx_map = {edar_id: i + 1 for i, edar_id in enumerate(edar_ids)}

    rows = []
    for edar_id in edar_ids:
        metapop_ids = edar_to_metapop[edar_id]

        # Get all outbound mobility rows for this EDAR region's metapop IDs
        source_indices = [metapop_to_idx[mid] for mid in metapop_ids if mid in metapop_to_idx]

        for source_idx in source_indices:
            source_rows = mobility_df[mobility_df['source_idx'] == source_idx]

            for _, row in source_rows.iterrows():
                target_idx = row['target_idx']
                # Find the target metapop ID
                target_metapop_id = rosetta_df[rosetta_df['idx'] == target_idx]['id'].values
                if len(target_metapop_id) == 0:
                    continue
                target_metapop_id = target_metapop_id[0]

                # Find which EDAR ID this target belongs to
                target_edar_id = None
                for ed, metapops in edar_to_metapop.items():
                    if target_metapop_id in metapops:
                        target_edar_id = ed
                        break

                if target_edar_id is not None:
                    rows.append({
                        'source_idx': new_idx_map[edar_id],
                        'target_idx': new_idx_map[target_edar_id],
                        'ratio': row['ratio']
                    })

    # Aggregate duplicate source-target pairs
    result_df = pd.DataFrame(rows)
    result_df = result_df.groupby(['source_idx', 'target_idx'], as_index=False)['ratio'].sum()

    return result_df


def create_rosetta_edar159(mapping: dict) -> pd.DataFrame:
    """Create rosetta file mapping EDAR IDs to 1-based indices."""
    edar_ids = sorted(list(mapping['exact_match'].keys()) +
                     list(mapping['with_suffix'].keys()) +
                     list(mapping['multi_part'].keys()))

    return pd.DataFrame({
        'id': edar_ids,
        'idx': range(1, len(edar_ids) + 1)
    })


def main():
    """Main function to build EDAR-metapopulation mapping and create filtered datasets."""

    # Paths
    project_root = Path('/Volumes/HUBSSD/code/EpiSim.jl')
    edar_nc_path = project_root / 'edar_muni_edges.nc'
    metapop_csv_path = project_root / 'models/mitma/metapopulation_data.csv'
    mobility_csv_path = project_root / 'models/mitma/R_mobility_matrix.csv'
    rosetta_csv_path = project_root / 'models/mitma/rosetta.csv'

    output_dir = project_root / 'models/mitma'
    mapping_json_path = output_dir / 'edar_to_metapop_mapping.json'
    metapop_output_path = output_dir / 'metapopulation_data_edar159.csv'
    mobility_output_path = output_dir / 'R_mobility_matrix_edar159.csv'
    rosetta_output_path = output_dir / 'rosetta_edar159.csv'

    print("Loading data...")
    edar_ids = load_edar_municipalities(str(edar_nc_path))
    metapop_df = load_metapop_data(str(metapop_csv_path))
    mobility_df = pd.read_csv(str(mobility_csv_path))
    rosetta_df = pd.read_csv(str(rosetta_csv_path))

    print(f"EDAR municipalities: {len(edar_ids)}")
    print(f"Metapopulation regions: {len(metapop_df)}")

    print("\nBuilding EDAR-to-metapopulation mapping...")
    mapping = build_edar_metapop_mapping(edar_ids, metapop_df)

    print("\nMapping summary:")
    for key, value in mapping['summary'].items():
        print(f"  {key}: {value}")

    # Save mapping to JSON
    with open(mapping_json_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"\nSaved mapping to {mapping_json_path}")

    print("\nCreating filtered metapopulation data...")
    filtered_metapop_df = create_filtered_metapop_data(mapping, metapop_df)
    filtered_metapop_df = filtered_metapop_df[['id', 'area', 'Y', 'M', 'O', 'total']]
    # Ensure ID column is string type to preserve leading zeros
    filtered_metapop_df['id'] = filtered_metapop_df['id'].astype(str).str.zfill(5)
    filtered_metapop_df.to_csv(metapop_output_path, index=False)
    print(f"Saved {len(filtered_metapop_df)} regions to {metapop_output_path}")

    print("\nCreating filtered mobility matrix...")
    filtered_mobility_df = create_filtered_mobility_matrix(
        mapping, metapop_df, mobility_df, rosetta_df
    )
    filtered_mobility_df.to_csv(mobility_output_path, index=False)
    print(f"Saved {len(filtered_mobility_df)} mobility entries to {mobility_output_path}")

    print("\nCreating rosetta file...")
    rosetta_edar159_df = create_rosetta_edar159(mapping)
    # Ensure ID column is string type to preserve leading zeros
    rosetta_edar159_df['id'] = rosetta_edar159_df['id'].astype(str).str.zfill(5)
    rosetta_edar159_df.to_csv(rosetta_output_path, index=False)
    print(f"Saved rosetta with {len(rosetta_edar159_df)} entries to {rosetta_output_path}")

    print("\nVerifying output files...")
    # Verify that all region IDs in rosetta are in metapop data
    metapop_ids = set(filtered_metapop_df['id'].values)
    rosetta_ids = set(rosetta_edar159_df['id'].values)
    assert metapop_ids == rosetta_ids, "Mismatch between metapop and rosetta IDs"
    print("  ✓ Metapop and rosetta IDs match")

    # Verify that mobility indices are valid
    max_idx = len(rosetta_edar159_df)
    assert filtered_mobility_df['source_idx'].max() <= max_idx, "Invalid source_idx in mobility"
    assert filtered_mobility_df['target_idx'].max() <= max_idx, "Invalid target_idx in mobility"
    assert filtered_mobility_df['source_idx'].min() >= 1, "source_idx must be >= 1"
    assert filtered_mobility_df['target_idx'].min() >= 1, "target_idx must be >= 1"
    print("  ✓ Mobility matrix indices are valid (1-based)")

    # Verify that all rosetta IDs exist in EDAR data
    edar_ids_from_nc = load_edar_municipalities(str(edar_nc_path))
    assert rosetta_ids.issubset(edar_ids_from_nc), "Some rosetta IDs not in EDAR data"
    print("  ✓ All rosetta IDs exist in EDAR data")

    print("\n✓ All files created successfully!")


if __name__ == '__main__':
    main()
