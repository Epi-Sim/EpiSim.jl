"""
Append mobility series to a synthetic observations zarr store.

This script adds full time-varying mobility matrices to an existing zarr dataset
that was created by process_synthetic_outputs.py. It reads mobility series from
NPZ files and appends them as new variables to the zarr store.

Usage:
    python append_mobility_to_zarr.py \
        --zarr-path ../runs/synthetic_test/raw_synthetic_observations.zarr \
        --runs-dir ../runs/synthetic_test \
        --metapop-csv ../models/catalonia/data/Metapopulation_data.csv

Output format (new variables in zarr):
    mobility_series: (run_id, date, origin, target) - Full time-varying OD matrices
        This is the full mobility tensor for each run and timestep, stored in
        sparse edgelist format compatible with the model's internal representation.

    mobility_edgelist: (edge_id, 2) - The edgelist mapping [origin, destination]
        This maps each edge index to its origin-destination pair, allowing
        reconstruction of the full dense mobility matrix if needed.

Note: For large datasets (many runs × long time series), this can significantly
increase the zarr file size. Consider using factorized format (mobility_base +
mobility_kappa0) for storage efficiency if you only need mobility variations.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AppendMobility")


def sanitize_run_id(run_dir_name: str, max_length: int = 50) -> str:
    """Sanitize run directory name into a valid run_id."""
    run_id = run_dir_name[4:] if run_dir_name.startswith("run_") else run_dir_name
    run_id = run_id.replace("/", "_").replace("\\", "_").strip("_")
    run_id = run_id or "run"
    return run_id.ljust(max_length)[:max_length]


def find_mobility_files(runs_dir: Path) -> dict[str, Path]:
    """Find all mobility_series.npz files in run directories."""
    mobility_files = {}
    runs_path = Path(runs_dir)

    for run_dir in sorted(runs_path.glob("run_*")):
        if not run_dir.is_dir():
            continue

        mobility_path = run_dir / "mobility" / "mobility_series.npz"
        if mobility_path.exists():
            run_id = sanitize_run_id(run_dir.name)
            mobility_files[run_id] = mobility_path
        else:
            logger.debug(f"No mobility file found for {run_dir.name}")

    return mobility_files


def load_mobility_series(mobility_path: Path) -> dict:
    """Load mobility series from NPZ file."""
    data = np.load(mobility_path)
    return {
        "R_series": data["R_series"],  # (T, E)
        "edgelist": data["edgelist"],  # (E, 2)
        "T": int(data["T"]),
        "E": int(data["E"]),
        "M": int(data["M"]),
    }


def append_mobility_to_zarr(
    zarr_path: str,
    runs_dir: str,
    metapop_csv: str,
    chunk_size: int = 256,
    compressor: str = "zstd",
    compressor_level: int = 3,
):
    """Append mobility series to the zarr store."""
    zarr_path = Path(zarr_path)
    runs_path = Path(runs_dir)

    if not zarr_path.exists():
        raise ValueError(f"Zarr store does not exist: {zarr_path}")

    # Find all mobility files
    logger.info(f"Scanning {runs_dir} for mobility files...")
    mobility_files = find_mobility_files(runs_path)

    if not mobility_files:
        logger.warning("No mobility files found! Nothing to append.")
        return

    logger.info(f"Found {len(mobility_files)} mobility files")

    # Load existing zarr to get coordinates
    logger.info(f"Opening existing zarr store: {zarr_path}")
    existing_ds = xr.open_zarr(zarr_path)

    # Get run_ids and dates from existing dataset
    existing_run_ids = existing_ds["run_id"].values
    existing_dates = existing_ds["date"].values

    logger.info(f"Existing zarr has {len(existing_run_ids)} runs, {len(existing_dates)} dates")

    # Load metapopulation data to get region IDs
    metapop_df = pd.read_csv(metapop_csv, dtype={"id": str})
    metapop_df["id"] = metapop_df["id"].astype(str)
    region_ids = metapop_df["id"].tolist()
    M = len(region_ids)

    # Load first mobility file to get structure
    first_run_id = list(mobility_files.keys())[0]
    first_mobility = load_mobility_series(mobility_files[first_run_id])
    edgelist = first_mobility["edgelist"]
    E = first_mobility["E"]
    T = first_mobility["T"]

    logger.info(f"Mobility structure: M={M}, E={E}, T={T}")

    # Validate edgelist bounds
    max_edgelist_id = int(np.max(edgelist))
    if max_edgelist_id >= M:
        raise ValueError(
            f"Edgelist contains ID {max_edgelist_id} but metapop only has {M} regions"
        )

    # Filter mobility files to only those in existing zarr
    mobility_files = {
        run_id: path
        for run_id, path in mobility_files.items()
        if run_id in existing_run_ids
    }

    if len(mobility_files) < len(existing_run_ids):
        logger.warning(
            f"Only {len(mobility_files)}/{len(existing_run_ids)} runs have mobility data"
        )

    # Create mobility tensor (run_id, date, edge_id)
    # We use edge_id as the last dimension for efficient storage
    n_runs = len(mobility_files)
    n_dates = len(existing_dates)
    n_edges = E

    logger.info(f"Allocating mobility tensor: ({n_runs}, {n_dates}, {n_edges})")
    mobility_tensor = np.full((n_runs, n_dates, n_edges), np.nan, dtype=np.float32)

    # Load and populate mobility data
    run_id_to_idx = {rid: i for i, rid in enumerate(existing_run_ids)}
    run_ids_ordered = []

    for run_id, mobility_path in mobility_files.items():
        if run_id not in run_id_to_idx:
            continue

        run_idx = run_id_to_idx[run_id]
        run_ids_ordered.append(run_id)

        mobility_data = load_mobility_series(mobility_path)
        R_series = mobility_data["R_series"]  # (T, E)

        # Handle dimension mismatches
        T_mobility = R_series.shape[0]
        T_zarr = n_dates

        if T_mobility != T_zarr:
            logger.warning(
                f"Run {run_id}: mobility T={T_mobility} != zarr T={T_zarr}, "
                f"truncating/padding to match"
            )

        # Copy mobility data (truncate or pad to match zarr dates)
        T_copy = min(T_mobility, T_zarr)
        mobility_tensor[run_idx, :T_copy, :] = R_series[:T_copy, :]

        if run_idx % 10 == 0:
            logger.debug(f"Loaded mobility for run {run_idx}/{n_runs}")

    # Create xarray dataset
    logger.info("Creating xarray dataset for mobility...")

    # Create run_id coordinate (only runs with mobility)
    run_ids_with_mobility = [rid for rid in run_ids_ordered if rid in run_id_to_idx]
    run_indices = [run_id_to_idx[rid] for rid in run_ids_with_mobility]

    mobility_ds = xr.Dataset(
        {
            "mobility_series": (
                ("run_id", "date", "edge_id"),
                mobility_tensor[run_indices],  # (n_runs, n_dates, n_edges)
                {
                    "description": "Time-varying mobility routing probabilities in sparse edgelist format",
                    "long_name": "Mobility Routing Matrix R_ij(t)",
                },
            ),
            "mobility_edgelist": (
                ("edge_id", "dim_2"),
                edgelist,
                {
                    "description": "Sparse edgelist mapping edge_id to [origin, destination]",
                    "long_name": "Mobility Edgelist",
                },
            ),
        },
        coords={
            "run_id": run_ids_with_mobility,
            "date": existing_dates,
            "edge_id": np.arange(n_edges),
            "origin": region_ids,
            "target": region_ids,
        },
    )

    # Set compression
    if compressor == "zstd":
        from zarr import Blosc

        compressor_obj = Blosc(cname="zstd", clevel=compressor_level, shuffle=Blosc.SHUFFLE)
    elif compressor == "lz4":
        from zarr import Blosc

        compressor_obj = Blosc(cname="lz4", clevel=compressor_level, shuffle=Blosc.SHUFFLE)
    elif compressor == "blosc":
        from zarr import Blosc

        compressor_obj = Blosc(
            cname="blosclz", clevel=compressor_level, shuffle=Blosc.SHUFFLE
        )
    else:
        compressor_obj = None

    # Set chunk sizes
    edge_chunk = min(chunk_size, n_edges)
    date_chunk = min(chunk_size, n_dates)

    mobility_ds["mobility_series"].encoding = {
        "chunksizes": (1, date_chunk, edge_chunk),
        "compressor": compressor_obj,
    }
    mobility_ds["mobility_edgelist"].encoding = {
        "chunksizes": (-1, -1),
        "compressor": compressor_obj,
    }

    # Append to zarr
    logger.info(f"Appending mobility data to {zarr_path}...")
    mobility_ds.to_zarr(zarr_path, mode="a", zarr_format=2)

    logger.info("Done!")
    logger.info(f"Added mobility_series: ({len(run_ids_with_mobility)}, {n_dates}, {n_edges})")
    logger.info(f"Added mobility_edgelist: ({n_edges}, 2)")


def main():
    parser = argparse.ArgumentParser(
        description="Append mobility series to a synthetic observations zarr store."
    )
    parser.add_argument(
        "--zarr-path",
        required=True,
        help="Path to existing zarr store (output from process_synthetic_outputs.py)",
    )
    parser.add_argument(
        "--runs-dir",
        required=True,
        help="Directory with run_* folders containing mobility_series.npz files",
    )
    parser.add_argument(
        "--metapop-csv",
        required=True,
        help="Path to metapopulation_data.csv (for region ID mapping)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size for date/edge dimensions (default: 256)",
    )
    parser.add_argument(
        "--compressor",
        default="zstd",
        choices=["zstd", "lz4", "blosc", "none"],
        help="Compressor for zarr (default: zstd)",
    )
    parser.add_argument(
        "--compressor-level",
        type=int,
        default=3,
        help="Compression level (default: 3)",
    )

    args = parser.parse_args()

    append_mobility_to_zarr(
        zarr_path=args.zarr_path,
        runs_dir=args.runs_dir,
        metapop_csv=args.metapop_csv,
        chunk_size=args.chunk_size,
        compressor=args.compressor,
        compressor_level=args.compressor_level,
    )


if __name__ == "__main__":
    main()
