import argparse
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MobilityConverter")


def load_date_range(start_date, end_date, kappa0_csv=None):
    if start_date and end_date:
        return pd.date_range(start=start_date, end=end_date)

    if kappa0_csv:
        kappa_df = pd.read_csv(kappa0_csv)
        if "date" not in kappa_df.columns:
            raise ValueError("kappa0 CSV must include a 'date' column")
        return pd.to_datetime(kappa_df["date"]).drop_duplicates().sort_values()

    raise ValueError("Provide --start-date/--end-date or --kappa0-csv")


def build_dense_matrix(mobility_csv, metapop_csv):
    mob_df = pd.read_csv(mobility_csv)
    metapop_df = pd.read_csv(metapop_csv)

    ids = metapop_df["id"].astype(str).tolist()
    n_regions = len(ids)

    matrix = np.zeros((n_regions, n_regions), dtype=np.float64)

    source_idx = mob_df["source_idx"].to_numpy() - 1
    target_idx = mob_df["target_idx"].to_numpy() - 1
    ratios = mob_df["ratio"].to_numpy(dtype=np.float64)

    matrix[source_idx, target_idx] = ratios
    return ids, matrix


def build_dataset(dates, region_ids, matrix, chunk_size=241):
    mobility = np.broadcast_to(matrix, (len(dates),) + matrix.shape)

    dataset = xr.Dataset(
        {
            "mobility": (
                ("date", "origin", "target"),
                mobility,
            )
        },
        coords={
            "date": dates,
            "origin": region_ids,
            "target": region_ids,
        },
    )

    dataset["mobility"].encoding = {
        "chunksizes": (
            1,
            min(chunk_size, matrix.shape[0]),
            min(chunk_size, matrix.shape[1]),
        )
    }
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert static mobility matrix CSV to downstream zarr format."
    )
    parser.add_argument(
        "--mobility-csv",
        required=True,
        help="Path to mobility matrix CSV (source_idx,target_idx,ratio)",
    )
    parser.add_argument(
        "--metapop-csv",
        required=True,
        help="Path to metapopulation CSV with region ids",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for zarr store",
    )
    parser.add_argument(
        "--start-date",
        help="Start date for date axis (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        help="End date for date axis (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--kappa0-csv",
        help="Optional kappa0 CSV to infer dates",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=241,
        help="Chunk size for origin/target dimensions",
    )

    args = parser.parse_args()

    dates = load_date_range(args.start_date, args.end_date, args.kappa0_csv)
    region_ids, matrix = build_dense_matrix(args.mobility_csv, args.metapop_csv)

    dataset = build_dataset(dates, region_ids, matrix, args.chunk_size)

    output_path = args.output
    if os.path.exists(output_path):
        logger.info("Removing existing output at %s", output_path)
        if os.path.isdir(output_path):
            import shutil

            shutil.rmtree(output_path)
        else:
            os.remove(output_path)

    logger.info(
        "Writing mobility zarr to %s (dates=%s, regions=%s)",
        output_path,
        len(dates),
        len(region_ids),
    )
    dataset.to_zarr(output_path, mode="w", zarr_format=2)


if __name__ == "__main__":
    main()
