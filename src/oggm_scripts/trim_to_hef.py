"""Nicolas Gampierakis"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import xarray as xr


def get_user_arguments():

    tagline = "Trim flattened W5E5 files to HEF."
    parser = argparse.ArgumentParser(prog="trim-to-hef", description=tagline)
    # Required
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        dest="input",
        type=Path,
        metavar="<path>",
        required=True,
        help="path to flattened W5E5 data",
    )
    parser.add_argument(
        "-r",
        "--reference",
        default=None,
        dest="reference",
        type=Path,
        metavar="<path>",
        required=True,
        help="path to reference W5E5 data",
    )

    # Switches
    parser.add_argument(
        "-d",
        "--daily",
        action="store_true",
        default=None,
        dest="make_daily",
        help="daily resolution",
    )

    # Optional
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=Path,
        metavar="<path>",
        required=False,
        default=None,
        help="path to trimmed output file",
    )

    parser.add_argument(
        "-s",
        "--start",
        dest="start_year",
        type=int,
        metavar="<int>",
        required=False,
        default=None,
        help="start year of time series",
    )

    arguments = parser.parse_args()

    return arguments


def get_coords(dataset):
    for lat, lon in zip(dataset.latitude.values, dataset.longitude.values):
        print(lat, lon)


def main():

    args = get_user_arguments()

    input_path = args.input.expanduser()
    reference_path = args.reference.expanduser()
    if not args.output:
        output_path = input_path.with_name(
            input_path.stem + "_trim" + args.input.suffix
        )
    else:
        output_path = args.output.expanduser()

    with xr.open_dataset(input_path) as ds_full:
        ds_full.load()
    with xr.open_dataset(reference_path) as ds_ref:
        ds_ref.load()

    ds_trim = ds_full.copy()
    if args.start_year and isinstance(args.start_year, int):
        start_date = np.datetime64(datetime.date(args.start_year, 1, 1))
        time_mask = slice(start_date, None)
        ds_trim = ds_trim.sel(time=time_mask)
        ds_ref = ds_ref.sel(time=time_mask)

    ds_trim.attrs.update(
        {
            "postprocessing_date": f"{np.datetime64('today', 'D')}",
            "postprocessing_scientist": "nicolas.gampierakis@bristol.ac.uk",
            "version": "2.1",
        }
    )

    ds_trim = ds_trim.where(
        np.logical_and(
            ds_trim.latitude.isin(ds_ref.latitude.values[ds_ref.latitude.values > 0]),
            ds_trim.longitude.isin(
                ds_ref.longitude.values[ds_ref.longitude.values < 90]
            ),
        ),
        drop=True,
    )

    if "invariant" not in f"{input_path}":
        get_coords(dataset=ds_trim)

    if output_path.exists():
        user_input = input("File exists. Overwrite? (yes/no): ")
        if user_input.lower() in ("yes"):
            pass
        else:
            raise SystemExit("Cancelled file write.")

    print(f"Writing file to: {output_path}")
    compression = {"zlib": True, "complevel": 9}
    encoding = {var: compression for var in ds_trim.data_vars}
    print(output_path)
    ds_trim.to_netcdf(output_path, encoding=encoding)
    print("Finished write.")


if __name__ == "__main__":
    main()
