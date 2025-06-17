""""""

from datetime import datetime

import numpy as np
from dateutil.tz import UTC
from oggm import cfg, utils, workflow
from oggm.core import massbalance


def get_ref_mb_candidates():
    candidates = utils.get_ref_mb_glaciers_candidates()
    return [ref for ref in candidates if "-06." in ref]


def get_ref_gdir_by_attr(gdirs: list, key: str, value):
    available_attrs = []

    for glacier in gdirs:
        if not hasattr(glacier, key):
            raise AttributeError(f"Attribute {key} not in gdir.")

        glacier_attribute = getattr(glacier, key)
        available_attrs.append(glacier_attribute)
        if isinstance(value, str):
            if value.lower() in glacier_attribute.lower():
                return glacier
    available_attrs = ", ".join(available_attrs)
    raise KeyError(f"{value} not found. Try {available_attrs}")


def init_oggm(tempdir="OGGM-dmb-cryotempo", reset: bool = False):
    cfg.initialize(logging_level="WARNING")
    cfg.PATHS["working_dir"] = utils.gettempdir(dirname=tempdir, reset=reset)
    cfg.PARAMS["border"] = 80
    cfg.PARAMS["use_multiprocessing"] = True
    cfg.PARAMS["store_model_geometry"] = True
    cfg.PARAMS["evolution_model"] = "FluxBased"


def get_gdirs(rgi_ids: list = None, base_url: str = ""):
    if not rgi_ids:
        rgi_ids = get_ref_mb_candidates()

    if not base_url:
        base_url = "https://cluster.klima.uni-bremen.de/data/gdirs/dems_v2/default"
        # base_url = (
        #     "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/"
        #     "L3-L5_files/2023.3/elev_bands/W5E5"
        # )

        # base_url = (
        #     "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/"
        #     "L3-L5_files/2025.1/elev_bands/W5E5_utm"
        # )

    gdirs = workflow.init_glacier_directories(
        rgi_ids,
        from_prepro_level=1,
        prepro_border=10,
        prepro_rgi_version="62",
        prepro_base_url=base_url,
    )
    for glacier in gdirs:
        print(glacier.name)
    return gdirs


def get_gdir_for_calibration(gdirs, key: str, value):
    try:
        gdir = get_ref_gdir_by_attr(gdirs=gdirs, key=key, value=value)
    except Exception as e:
        print(f"{e}")
        print("Selecting first available glacier directory.")
        gdir = gdirs[0]
    print(f"Path to the DEM: {gdir.get_filepath('glacier_mask')}")
    return gdir


def get_calibration_uncertainties(gdir, mb_calib_suffix: str = ""):
    calib_results = gdir.read_json("mb_calib", filesuffix=mb_calib_suffix)
    uncertainties = [
        f"Reference MB: {calib_results['reference_mb']} kg m-2\n"
        f"Reference MB Uncertainties: {calib_results['reference_mb_err']} kg m-2\n"
        f"Reference MB Time Period: {calib_results['reference_period']}\n"
    ]
    uncertainties = "".join(uncertainties)
    print(uncertainties)

    return uncertainties


def get_calibrated_results(gdir, mb_calib_suffix: str = "") -> list:
    calib_results = gdir.read_json("mb_calib", filesuffix=mb_calib_suffix)
    results = [
        f"melt_f: {calib_results['melt_f']}\n"
        f"prcp_fac: {calib_results['prcp_fac']}\n"
        f"temp_bias: {calib_results['temp_bias']}\n"
    ]

    results = "".join(results)
    print(results)

    return results


def get_specific_mb_data(
    gdir, mb_model_class=massbalance.MonthlyTIModel, period=(1979, 2019)
):
    mbmod = massbalance.MultipleFlowlineMassBalance(gdir, mb_model_class=mb_model_class)
    fls = gdir.read_pickle("model_flowlines")
    years = np.arange(period[0], period[1])
    mb_ts = mbmod.get_specific_mb(fls=fls, year=years)
    return mb_ts


def get_eolis_dates(ds):
    return np.array([datetime.fromtimestamp(t, tz=UTC) for t in ds.t.values])


def get_eolis_mean_dh(ds):
    mean_time_series = [
        np.nanmean(elevation_change_map.where(ds.glacier_mask == 1))
        for elevation_change_map in ds.eolis_gridded_elevation_change
    ]
    return mean_time_series
