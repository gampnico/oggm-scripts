""""""

from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.tz import UTC
from oggm import GlacierDirectory, cfg, utils, workflow
from oggm.core import massbalance


def get_ref_mb_candidates(rgi_region: str = "06") -> list:
    """Get reference glaciers for Iceland."""
    candidates = utils.get_ref_mb_glaciers_candidates()
    return [ref for ref in candidates if f"-{rgi_region}." in ref]


def get_ref_gdir_by_attr(gdirs: list, key: str, value) -> GlacierDirectory:
    """Get the first glacier matching a specific attribute's value.

    Parameters
    ----------
    gdirs : list
        List of glacier directories.
    key : str
        Name of a ``GlacierDirectory`` attribute.
    value : Any
        Matching attribute value.
    """
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


def init_oggm(
    tempdir="OGGM-dmb-cryotempo",
    reset: bool = False,
    border: int = 80,
    use_multiprocessing: bool = True,
    store_model_geometry: bool = True,
):
    """Initialise OGGM."""
    cfg.initialize(logging_level="CRITICAL")
    cfg.PATHS["working_dir"] = utils.gettempdir(dirname=tempdir, reset=reset)
    cfg.PARAMS["border"] = 80
    cfg.PARAMS["use_multiprocessing"] = True
    cfg.PARAMS["store_model_geometry"] = True
    # cfg.PARAMS["evolution_model"] = "FluxBased"


def get_gdirs(rgi_ids: list = None, base_url: str = "", prepro_border: int = None):
    """Get glacier directories"""
    if not rgi_ids:
        rgi_ids = get_ref_mb_candidates()

    if not base_url:
        # base_url = "https://cluster.klima.uni-bremen.de/data/gdirs/dems_v2/default"
        # base_url = (
        #     "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/"
        #     "L3-L5_files/2023.3/elev_bands/W5E5"
        # )

        base_url = (
            "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/"
            "L3-L5_files/2025.1/elev_bands/W5E5_utm"
        )

    if not prepro_border:
        prepro_border = cfg.PARAMS["border"]

    gdirs = workflow.init_glacier_directories(
        rgi_ids,
        from_prepro_level=4,
        prepro_border=prepro_border,
        prepro_rgi_version="62",
        prepro_base_url=base_url,
    )
    # for glacier in gdirs:
    #     print(glacier.name)
    return gdirs


def get_gdir_for_calibration(gdirs, key: str, value) -> GlacierDirectory:
    """Get matching glacier directory for calibration.

    Parameters
    ----------
    gdirs : list
        List of glacier directories.
    key : str
        Name of a ``GlacierDirectory`` attribute.
    value : Any
        Matching attribute value.
    """
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


def get_calibrated_models(
    gdir,
    model_class,
    ref_mb,
    geodetic_period: str = "",
    years=None,
    model_calib=None,
    model_flowlines=None,
    smb=None,
    daily=False,
    calibration_filesuffix="",
    extra_model_kwargs=None,
):
    """Get calibrated models.

    Note this uses all three calibration parameters, with ``prcp_fac``
    as the first parameter.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
        Glacier directory
    model_class : oggm.MassBalanceModel
        Any mass balance model that subclasses MonthlyTIModel.
    ref_mb : pd.DataFrame
        Reference mass balance.
    geodetic_period : str, default empty string
        The reference calibration period in the format: "Y-M-D_Y-M-D"
    years : list, default None
        Years for which to calculate the specific mass balance. Ensure
        these are float years when using ``MonthlyTI``.
    model_calib : dict
        Store calibrated models derived from ``mb_model_class``
    model_flowlines : dict
        Store calibrated ``MultipleFlowlineMassBalanceModel``.
    smb : dict
        Store specific mass balance.
    daily : bool, default False
        Process daily specific mass balance.
    calibration_filesuffix : str, default empty string
        Calibration filesuffix

    Returns
    -------
    tuple
        Calibrated model instances for each calibration parameter,
        calibrated flowline models for each parameter,
        and specific mass balance for each calibrated flowline model.
    """
    model_name = model_class.__name__
    if not geodetic_period:
        geodetic_period = cfg.PARAMS["geodetic_mb_period"]

    if not daily:
        baseline_climate_suffix = ""
    else:
        baseline_climate_suffix = "_daily"

    if model_calib is None:
        model_calib = {}
    if model_flowlines is None:
        model_flowlines = {}
    if smb is None:
        smb = {}

    if not calibration_filesuffix:
        calibration_filesuffix = f"{model_name}_{geodetic_period}"
    model_key = calibration_filesuffix.removesuffix("Model")

    # This follows mb_calibration_from_geodetic_mb
    prcp_fac = massbalance.decide_winter_precip_factor(gdir)
    mi, ma = cfg.PARAMS["prcp_fac_min"], cfg.PARAMS["prcp_fac_max"]
    prcp_fac_min = utils.clip_scalar(prcp_fac * 0.8, mi, ma)
    prcp_fac_max = utils.clip_scalar(prcp_fac * 1.2, mi, ma)

    massbalance.mb_calibration_from_scalar_mb(
        gdir,
        ref_mb=ref_mb,
        ref_period=geodetic_period,
        calibrate_param1="prcp_fac",
        calibrate_param2="melt_f",
        calibrate_param3="temp_bias",
        prcp_fac=prcp_fac,
        prcp_fac_min=prcp_fac_min,
        prcp_fac_max=prcp_fac_max,
        mb_model_class=model_class,
        overwrite_gdir=True,
        use_2d_mb=False,
        baseline_climate_suffix=baseline_climate_suffix,
        filesuffix=calibration_filesuffix,
        extra_model_kwargs=extra_model_kwargs,
    )

    if not extra_model_kwargs:
        model_calib[model_key] = model_class(
            gdir, mb_params_filesuffix=calibration_filesuffix
        )
    else:
        model_calib[model_key] = model_class(
            gdir, mb_params_filesuffix=calibration_filesuffix, **extra_model_kwargs
        )
    model_flowlines[model_key] = massbalance.MultipleFlowlineMassBalance(
        gdir,
        mb_model_class=model_class,
        use_inversion_flowlines=True,
        mb_params_filesuffix=calibration_filesuffix,
    )
    fls = gdir.read_pickle("inversion_flowlines")
    if not years:
        years = np.arange(1979, 2020)
    if not daily:
        smb[model_key] = model_flowlines[model_key].get_specific_mb(fls=fls, year=years)
    else:
        smb[model_key] = model_flowlines[model_key].get_specific_mb_daily(
            fls=fls, year=years
        )

    return model_calib, model_flowlines, smb


def get_nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


def get_dhdt_data(ds, calendar: bool = False, year_start=2010, year_end=2020):
    """
    Parameters
    ----------
    ds : xr.DataArray
        OGGM dataset with EOLIS datacube.
    calendar : bool, default False
        Reference period is bound to calendar years. This will find the
        nearest available date to the selected first and last years.
    year_start : int
        Start of reference period.
    year_end : int
        End of reference period.
    """
    dates = get_eolis_dates(ds)
    elevation = np.array(
        [
            np.nanmean(elevation_change_map.where(ds.glacier_mask == 1))
            for elevation_change_map in ds.eolis_gridded_elevation_change
        ]
    )

    if calendar:
        year_start = datetime(year_start - 1, 12, 31, tzinfo=UTC)
        year_end = datetime(year_end, 1, 1, tzinfo=UTC)

        # EOLIS data is in 30-day periods, so adjust to nearest available date
        year_start = get_nearest(dates, year_start)
        year_end = get_nearest(dates, year_end)

        mask = (year_start <= dates) & (dates <= year_end)
        dates = dates[mask]
        elevation = elevation[mask]

    calib_frame = pd.DataFrame({"dh": elevation}, index=dates)
    # calib_frame["dh"] = calib_frame["elevation"] - calib_frame["elevation"].iloc[0]
    calib_frame["dt"] = calib_frame.index.diff()
    calib_frame["dt"] = calib_frame["dt"].astype("timedelta64[s]")
    calib_frame.loc[calib_frame.index[0], "dt"] = calib_frame["dt"].iloc[1]

    calib_frame["dhdt"] = calib_frame["dh"] / calib_frame["dt"].values.astype("float")
    return calib_frame


def get_dmdtda(gdir, calib_data):
    dhdt = calib_data["dh"].mean()

    da = gdir.rgi_area_km2
    dmdtda = dhdt * cfg.PARAMS["ice_density"] * da / 1e6
    return dmdtda


def get_geodetic_mb_from_dataset(
    gdir, ds, calendar: bool = False, year_start=2011, year_end=2020
) -> pd.DataFrame:
    calib_data = get_dhdt_data(
        ds, calendar=calendar, year_start=year_start, year_end=year_end
    )
    dmdtda = get_dmdtda(gdir, calib_data)
    dates = calib_data.index
    geodetic_mb_period = (
        f"{dates[0].strftime('%Y-%m-%d')}_{dates[-1].strftime('%Y-%m-%d')}"
    )
    geodetic_mb = {
        "rgiid": [gdir.rgi_id],
        "period": geodetic_mb_period,
        "area": gdir.rgi_area_m2,
        "dmdtda": dmdtda,
        "err_dmdtda": 0.0,
        "reg": 6,
        "is_cor": False,
    }

    return pd.DataFrame.from_records(geodetic_mb, index="rgiid")


def get_geodetic_mb_for_calibration(gdir, ds):
    pd_geodetic = utils.get_geodetic_mb_dataframe()
    for x in [True, False]:
        geodetic_mb = get_geodetic_mb_from_dataset(gdir, ds, calendar=x)
        pd_geodetic = pd.concat([pd_geodetic, geodetic_mb])
    geodetic_mb = get_geodetic_mb_from_dataset(
        gdir, ds, calendar=True, year_start=2015, year_end=2016
    )
    pd_geodetic = pd.concat([pd_geodetic, geodetic_mb])
    return pd_geodetic
