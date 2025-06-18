"""LPS Dashboard"""

import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dtcg.datacube import cryotempo_eolis
from oggm import cfg, tasks, workflow
from oggm.core import massbalance
from oggm.shop import its_live, w5e5
from tqdm import tqdm

from . import datacube_calib, plotting

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.Set1.colors)

try:
    import holoviews as hv

    hv.extension("bokeh")
except ImportError:
    pass


def initialise_dashboard_model(rgi_ids: list = None):
    datacube_calib.init_oggm(tempdir="OGGM-lps-demo", reset=False)
    gdirs = datacube_calib.get_gdirs(rgi_ids=rgi_ids)

    if rgi_ids:
        gdir = datacube_calib.get_gdir_for_calibration(gdirs, "rgi_id", rgi_ids[0])
    else:
        gdir = datacube_calib.get_gdir_for_calibration(gdirs, "name", "Bruarjoekull")
    gdirs = [gdir]

    workflow.execute_entity_task(tasks.glacier_masks, gdirs)
    workflow.execute_entity_task(its_live.itslive_velocity_to_gdir, gdirs)

    # add daily data
    workflow.execute_entity_task(gdirs=[gdir], task=w5e5.process_w5e5_data)
    workflow.execute_entity_task(gdirs=[gdir], task=w5e5.process_w5e5_data, daily=True)

    # Check data exists
    path_day = gdir.get_filepath("climate_historical_daily")
    path_month = gdir.get_filepath("climate_historical")
    for path in [path_day, path_month]:
        assert os.path.exists(path)

    return gdir


def get_eolis_data(gdir):
    with xr.open_dataset(gdir.get_filepath("gridded_data")) as datacube:
        datacube = datacube.load()

    DataCubeManager = cryotempo_eolis.DatacubeCryotempoEolis()
    DataCubeManager.retrieve_prepare_eolis_gridded_data(
        oggm_ds=datacube, grid=gdir.grid
    )
    return gdir, datacube


def set_flowlines(gdir):
    if not os.path.exists(gdir.get_filepath("elevation_band_flowline")):
        tasks.elevation_band_flowline(gdir=gdir, preserve_totals=True)
    if not os.path.exists(gdir.get_filepath("inversion_flowlines")):
        tasks.fixed_dx_elevation_band_flowline(gdir, preserve_totals=True)


def run_calibration(model_matrix, gdir, ref_mb):
    # Store results
    mb_model_calib = {}
    mb_model_flowlines = {}
    smb_daily = {}

    for matrix_name, model_params in tqdm(model_matrix.items()):
        daily = model_params["daily"]
        mb_model_class = model_params["model"]
        geo_period = model_params["geo_period"]
        source = model_params["source"]
        calibration_filesuffix = f"{matrix_name}_{geo_period}"
        cfg.PARAMS["geodetic_mb_period"] = geo_period

        mb_geodetic = (
            ref_mb.loc[
                np.logical_and(ref_mb["source"] == source, ref_mb.period == geo_period)
            ].dmdtda.values[0]
            * 1000
        )

        mb_model_calib, mb_model_flowlines, smb_daily = (
            datacube_calib.get_calibrated_models(
                gdir=gdir,
                model_class=mb_model_class,
                ref_mb=mb_geodetic,
                geodetic_period=cfg.PARAMS["geodetic_mb_period"],
                model_calib=mb_model_calib,
                model_flowlines=mb_model_flowlines,
                smb=smb_daily,
                daily=daily,
                calibration_filesuffix=calibration_filesuffix,
                extra_model_kwargs=model_params.get("extra_kwargs", None),
            )
        )

    return mb_model_calib, mb_model_flowlines, smb_daily


def set_calibration(gdir, datacube):

    ref_mb = datacube_calib.get_geodetic_mb_for_calibration(gdir=gdir, ds=datacube)
    sfc_model_kwargs = {
        "resolution": "day",
        "gradient_scheme": "annual",
        "check_data_exists": False,
    }
    model_matrix = {
        "DailySfc_Cryosat": {
            "model": massbalance.DailySfcTIModel,
            "geo_period": "2011-01-01_2020-01-01",
            "daily": True,
            "source": "CryoTEMPO-EOLIS",
            "extra_kwargs": sfc_model_kwargs,
        },
    }

    mb_model_calib, mb_model_flowlines, smb_daily = run_calibration(
        model_matrix=model_matrix, gdir=gdir, ref_mb=ref_mb
    )
    return mb_model_calib, mb_model_flowlines, smb_daily


def get_data(rgi_id: str = "RGI60-06.00377"):
    print("Initialising OGGM...")
    gdir = initialise_dashboard_model(rgi_ids=[rgi_id])
    print("Streaming data from Specklia...")
    gdir, datacube = get_eolis_data(gdir)
    print("Checking flowlines...")
    set_flowlines(gdir)
    print("Running calibration...")
    base_model, flowline_model, smb = set_calibration(gdir, datacube)
    return gdir, datacube, smb


def plot_data(smb: dict, gdir, datacube, ref_year: int = 2017, resample: bool = False):
    layout = plotting.plot_mb_comparison_lps(
        smb=smb,
        ref_year=ref_year,
        glacier_name=gdir.name,
        datacube=datacube,
        gdir=gdir,
        resample=resample,
    )
    return layout


def main():
    pass


if __name__ == "__main__":
    main()
