"""
Generate datacubes for DTCG
"""


import dtcg
import dtcg.integration.oggm_bindings as oggm_bindings
import oggm
from oggm.core import massbalance
import json
from pathlib import Path
from datetime import datetime, date, timezone, timedelta

import numpy as np
import xarray as xr

def load_ids_from_json(path: str) -> list:
    with open(path) as file:
        rgi_ids = json.load(file)
    assert isinstance(rgi_ids, list)
    return rgi_ids

def get_data(binder, rgi_ids: list):
    """Get dashboard data.

    Returns
    -------
    tuple
        Glacier directory, EOLIS-enhanced gridded data, and specific mass balance.
    """
    binder.init_oggm(dirname="gen-datacubes", reset=True)
    gdirs = binder.get_glacier_directories(
        rgi_ids=rgi_ids, from_prepro_level=4, prepro_border=80
    )
    print("Fetching OGGM data from shop...")

    binder.get_glacier_data(gdirs=gdirs)
    # workflow.execute_entity_task(
    #         gdirs=gdirs, task=w5e5.process_w5e5_data, daily=True
    #     )
    for gdir in gdirs:
        binder.set_flowlines(gdir)
    return gdirs

def calibration_hugonnet(binder, data):
    # Hugonnet
    for glacier_id, glacier in data.items():
        gdir = glacier["gdir"]
        for key in ["smb", "model"]:
            if key not in data[glacier_id].keys():
                glacier[key] = {}
        ref_mb = binder.calibrator.get_geodetic_mb(gdir=gdir, dataset=None)
        source = "Hugonnet"
        geo_period = "2010-01-01_2020-01-01"
        for oggm_model in [massbalance.DailyTIModel, massbalance.SfcTypeTIModel]:
            if issubclass(oggm_model, massbalance.SfcTypeTIModel):
                sfc_model_kwargs = {
                    "climate_resolution": "daily",
                }
            else:
                sfc_model_kwargs = {}
            binder.calibrator.set_model_matrix(
                name=f"{oggm_model.__name__}_{source}",
                model=oggm_model,
                geo_period=geo_period,
                daily=True,
                source=source,
                **sfc_model_kwargs,
            )
        mb_model_calib, mb_model_flowlines, smb = binder.calibrator.calibrate(
            model_matrix=binder.calibrator.model_matrix,
            gdir=gdir,
            ref_mb=ref_mb,
            # **sfc_model_kwargs,
        )

        glacier["smb"][source] = smb
        glacier["model"][source] = mb_model_flowlines
        binder.calibrator.model_matrix = {}
    return binder, data

def calibration_cryosat(binder, data):
    for glacier_id, glacier in data.items():
        gdir = glacier["gdir"]
        for key in ["smb", "model"]:
            if key not in data[glacier_id].keys():
                glacier[key] = {}
        if glacier["datacube"] is not None:
            ref_mb = binder.calibrator.get_geodetic_mb(
                gdir=gdir, dataset=glacier["datacube"].get_layer("L1")
            )
            source = "CryoTEMPO-EOLIS"
            geo_period = "2011-01-01_2020-01-01"
            for oggm_model in [massbalance.DailyTIModel]:  # , massbalance.SfcTypeTIModel]:
                if issubclass(oggm_model, massbalance.SfcTypeTIModel):
                    sfc_model_kwargs = {
                        "climate_resolution": "daily",
                    }
                else:
                    sfc_model_kwargs = {}

                binder.calibrator.set_model_matrix(
                    name=f"{oggm_model.__name__}_{source}",
                    model=oggm_model,
                    geo_period=geo_period,
                    daily=True,
                    source=source,
                    **sfc_model_kwargs,
                )
                # print(binder.calibrator.model_matrix)
            mb_model_calib, mb_model_flowlines, smb = binder.calibrator.calibrate(
                model_matrix=binder.calibrator.model_matrix,
                gdir=gdir,
                ref_mb=ref_mb,
                # **sfc_model_kwargs,
            )

            glacier["smb"][source] = smb
            glacier["model"][source] = mb_model_flowlines
            binder.calibrator.model_matrix = {}
        binder.calibrator.model_matrix = {}
    return binder, data


def main():

    base_url = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2025.6/elev_bands_w_data/W5E5/per_glacier/"
    output_dir = Path("./static/data/datacube_gen/")
    
    iceland_ids = load_ids_from_json(path=output_dir / "vatnajokull_rgi_ids.json")
    alpine_ids = load_ids_from_json(path=output_dir / "oetztal_rgi_ids.json")
    rgi_ids = set(alpine_ids + iceland_ids)

    binder = dtcg.integration.oggm_bindings.BindingsCryotempo()
    data = {}
    gdirs = get_data(binder, rgi_ids)
    for gdir in gdirs:
        data[gdir.rgi_id] = {}


    for gdir in gdirs:
        if "-06." in gdir.rgi_id:
            try:
                gdir, datacube = binder.get_eolis_data(gdir)
                data[gdir.rgi_id]["datacube"] = datacube
            except Exception as e:
                print(e)
                print(f"Failed to download Specklia for {gdir.rgi_id}.")
                data[gdir.rgi_id]["datacube"] = None
        else:
            data[gdir.rgi_id]["datacube"] = None
        data[gdir.rgi_id]["gdir"] = gdir

        binder.calibrator.model_matrix = {}
    binder, data = calibration_hugonnet(binder, data)
    binder, data = calibration_cryosat(binder, data)

if __name__ == "__main__":
    main()