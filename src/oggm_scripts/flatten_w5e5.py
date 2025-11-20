"""Lilian Schuster

updated script version for flattening W5E5 v2023.2"""

# (for 11900 glaciers not the nearest climate gridpoint was chosen in v2022.2, but should now work
# in this v2023.2)

# - from: https://data.isimip.org/search/query/w5e5/climate_forcing/w5e5v2.0/climate_variable/pr/climate_variable/orog/climate_variable/tas/
# - download in terminal: `wget -i download_W5E5_script.txt`

###
# for the monthly files, I had to do that here to remove the time variable from lon/lat
# with xr.open_dataset(w5e5_files[n]) as _d:
#    _d['latitude'] = _d.latitude.isel(time=0, drop=True)
#    _d['longitude'] = _d.longitude.isel(time=0, drop=True)
#    _d.to_netcdf(isimip3a_files[n][:-3]+'x.nc')

import xarray as xr
import numpy as np
import pandas as pd
from oggm import utils

import sys
import matplotlib.pyplot as plt
import os


#### this part here is the same for all W5E5 / ISIMIP3a/ISIMIP3b files
ds_inv = xr.open_dataset("orog_W5E5v2.0.nc")
ds_inv = ds_inv.rename({"lat": "latitude"})
ds_inv = ds_inv.rename({"lon": "longitude"})
ds_inv.coords["longitude"] = np.where(
    ds_inv.longitude.values < 0, ds_inv.longitude.values + 360, ds_inv.longitude.values
)
ds_inv = ds_inv.sortby(ds_inv.longitude)
ds_inv.longitude.attrs["units"] = "degrees_onlypositive"

# get the dataset where coordinates of glaciers are stored
frgi = utils.file_downloader(
    "https://cluster.klima.uni-bremen.de/~oggm/rgi/rgi62_stats.h5"
)
odf = pd.read_hdf(frgi, index_col=0)

nx, ny = ds_inv.dims["longitude"], ds_inv.dims["latitude"]
# just make them into 0-> 360 scheme
cenlon_for_bins = np.where(odf["CenLon"] < 0, odf["CenLon"] + 360, odf["CenLon"])
# Nearest neighbor lookup
lon_bins = np.linspace(
    ds_inv.longitude.data[0] - 0.25, ds_inv.longitude.data[-1] + 0.25, nx + 1
)
# !!! attention W5E5 sorted from 90 to -90 !!!!
lat_bins = np.linspace(
    ds_inv.latitude.data[0] + 0.25, ds_inv.latitude.data[-1] - 0.25, ny + 1
)
# before it was wrong: (see below)
# lon_bins = np.linspace(0, 360, nx), lat_bins = np.linspace(90, -90, ny)
# which created a non-aligned bins, in addition there was one bin missing, creating a slightly
# larger resolution which after adding up a lot got problematic...
# at the end it resulted in some glaciers where the nearest grid point was not used

odf["lon_id"] = np.digitize(cenlon_for_bins, lon_bins) - 1
odf["lat_id"] = np.digitize(odf["CenLat"], lat_bins) - 1
odf["lon_val"] = ds_inv.longitude.data[odf.lon_id]
odf["lat_val"] = ds_inv.latitude.data[odf.lat_id]
# Use unique grid points as index and compute the area per location
odf["unique_id"] = [
    "{:03d}_{:03d}".format(lon, lat) for (lon, lat) in zip(odf["lon_id"], odf["lat_id"])
]
mdf = odf.drop_duplicates(subset="unique_id").set_index("unique_id")
mdf["Area"] = odf.groupby("unique_id").sum()["Area"]
print(
    "Total number of glaciers: {} and number of W5E5 gridpoints with glaciers in them: {}".format(
        len(odf), len(mdf)
    )
)

# this is the mask that we need to remove all non-glacierized gridpoints
mask = np.full((ny, nx), np.NaN)
mask[mdf["lat_id"], mdf["lon_id"]] = 1
ds_inv["glacier_mask"] = (("latitude", "longitude"), np.isfinite(mask))

# check the distance to the gridpoints-> it should never be larger than
diff_lon = ds_inv.longitude.data[odf.lon_id] - odf.CenLon
# if the distance is 360 -> it is the same as 0,
diff_lon = np.where(np.abs(diff_lon - 360) < 170, diff_lon - 360, diff_lon)
odf["ll_dist_to_point"] = (
    (diff_lon) ** 2 + (ds_inv.latitude.data[odf.lat_id] - odf.CenLat) ** 2
) ** 0.5
assert odf["ll_dist_to_point"].max() < (0.25**2 + 0.25**2) ** 0.5
### from here it is different for every script ...


make_daily = True
if make_daily:

    try:
        os.mkdir("tmp_pr")
        os.mkdir("tmp_tas")
    except:
        pass

    yss = [1979, 1981, 1991, 2001, 2011]
    yee = [1980, 1990, 2000, 2010, 2019]
    for var in ["pr", "tas"]:
        for ys, ye in zip(yss, yee):
            ds = xr.open_dataset("{}_W5E5v2.0_{}0101-{}1231.nc".format(var, ys, ye))

            ds = ds.rename({"lon": "longitude"}).rename({"lat": "latitude"})
            ds.coords["longitude"] = np.where(
                ds.longitude.values < 0, ds.longitude.values + 360, ds.longitude.values
            )

            ds = ds.sortby(ds.longitude)
            ds.longitude.attrs["units"] = "degrees_onlypositive"

            # this takes too long !!!
            # ds_merged_glaciers = xr_prcp.where(ds_inv['glacier_mask'], drop = True)

            ds["ASurf"] = ds_inv["orog"]  # ds_inv['ASurf']
            if ys == 1979 and var == "pr":

                # as we use here only those gridpoints where glaciers are involved, need to put the mask on dsi as well!
                # dsi = ds_inv.where(ds_inv['glacier_mask'], drop = True)  # this makes out of in total 6483600 points only 116280 points!!!
                dsi = ds.isel(time=[0]).where(ds_inv["glacier_mask"], drop=True)
                # we do not want any dependency on latitude and longitude
                dsif = dsi.stack(points=("latitude", "longitude")).reset_index(
                    ("points")
                )
                # dsif # so far still many points

                # drop the non-glacierized points
                dsifs = dsif.where(
                    np.isfinite(
                        dsi[var]
                        .stack(points=("latitude", "longitude"))
                        .reset_index(("time", "points"))
                    ),
                    drop=True,
                )
                # I have to drop the 'time_' dimension, to be equal to the era5_land example, because the invariant file should not have any time dependence !
                dsifs = dsifs.drop_vars(var)
                dsifs = dsifs.drop("time")
                dsifs = dsifs.drop("time_")
                dsifs.attrs["version"] = "2023.2"
                dsifs.to_netcdf(
                    "../flattened/2023.2/w5e5v2.0_glacier_invariant_flat.nc"
                )

                # check if gridpoint nearest to hef is right!!!
                # happened once that something went wrong here ...
                lon, lat = (10.7584, 46.8003)
                # orog = xr.open_dataset('/home/lilianschuster/Downloads/w5e5v2.0_glacier_invariant_flat.nc').ASurf
                orog = dsifs.ASurf
                c = (orog.longitude - lon) ** 2 + (orog.latitude - lat) ** 2
                surf_hef = orog.isel(points=c.argmin())
                np.testing.assert_allclose(surf_hef, 2250, rtol=0.1)

                # check wrong glacier in previous version
                # RGI60-19.00124
                lon, lat = (-70.8931 + 360, -72.4474)
                orog = dsifs.ASurf
                c = (orog.longitude - lon) ** 2 + (orog.latitude - lat) ** 2
                surf_g = orog.isel(points=c.argmin())
                np.testing.assert_allclose(surf_g.longitude, lon, rtol=0.2)
                np.testing.assert_allclose(surf_g.latitude, lat, rtol=0.2)

            ds = ds.drop_vars("ASurf")

            for t in ds.time.values[
                :
            ]:  # ds_merged_glaciers.time.values: .sel(time=slice('1901-04','2016-12'))
                # produce a temporary file for each month
                sel_l = ds.sel(time=[t])
                # don't do the dropping twice!!!!
                # sel = sel_l.where(ds_inv['glacier_mask'], drop = True)
                sel = sel_l.where(ds_inv["glacier_mask"])
                sel = sel.stack(points=("latitude", "longitude")).reset_index(
                    ("time", "points")
                )
                sel = sel.where(np.isfinite(sel[var]), drop=True)
                sel.to_netcdf("tmp_{}/tmp_{}.nc".format(var, str(t)[:10]))


###

# rename precipitation to 'pr'
# make yearly files
for var in ["pr", "tas"]:
    for y in np.arange(1979, 2020):
        dso_y = xr.open_mfdataset(
            "tmp_{}/tmp_{}-*.nc".format(var, str(y)),
            concat_dim="time",
            combine="nested",
        )  # .rename_vars({'time_':'time'})
        dso_y.to_netcdf("tmp_{}/tmp2_{}.nc".format(var, str(y)))
        dso_y.close()

# aggregate yearly files
for var in ["pr", "tas"]:
    dso_all2 = xr.open_mfdataset(
        "tmp_{}/tmp2_*.nc".format(var),
        concat_dim="time",
        combine="nested",
        parallel=True,
    ).rename_vars({"time_": "time"})

    dso_all2.attrs["history"] = (
        "longitudes to 0 -> 360,  only glacier gridpoints chosen and flattened latitude/longitude --> points"
    )
    dso_all2.attrs["postprocessing_date"] = str(np.datetime64("today", "D"))
    dso_all2.attrs["postprocessing_scientist"] = "lilian.schuster@student.uibk.ac.at"
    dso_all2.attrs["version"] = "2023.2"
    dso_all2.to_netcdf(
        "../flattened/2023.2/daily/w5e5v2.0_{}_global_daily_flat_glaciers_1979_2019.nc".format(
            var
        )
    )

# make monthly files
for var in ["pr", "tas"]:

    pathi = "../flattened/2023.2/daily/w5e5v2.0_{}_global_daily_flat_glaciers_1979_2019.nc".format(
        var
    )
    ds = xr.open_dataset(pathi)

    ds_monthly = ds.resample(time="MS").mean(keep_attrs=True)

    ds_monthly.attrs["postprocessing_scientist"] = "lilian.schuster@student.uibk.ac.at"
    ds_monthly.attrs["postprocessing_actions"] = (
        "using xarray: ds_monthly = ds.resample(time='MS', keep_attrs=True).mean(keep_attrs=True)\n"
        "ds_monthly.to_netcdf()\n"
    )

    ds_monthly.to_netcdf(
        "../flattened/2023.2/monthly/w5e5v2.0_{}_global_monthly_flat_glaciers_1979_2019.nc".format(
            var
        )
    )

    if var == "tas":
        # also compute monthly daily std:

        # attention- >this below is slightly wrong, it creates lon/ lat =0 because the std of lon/lat is zero
        # ds_tas_daily_std = _d.resample(time='MS').std(keep_attrs=True)
        # instead only do std over tas and add lat/lon afterwards again ...
        ds_tas_daily_std = ds.tas.resample(time="MS").std(keep_attrs=True)
        ds_tas_daily_std["latitude"] = ds.latitude
        ds_tas_daily_std["longitude"] = ds.longitude
        ds_tas_daily_std = ds_tas_daily_std.to_dataset()

        ds_tas_daily_std = ds_tas_daily_std.rename_vars(dict(tas="tas_std"))
        # now have to change variable tas to tas_std and its attributes
        ds_tas_daily_std.tas_std.attrs["standard_name"] = "air_temperature_daily_std"
        ds_tas_daily_std.tas_std.attrs["long_name"] = (
            "Near-Surface Air Temperature daily standard deviation"
        )
        ds_tas_daily_std.attrs["postprocessing_date"] = str(np.datetime64("today", "D"))
        ds_tas_daily_std.attrs["postprocessing_scientist"] = (
            "lilian.schuster@student.uibk.ac.at"
        )

        ds_tas_daily_std.to_netcdf(
            "../flattened/2023.2/monthly/w5e5v2.0_tas_std_global_monthly_flat_glaciers_1979_2019.nc"
        )


# import os
import glob

for var in ["pr", "tas"]:
    files = glob.glob("tmp_{}/tmp*".format(var))
    for f in files:
        os.remove(f)
