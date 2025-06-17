import sys
from datetime import datetime

import contextily as ctx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import oggm.cfg as cfg
import pandas as pd
import seaborn as sns
from dateutil.tz import UTC
from oggm.core import massbalance

from . import datacube_calib

try:
    import holoviews as hv

    hv.extension("bokeh")
except ImportError:
    pass


def get_default_hv_opts() -> dict:
    default_opts = {
        "aspect": 2,
        "active_tools": ["pan", "wheel_zoom"],
        "fontsize": {"title": 18},
        "fontscale": 1.2,
        "bgcolor": "white",
        "backend_opts": {"title.align": "center", "toolbar.autohide": True},
        "scalebar_opts": {"padding": 5},
        "show_frame": False,
        "margin": 0,
        "border": 0,
    }
    return default_opts


def get_figure_axes(ax=None, despine=True, showgrid="y", figsize=(8, 4)):
    if ax is None:
        fig = plt.figure(dpi=100, figsize=figsize, tight_layout=True)
        ax = plt.gca()
    else:
        fig = plt.gcf()
    if despine:
        sns.despine(top=True, right=True)
    if showgrid:
        ax.grid(visible=True, axis=showgrid, alpha=0.2)

    return fig, ax


def get_plot_data(gdir, mb_model_class=massbalance.MonthlyTIModel, period=(1979, 2019)):
    mbmod = mb_model_class(gdir)
    fls = gdir.read_pickle("model_flowlines")
    heights = fls[0].surface_h
    time_index = np.arange(period[0], period[1])
    return mbmod, heights, fls, time_index


def set_plot(x, y, label: str = "", ax=None):
    fig, ax = get_figure_axes(ax=ax)
    ax.plot(x, y, label=f"{label}")
    if label:
        plt.legend()
    ax.set_ylabel("MB (mm w.e.)")

    return fig, ax


def plot_annual_mb(
    gdir, mb_model_class=massbalance.MonthlyTIModel, period=(1979, 2019), ax=None
):
    mbmod, heights, fls, years = get_plot_data(
        gdir=gdir, mb_model_class=mb_model_class, period=period
    )
    mb_ts = (
        (mbmod.get_annual_mb(heights, year=period[0]))
        * cfg.SEC_IN_YEAR
        * cfg.PARAMS["ice_density"]
    )

    fig, ax = set_plot(x=mb_ts, y=heights, label=mb_model_class.__name__, ax=ax)
    ax.set_ylabel("Elevation (m a.s.l.)")
    ax.set_xlabel("MB (mm w.e. yr$^{-1}$)")
    ax.set_title("Model Comparison (Annual MB)")

    return fig, ax


def plot_monthly_mb(
    gdir, mb_model_class=massbalance.MonthlyTIModel, period=(1979, 2019), ax=None
):
    mbmod, heights, fls, years = get_plot_data(
        gdir=gdir, mb_model_class=mb_model_class, period=period
    )
    mb_ts = (
        (mbmod.get_monthly_mb(heights, year=period[0]))
        * cfg.SEC_IN_YEAR
        * cfg.PARAMS["ice_density"]
    )

    fig, ax = set_plot(x=mb_ts, y=heights, label=mb_model_class.__name__, ax=ax)
    ax.set_ylabel("Elevation (m a.s.l.)")
    ax.set_xlabel("MB (mm w.e. month$^{-1}$)")
    ax.legend()
    ax.set_title("Model Comparison (Monthly MB)")

    return fig, ax


def plot_daily_mb(
    gdir, mb_model_class=massbalance.DailyTIModel, period=(1979, 2019), ax=None
):
    mbmod, heights, fls, years = get_plot_data(
        gdir=gdir, mb_model_class=mb_model_class, period=period
    )
    mb_ts = (
        (mbmod.get_daily_mb(heights, year=period[0]))
        * cfg.SEC_IN_YEAR
        * cfg.PARAMS["ice_density"]
    )

    fig, ax = set_plot(x=mb_ts, y=heights, ax=ax)
    ax.set_ylabel("Elevation (m a.s.l.)")
    ax.set_xlabel("MB (mm w.e. day$^{-1}$)")
    # ax.legend()
    ax.set_title("Model Comparison (Daily MB)")

    return fig, ax


def plot_reference_mb(
    gdir, mb_model_class=massbalance.MonthlyTIModel, period=(1979, 2019), ax=None
):
    # mbmod, heights, fls, years = get_plot_data(gdir=gdir, mb_model_class=mb_model_class, period=period)
    ref_df = gdir.get_ref_mb_data()
    # ref_df['OGGM'] = mbmod.get_specific_mb(fls=fls, year=ref_df.index.values)

    fig, ax = get_figure_axes(ax=ax)
    # ax.plot(ref_df.index, ref_df.OGGM, label=f"OGGM ({mb_model_class.__name__})");
    ax.plot(
        ref_df.index, ref_df.ANNUAL_BALANCE, label="WGMS", color="black", linestyle="--"
    )
    ax.set_ylabel("Specific MB (mm w.e.)")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Year")
    ax.legend()

    return fig, ax


def plot_specific_mb(
    gdir, mb_model_class=massbalance.MonthlyTIModel, period=(1979, 2019), ax=None
):
    mbmod = massbalance.MultipleFlowlineMassBalance(gdir, mb_model_class=mb_model_class)
    fls = gdir.read_pickle("model_flowlines")
    years = np.arange(period[0], period[1])
    mb_ts = mbmod.get_specific_mb(fls=fls, year=years)

    fig, ax = get_figure_axes(ax=ax)
    ax.plot(years, mb_ts, label=f"{mb_model_class.__name__}", linewidth=0.8)
    ax.legend()
    ax.set_ylabel("Specific MB (mm w.e.)")
    ax.set_xlabel("Year")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title("Comparison (MultipleFlowlineMassBalance)")

    return fig, ax


def get_percentage_difference(a, b):
    return 100 * np.absolute(b - a) / ((a + b) / 2)


def get_bold(text: str, prefix="", suffix: str = "") -> str:
    bold_text = f"\033[1m{text}\033[0m"
    ordered = (prefix, bold_text, suffix)
    return " ".join(filter(None, ordered))


def plot_elevation_maps(gdir, ds):
    fig = plt.figure(figsize=(15, 12))
    gs = matplotlib.gridspec.GridSpec(2, 2, height_ratios=[2, 1])

    # Upper maps
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # Bottom timeseries axis spanning both columns
    ax2 = fig.add_subplot(gs[1, :])

    # Plot maps
    ds.itslive_v.plot(ax=ax0)
    ds.eolis_gridded_elevation_change[-1].plot(ax=ax1)

    glacier_outlines_gdf = gdir.read_shapefile("outlines")
    # Add glacier outlines and basemap to maps
    for ax in [ax0, ax1]:
        glacier_outlines_gdf.plot(ax=ax, facecolor="None", edgecolor="black")
        ctx.add_basemap(
            ax,
            source=ctx.providers.Esri.WorldImagery,
            crs=ds.pyproj_srs,
            zoom=8,
            attribution_size=4,
        )
        minx, miny, maxx, maxy = glacier_outlines_gdf.to_crs(ds.pyproj_srs).total_bounds
        buffer = 20000
        ax.set_xlim(minx - buffer, maxx + buffer)
        ax.set_ylim(miny - buffer, maxy + buffer)

    # plot aggregated time series
    ax2.axhline(ls="--", c="darkgrey")
    mean_time_series = [
        np.nanmean(elevation_change_map.where(ds.glacier_mask == 1))
        for elevation_change_map in ds.eolis_gridded_elevation_change
    ]
    dates = [datetime.fromtimestamp(t, tz=UTC) for t in ds.t.values]
    ax2.plot(
        dates, mean_time_series - mean_time_series[0], label="Monthly dh", color="k"
    )  # subtract first timestep to start time series at 0
    ax2.set_title(f"Elevation Change Over {gdir.name}")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Elevation Change [m]")
    cryoseries = pd.DataFrame({"dh": mean_time_series}, index=dates)
    cryoseries["dh"] = cryoseries["dh"] - cryoseries.iloc[0]["dh"]
    cryo_annual = (
        cryoseries.groupby([cryoseries.index.year])
        .mean()
        .rename_axis(
            index=["year"],
        )
        # .reset_index()
    )
    ax2.plot(
        pd.to_datetime(cryo_annual.index, format="%Y"), cryo_annual, label="Annual dh"
    )

    plt.tight_layout()
    # plt.show()
    return fig


def check_holoviews():
    if "holoviews" not in sys.modules:
        raise SystemError("Holoviews is not installed.")


def plot_monthly_mass_balance(
    smb: dict, years: list, glacier_name: str = "", gdir=None, datacube=None
):
    check_holoviews()
    plot_data = {}
    figures = []

    if gdir:
        wgms_data = gdir.get_ref_mb_data()["ANNUAL_BALANCE"]
        wgms_data.index = pd.to_datetime(wgms_data.index, format="%Y")
        rys = "2011"
        rye = "2019"
        label = f"WGMS ({(wgms_data.loc[rys:rye]).mean():.1f} kg m-2)"
        curve = hv.Curve(wgms_data, label=label).opts(line_width=0.8, color="k")
        figures.append(curve)

    if datacube:
        cryotempo_dates = datacube_calib.get_eolis_dates(datacube)
        cryotempo_dh = datacube_calib.get_eolis_mean_dh(datacube)

        cryotempo_mb = np.array(cryotempo_dh) * cfg.PARAMS["ice_density"]
        plot_data["CryoTEMPO_EOLIS"] = cryotempo_mb
        df = pd.DataFrame(cryotempo_mb, columns=["smb"], index=cryotempo_dates)
        label = f"CryoTEMPO-EOLIS ({cryotempo_mb.mean():.1f} kg m-2)"
        curve = hv.Curve(df, label=label).opts(
            line_width=1.0, color="grey", line_dash="dotted"
        )
        figures.append(curve)

        cryo_annual = (
            df.rename_axis(
                index=["date"],
            )
            .reset_index()
            .groupby(pd.Grouper(key="date", freq="1YS"))
            .mean()
        )
        label = f"CryoTEMPO-EOLIS Annual ({cryotempo_mb.mean():.1f} kg m-2)"
        curve = hv.Curve(cryo_annual, label=label).opts(line_width=1.0, color="purple")
        figures.append(curve)

    plot_dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="1D")
    for k, v in smb.items():
        if "Daily" in k:
            df = pd.DataFrame(v, columns=["smb"], index=plot_dates)
            df_monthly_smb = (
                df.rename_axis(
                    index=["date"],
                )
                .reset_index()
                .groupby(pd.Grouper(key="date", freq="1ME"))
                .sum()
            )
            plot_data[k] = df_monthly_smb["smb"]
            curve = hv.Curve(plot_data[k], label=k).opts(
                line_width=0.8
            )  # , color=hv.Cycle("Set1"))
            figures.append(curve)

            df_annual_smb = (
                df.rename_axis(
                    index=["date"],
                )
                .reset_index()
                .groupby(pd.Grouper(key="date", freq="1YE"))
                .sum()
            )
            df_annual_smb = df_annual_smb / len(df_annual_smb)
            label = f"{k} Annual ({df_annual_smb['smb'].sum():.1f} kg m-2)"
            curve = hv.Curve(df_annual_smb, label=label).opts(line_width=0.8)
            figures.append(curve)

    default_opts = get_default_hv_opts()
    if glacier_name:
        glacier_name = f"{glacier_name}, "
    overlay = (
        hv.Overlay(figures)
        .opts(**default_opts)
        .opts(
            aspect=4,
            ylabel="Monthly SMB (mm w.e.)",
            title=f"Monthly SMB for Different Calibration Periods\n {glacier_name}{years[0]}-{years[-1]+1}",
            xlabel="Date",
            # xformatter=f"%d",
            tools=["xwheel_zoom", "xpan"],
            active_tools=["xwheel_zoom"],
            legend_position="bottom_left",
            legend_opts={
                "orientation": "vertical",
                "css_variables": {"font-size": "1em", "display": "inline"},
            },
        )
    )
    layout = (
        hv.Layout([overlay])
        .cols(1)
        .opts(sizing_mode="stretch_width", shared_axes=False)
    )
    return layout


def plot_smb_bokeh(smb: dict, years, glacier_name: str = ""):
    check_holoviews()

    plot_data_daily = {}
    plot_data_annual = {}
    figures_year = []
    figures_day = []

    plot_dates_day = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="1D")
    plot_dates_year = pd.date_range(
        f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="1YS"
    )
    for k, v in smb.items():
        if "Daily" in k:
            df = pd.DataFrame(v, columns=["smb"], index=plot_dates_day)
            df_daily_mean = (
                df.groupby([df.index.day_of_year])
                .mean()
                .rename_axis(
                    index=["doy"],
                )
                # .reset_index()
            )
            plot_data_daily[k] = df_daily_mean["smb"]
            curve = hv.Curve(plot_data_daily[k], label=k).opts(
                line_width=0.8, color=hv.Cycle("Set1")
            )
            figures_day.append(curve)

            df_annual_mean = (
                df.groupby([df.index.year])
                .sum()
                .rename_axis(
                    index=["year"],
                )
                # .reset_index()
            )
            print(f"{k}: {df_annual_mean.mean()}")

        elif "Monthly" in k:
            # plot_dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="1YS")
            df = pd.DataFrame(v, columns=["smb"], index=plot_dates_year)
            df_annual_mean = (
                df.groupby([df.index.year])
                .mean()
                .rename_axis(
                    index=["year"],
                )
                # .reset_index()
            )

        plot_data_annual[k] = df_annual_mean["smb"]
        curve = hv.Curve(plot_data_annual[k], label=k).opts(
            line_width=0.8
        )  # , color=hv.Cycle("Set1"))
        figures_year.append(curve)
    # return plot_data_annual
    default_opts = get_default_hv_opts()
    if glacier_name:
        glacier_name = f"{glacier_name}, "
    overlay = hv.Overlay(figures_day).opts(**default_opts).opts(
        aspect=4,
        ylabel="Mean Daily SMB (mm w.e.)",
        title=f"Mean Daily SMB for Different Calibration Periods\n {glacier_name}{years[0]}-{years[-1]+1}",
        xlabel="Day of Year",
        xformatter=f"%d",
        tools=["xwheel_zoom", "xpan"],
        active_tools=["xwheel_zoom"],
        legend_position="bottom_left",
        legend_opts={
            "orientation": "vertical",
            "css_variables": {"font-size": "1em", "display": "inline"},
        },
    ) + hv.Overlay(figures_year).opts(**default_opts).opts(
        aspect=4,
        ylabel="Mean Annual SMB (mm w.e.)",
        title=f"Mean Annual SMB for Different Calibration Periods\n {glacier_name}{years[0]}-{years[-1]+1}",
        xlabel="Year",
        # xformatter=f"%d",
        tools=["xwheel_zoom", "xpan"],
        active_tools=["xwheel_zoom"],
        legend_position="bottom_left",
        legend_opts={
            "orientation": "vertical",
            "css_variables": {"font-size": "1em", "display": "inline"},
        },
    )
    layout = (
        hv.Layout([overlay])
        .cols(1)
        .opts(sizing_mode="stretch_width", shared_axes=False)
    )
    return layout


def plot_cumulative_smb(smb: dict, years, glacier_name: str = "", ref_year: int = 2015):
    check_holoviews()

    plot_data_daily = {}
    figures_day = []

    plot_dates_day = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="1D")
    plot_dates_ref_year = pd.date_range(
        f"{ref_year}-01-01", f"{ref_year}-12-31", freq="1D"
    )
    for k, v in smb.items():
        if "Daily" in k:
            df = pd.DataFrame(v, columns=["smb"], index=plot_dates_day)
            df_daily_mean = (
                df.groupby([df.index.day_of_year])
                .mean()
                .rename_axis(
                    index=["doy"],
                )
                # .reset_index()
            ).cumsum()
            plot_data_daily[k] = df_daily_mean["smb"]
            curve = hv.Curve(plot_data_daily[k], label=k).opts(
                line_width=0.8, color=hv.Cycle("Paired")
            )
            figures_day.append(curve)
            # return df
            df_ref = df.loc[plot_dates_ref_year].cumsum().rename_axis(index="doy")
            df_ref.index = df_ref.index.day_of_year

            label = f"{k} 2015"
            curve = hv.Curve(df_ref, label=label).opts(
                line_width=0.8, color=hv.Cycle("Paired")
            )
            figures_day.append(curve)
            # return df_ref, plot_data_daily[k]

    # return plot_data_annual
    default_opts = get_default_hv_opts()
    if glacier_name:
        glacier_name = f"{glacier_name}, "
    overlay = (
        hv.Overlay(figures_day)
        .opts(**default_opts)
        .opts(
            aspect=4,
            ylabel="Cumulative SMB (mm w.e.)",
            title=f"Cumulative SMB for Different Calibration Periods\n {glacier_name}{years[0]}-{years[-1]+1}",
            xlabel="Day of Year",
            xformatter=f"%d",
            tools=["xwheel_zoom", "xpan"],
            active_tools=["xwheel_zoom"],
            legend_position="bottom_left",
            legend_opts={
                "orientation": "vertical",
                "css_variables": {"font-size": "1em", "display": "inline"},
            },
        )
    )
    layout = (
        hv.Layout([overlay])
        .cols(1)
        .opts(sizing_mode="stretch_width", shared_axes=False)
    )
    return layout


def get_data_mean(data: np.ndarray, time_idx) -> pd.DataFrame:
    dataframe = pd.DataFrame(data, columns=["smb"], index=time_idx)
    dataframe_mean = (
        dataframe.groupby([dataframe.index.day_of_year])
        .mean()
        .rename_axis(
            index=["doy"],
        )
        # .reset_index()
    )
    return dataframe_mean


def get_label_from_key(key):

    key_split = key.split("_")
    model_name = key_split[0]
    if "Sfc" in model_name:
        model_name = f"{model_name.removesuffix('Sfc')}, Tracking"
    label = f"{model_name}, {key_split[1]}"

    if len(key_split) > 4:
        if key_split[2] != "month":
            label = f"{label} ({key_split[2]})"
    else:
        years = [i.split("-")[0] for i in key_split[-2:]]
        geo_period = "-".join(years)
        label = f"{label} ({geo_period})"
    return label


def get_date_mask(df, start_date, end_date):
    date_mask = (
        datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC) <= df.index
    ) & (df.index <= datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC))
    return date_mask


def get_mean_by_doy(dataframe) -> pd.DataFrame:
    return (
        dataframe.groupby([dataframe.index.day_of_year])
        .mean()
        .rename_axis(index=["doy"])
    )


def add_curve_to_figures(figures: list, data: dict, key: str, label: str = "") -> list:
    if not label:
        label = get_label_from_key(key)
    curve = hv.Curve(data[key], label=label).opts(line_width=0.8)
    figures.append(curve)
    return figures


def plot_mass_balance_lps(
    smb: dict,
    years,
    glacier_name: str = "",
    gdir=None,
    datacube=None,
    adjust_dates=False,
):
    check_holoviews()
    plot_data = {}
    figures = []

    if gdir:
        wgms_data = gdir.get_ref_mb_data()["ANNUAL_BALANCE"]
        wgms_data.index = pd.to_datetime(wgms_data.index, format="%Y")
        rys = "2011"
        rye = "2019"
        label = f"WGMS ({(wgms_data.loc[rys:rye]).mean():.1f} kg m-2)"
        # curve = hv.Curve(wgms_data, label=label).opts(line_width=0.8, color="k")
        # figures.append(curve)

    if datacube:
        cryotempo_dates = datacube_calib.get_eolis_dates(datacube)
        cryotempo_dh = datacube_calib.get_eolis_mean_dh(datacube)
        if adjust_dates:
            cryotempo_dates = pd.date_range(
                "2010-05-01", "2020-06-01", freq="1MS", tz=UTC
            )

        cryotempo_mb = (
            1000
            * np.array(cryotempo_dh - cryotempo_dh[0])
            * cfg.PARAMS["ice_density"]
            / gdir.rgi_area_km2
        )
        plot_data["CryoTEMPO_EOLIS"] = cryotempo_mb
        df = pd.DataFrame(cryotempo_mb, columns=["smb"], index=cryotempo_dates)

        # label = f"CryoTEMPO-EOLIS ({cryotempo_mb.mean():.1f} kg m-2)"
        label = f"CryoTEMPO-EOLIS"
        curve = hv.Curve(df, label=label).opts(
            line_width=1.0, color="grey", line_dash="dotted"
        )
        figures.append(curve)

        cryo_annual = (
            df.rename_axis(
                index=["date"],
            )
            .reset_index()
            .groupby(pd.Grouper(key="date", freq="1YS"))
            .mean()
        )
        date_mask = get_date_mask(df, "2010-01-01", "2020-01-01")
        # label = f"CryoTEMPO-EOLIS Annual ({cryotempo_mb[date_mask].mean():.1f} kg m-2)"
        label = f"CryoTEMPO-EOLIS, Annual Mean"
        curve = hv.Curve(cryo_annual, label=label).opts(line_width=1.0, color="purple")
        figures.append(curve)

    plot_dates_day = pd.date_range(
        f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="1D", tz=UTC
    )
    plot_dates_month = pd.date_range(
        f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="1MS", tz=UTC
    )
    for k, v in smb.items():
        if "Monthly" in k and "_month_" in k:
            df = pd.DataFrame(v, columns=["smb"], index=plot_dates_month)
            date_mask = get_date_mask(df, "2010-01-01", "2020-01-01")

            df_monthly_smb = (
                df[date_mask]
                .rename_axis(
                    index=["date"],
                )
                .reset_index()
                .groupby(pd.Grouper(key="date", freq="1MS"))
                .sum()
            )

            plot_data[k] = df_monthly_smb["smb"]
            # mean_smb = df[date_mask]["smb"].mean()
            # label = f"{label} [{mean_smb} kg m-2]"
            figures = add_curve_to_figures(figures=figures, data=plot_data, key=k)

        if "Daily" in k:

            df = pd.DataFrame(v, columns=["smb"], index=plot_dates_day)
            date_mask = get_date_mask(df, "2010-01-01", "2020-01-01")

            df_monthly_smb = (
                df[date_mask]
                .rename_axis(
                    index=["date"],
                )
                .reset_index()
                .groupby(pd.Grouper(key="date", freq="1MS"))
                .sum()
            )

            mean_smb = df[date_mask]["smb"].mean()
            label = get_label_from_key(k)
            # label = f"{label} [{mean_smb} kg m-2]"

            plot_data[k] = df_monthly_smb["smb"]
            figures = add_curve_to_figures(figures=figures, data=plot_data, key=k)

    default_opts = get_default_hv_opts()
    if glacier_name:
        glacier_name = f"{glacier_name}, "
    overlay = (
        hv.Overlay(figures)
        .opts(**default_opts)
        .opts(
            aspect=4,
            ylabel="Monthly SMB (mm w.e.)",
            title=f"Monthly SMB for Different TI Models\n {glacier_name}{2010}-{years[-1]+1}",
            xlabel="Date",
            # xformatter=f"%d",
            tools=["xwheel_zoom", "xpan"],
            active_tools=["xwheel_zoom"],
            legend_position="right",
            legend_opts={
                "orientation": "vertical",
                "css_variables": {"font-size": "1em", "display": "inline"},
            },
        )
    )
    layout = (
        hv.Layout([overlay])
        .cols(1)
        .opts(sizing_mode="stretch_width", shared_axes=False)
    )
    return layout


def plot_cumulative_smb_lps(
    smb: dict,
    years,
    glacier_name: str = "",
    ref_year: int = 2015,
    datacube=None,
    gdir=None,
):
    check_holoviews()

    plot_data_daily = {}
    figures_day = []

    plot_dates_day = pd.date_range(
        f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="1D", tz=UTC
    )
    plot_dates_month = pd.date_range(
        f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="1MS"
    )

    if datacube:
        cryotempo_dates = datacube_calib.get_eolis_dates(datacube)
        cryotempo_dh = datacube_calib.get_eolis_mean_dh(datacube)

        df = pd.DataFrame(cryotempo_dh, columns=["smb"], index=cryotempo_dates)

        date_mask = get_date_mask(df, f"{ref_year}-01-01", f"{ref_year+1}-01-01")
        df = df[date_mask]
        df["smb"] = (
            1000
            * (df["smb"] - df["smb"].iloc[0])
            * cfg.PARAMS["ice_density"]
            / gdir.rgi_area_km2
        )

        df_daily_mean = get_mean_by_doy(df).cumsum()
        plot_data_daily["CryoTEMPO-EOLIS Observations"] = df_daily_mean["smb"]

        label = f"CryoTEMPO-EOLIS Observations (2015)"
        curve = hv.Curve(
            plot_data_daily["CryoTEMPO-EOLIS Observations"], label=label
        ).opts(line_width=1.0, color="grey", line_dash="dotted")
        figures_day.append(curve)

    for k, v in smb.items():
        if "Daily" in k:
            label = get_label_from_key(k)

            df = pd.DataFrame(v, columns=["smb"], index=plot_dates_day)
            if "(2015)" in label:
                date_mask = get_date_mask(
                    df, f"{ref_year}-01-01", f"{ref_year+1}-01-01"
                )
                df = df[date_mask]
            df_daily_mean = get_mean_by_doy(df).cumsum()
            plot_data_daily[k] = df_daily_mean["smb"]

            figures_day = add_curve_to_figures(
                data=plot_data_daily, key=k, figures=figures_day
            )

        elif "Monthly" in k and "_month_" in k:
            df = pd.DataFrame(v, columns=["smb"], index=plot_dates_month)
            df_daily_mean = get_mean_by_doy(df).cumsum()
            plot_data_daily[k] = df_daily_mean["smb"]

            figures_day = add_curve_to_figures(
                data=plot_data_daily, key=k, figures=figures_day
            )

    default_opts = get_default_hv_opts()
    if glacier_name:
        glacier_name = f"{glacier_name}, "
    overlay = (
        hv.Overlay(figures_day)
        .opts(**default_opts)
        .opts(
            aspect=4,
            ylabel="Cumulative SMB (mm w.e.)",
            title=f"Cumulative SMB for Different Calibration Periods\n {glacier_name}{2010}-{2020}",
            xlabel="Day of Year",
            xformatter=f"%d",
            tools=["xwheel_zoom", "xpan"],
            active_tools=["xwheel_zoom"],
            legend_position="bottom_left",
            legend_opts={
                "orientation": "vertical",
                "css_variables": {"font-size": "1em", "display": "inline"},
            },
        )
    )
    layout = (
        hv.Layout([overlay])
        .cols(1)
        .opts(sizing_mode="stretch_width", shared_axes=False)
    )
    return layout
