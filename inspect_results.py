# -*- coding: utf-8 -*-
import sys
import logging

from dataclasses import dataclass
from typing import List, Union, Optional

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error

from pre_processing.cruts_config import CRUTSConfig


logging.basicConfig(level=logging.INFO)
logging.info("Python %s on %s" % (sys.version, sys.platform))

lats = [
    51.0571888888889,
    51.0422222222222,
    50.7934805555556,
    50.7901777777778,
    50.4172222222222,
    50.3986111111111,
    50.4443333333333,
    50.4421111111111,
    50.7614722222222,
    50.7603333333333,
    50.8071111111111,
    50.8091111111111,
    50.81275,
    50.7593611111111,
    50.6729722222222,
    50.6456666666667,
    50.6971666666667,
    50.8953611111111,
    50.8190277777778,
    50.8917222222222,
    50.96025,
    50.76033,
    51.04222,
]

lons = [
    15.478,
    15.6838888888889,
    15.6849972222222,
    15.6788361111111,
    16.48975,
    16.4648611111111,
    16.7188055555556,
    16.7666388888889,
    15.7469166666667,
    15.7261388888889,
    15.6083055555556,
    15.5868611111111,
    15.6107777777778,
    15.7611666666667,
    16.1326666666667,
    16.3668333333333,
    16.4656944444444,
    15.6205,
    15.5150555555556,
    15.3588611111111,
    15.4858888888889,
    15.72614,
    15.68389,
]

alts = [
    402,
    325,
    646,
    709,
    486,
    499,
    486,
    546,
    825,
    930,
    726,
    744,
    620,
    752,
    604,
    706,
    658,
    543,
    717,
    606,
    469,
    930,
    325,
]

ds_temp_nn = xr.open_dataset(
    "/media/xultaeculcis/2TB/datasets/cruts/inference-europe-extent-nc/esrgan.cru_ts4.04.nn.inference.1901.2019.tmp.dat.nc"
)
ds_temp_cru = xr.open_dataset(
    "/media/xultaeculcis/2TB/datasets/cruts/original/cru_ts4.04.1901.2019.tmp.dat.nc"
)
peaks = pd.read_csv("./datasets/mountain_peaks.csv")


@dataclass
class StatsResult:
    minima: np.ndarray
    means: np.ndarray
    medians: np.ndarray
    q25: np.ndarray
    q50: np.ndarray
    q75: np.ndarray
    maxima: np.ndarray

    @classmethod
    def empty(cls, size):
        return cls(
            minima=np.zeros(size),
            means=np.zeros(size),
            medians=np.zeros(size),
            q25=np.zeros(size),
            q50=np.zeros(size),
            q75=np.zeros(size),
            maxima=np.zeros(size),
        )


@dataclass
class CompareStatsResults:
    stats_cru: StatsResult
    stats_nn: StatsResult
    var: str
    ds_cru: xr.Dataset
    ds_nn: xr.Dataset
    time_range: pd.DatetimeIndex
    lats: Union[List, np.ndarray]
    lons: Union[List, np.ndarray]
    alts: Union[List, np.ndarray]
    names: Optional[Union[List, np.ndarray]]
    mse: float
    rmse: float
    mae: float

    @classmethod
    def compute(cls, var, time_range, lats, lons, alts, ds_cru, ds_nn, names=None):
        cru_stats = StatsResult.empty(len(lats))
        nn_stats = StatsResult.empty(len(lats))

        mae = np.zeros(len(lats))
        mse = np.zeros(len(lats))
        rmse = np.zeros(len(lats))

        for idx, (lat, lon) in enumerate(zip(lats, lons)):
            cru_data = ds_cru[var].sel(
                lat=lat, lon=lon, time=time_range, method="nearest"
            )

            cru_stats.q25[idx] = cru_data.quantile(0.25)
            cru_stats.q50[idx] = cru_data.quantile(0.5)
            cru_stats.q75[idx] = cru_data.quantile(0.75)
            cru_stats.minima[idx] = cru_data.min()
            cru_stats.maxima[idx] = cru_data.max()
            cru_stats.means[idx] = cru_data.mean()
            cru_stats.medians[idx] = cru_data.median()

            nn_data = ds_nn[var].sel(
                lat=lat, lon=lon, time=time_range, method="nearest"
            )

            nn_stats.q25[idx] = nn_data.quantile(0.25)
            nn_stats.q50[idx] = nn_data.quantile(0.5)
            nn_stats.q75[idx] = nn_data.quantile(0.75)
            nn_stats.minima[idx] = nn_data.min()
            nn_stats.maxima[idx] = nn_data.max()
            nn_stats.means[idx] = nn_data.mean()
            nn_stats.medians[idx] = nn_data.median()

            mae[idx] = mean_absolute_error(cru_data, nn_data)
            mse[idx] = mean_squared_error(cru_data, nn_data)
            rmse[idx] = mean_squared_error(cru_data, nn_data, squared=False)

        return cls(
            stats_cru=cru_stats,
            stats_nn=nn_stats,
            var=var,
            ds_cru=ds_cru,
            ds_nn=ds_nn,
            time_range=time_range,
            lats=lats,
            lons=lons,
            alts=alts,
            names=names,
            mae=mae.mean(),
            mse=mse.mean(),
            rmse=rmse.mean(),
        )

    def line_plot(self):
        plt.figure(figsize=(15, 15))
        ax = plt.subplot(1, 1, 1)
        for lat, lon in zip(self.lats, self.lons):
            cru_data = self.ds_cru[self.var].sel(
                lat=lat, lon=lon, time=self.time_range, method="nearest"
            )
            nn_data = self.ds_nn[self.var].sel(
                lat=lat, lon=lon, time=self.time_range, method="nearest"
            )
            cru_data.plot(marker="x", color="blue", ax=ax)
            nn_data.plot(marker="o", color="orange", ax=ax)

        ax.set_title("Temperature comparison between CRU-TS and SR across time")

        plt.gca().legend(("CRU-TS", "SR"))
        plt.show()

    def box_plot(self):
        plt.figure(figsize=(20, 10))
        values = []
        locations = []
        sources = []

        for idx, (lat, lon) in enumerate(zip(self.lats, self.lons)):
            cru_data = self.ds_cru[self.var].sel(
                lat=lat, lon=lon, time=self.time_range, method="nearest"
            )
            nn_data = self.ds_nn[self.var].sel(
                lat=lat, lon=lon, time=self.time_range, method="nearest"
            )

            values.extend(cru_data.values.tolist())
            values.extend(nn_data.values.tolist())
            sources.extend(["CRU-TS" for _ in range(len(nn_data))])
            sources.extend(["SR" for _ in range(len(nn_data))])

            if self.names is None:
                locations.extend(
                    [
                        f"#{idx} - {self.alts[idx]} m"
                        for _ in range(len(nn_data) + len(cru_data))
                    ]
                )
            else:
                locations.extend(
                    [
                        f"{self.names[idx]} - {self.alts[idx]} m"
                        for _ in range(len(nn_data) + len(cru_data))
                    ]
                )

        df = pd.DataFrame(
            data={
                "Location": locations,
                "Data source": sources,
                "Temperature (Celsius)": values,
            }
        )
        plt.figure(figsize=(np.maximum(0.25 * len(df["Location"].unique()), 20), 10))
        chart = sns.boxplot(
            x="Location", y="Temperature (Celsius)", data=df, hue="Data source"
        )
        chart.set_xticklabels(
            chart.get_xticklabels(), rotation=45, horizontalalignment="right"
        )
        plt.show()

    def print_comparison_summary(self):
        logging.info(
            f"Avg min CTS: {self.stats_cru.minima.mean()}, "
            f"NN: {self.stats_nn.minima.mean()}, "
            f"diff: {(self.stats_cru.minima - self.stats_nn.minima).mean()}"
        )
        logging.info(
            f"Avg mean CTS: {self.stats_cru.means.mean()}, "
            f"NN: {self.stats_nn.means.mean()}, "
            f"diff: {(self.stats_cru.means - self.stats_nn.means).mean()}"
        )
        logging.info(
            f"Avg median CTS: {self.stats_cru.medians.mean()}, "
            f"NN: {self.stats_nn.medians.mean()}, "
            f"diff: {(self.stats_cru.medians - self.stats_nn.medians).mean()}"
        )
        logging.info(
            f"Avg max CTS: {self.stats_cru.maxima.mean()}, "
            f"NN: {self.stats_nn.maxima.mean()}, "
            f"diff: {(self.stats_cru.maxima - self.stats_nn.maxima).mean()}"
        )
        logging.info(
            f"Avg q25 CTS: {self.stats_cru.q25.mean()}, "
            f"NN: {self.stats_nn.q25.mean()}, "
            f"diff: {(self.stats_cru.q25 - self.stats_nn.q25).mean()}"
        )
        logging.info(
            f"Avg q50 CTS: {self.stats_cru.q50.mean()}, "
            f"NN: {self.stats_nn.q50.mean()}, "
            f"diff: {(self.stats_cru.q50 - self.stats_nn.q50).mean()}"
        )
        logging.info(
            f"Avg q75 CTS: {self.stats_cru.q75.mean()}, "
            f"NN: {self.stats_nn.q75.mean()}, "
            f"diff: {(self.stats_cru.q75 - self.stats_nn.q75).mean()}"
        )
        logging.info(f"Mean Absolute Error between CRU-TS and SR-NN: {self.mae}")
        logging.info(f"Mean Squared Error between CRU-TS and SR-NN: {self.mse}")
        logging.info(f"Root Mean Squared Error between CRU-TS and SR-NN: {self.rmse}")


# data from Tomasz
logging.info("Tomasz's data")
may_only = pd.date_range(
    ds_temp_cru["time"][5].values, ds_temp_cru["time"][-1].values, freq="AS-MAY"
)

results = CompareStatsResults.compute(
    CRUTSConfig.tmp, may_only, lats, lons, alts, ds_temp_cru, ds_temp_nn
)
results.print_comparison_summary()
results.line_plot()
results.box_plot()

# mountain peaks
logging.info("Mountain peaks")
results = CompareStatsResults.compute(
    CRUTSConfig.tmp,
    may_only,
    peaks["lat"].values,
    peaks["lon"].values,
    peaks["altitude"].values,
    ds_temp_cru,
    ds_temp_nn,
    peaks["mountain_peak_name"].values,
)
results.print_comparison_summary()
results.line_plot()
results.box_plot()

logging.info("Only 2 locations")
results = CompareStatsResults.compute(
    CRUTSConfig.tmp, may_only, lats[-2:], lons[-2:], alts[-2:], ds_temp_cru, ds_temp_nn
)
results.print_comparison_summary()
results.line_plot()
results.box_plot()
