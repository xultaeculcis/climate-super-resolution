# -*- coding: utf-8 -*-
import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


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
