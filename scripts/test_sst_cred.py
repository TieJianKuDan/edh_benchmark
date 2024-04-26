import sys

sys.path.append("./")

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from matplotlib.font_manager import FontProperties
from torch.nn.functional import mse_loss

from scripts.utils import view


def merge_filt(ds, name):
    name1 = ds[name].data[5539:14173, 0, :, :]
    name1 = np.nan_to_num(name1)
    name2 = ds[name].data[5539:14173, 1, :, :]
    name2 = np.nan_to_num(name2)
    sst = name1 + name2
    return sst

def filt_era5():
    # preserve data between 2022-8-19 and 2023-8-19
    era5 = xr.open_dataset(R"data\other\ERA5_2022_2023.nc")

    time = era5.time.data[5539:14173]

    u10 = merge_filt(era5, "u10")
    v10 = merge_filt(era5, "v10")
    d2m = merge_filt(era5, "d2m")
    t2m = merge_filt(era5, "t2m")
    msl = merge_filt(era5, "msl")
    sst = merge_filt(era5, "sst")

    era5_new = xr.Dataset(coords={
        "longitude": era5.longitude.data,
        "latitude": era5.latitude.data,
        "time": time
    })
    era5_new["u10"] = (["time", "latitude", "longitude"], u10)
    era5_new["v10"] = (["time", "latitude", "longitude"], v10)
    era5_new["d2m"] = (["time", "latitude", "longitude"], d2m)
    era5_new["t2m"] = (["time", "latitude", "longitude"], t2m)
    era5_new["msl"] = (["time", "latitude", "longitude"], msl)
    era5_new["sst"] = (["time", "latitude", "longitude"], sst)
    era5_new.to_netcdf(R"data\other\era5_with_raw_sst.nc")

def replace_sst():
    era5 = xr.open_dataset(R"data\other\era5_with_raw_sst.nc")
    tao = xr.open_dataset(R"data\other\TAO_T8N165E_DM445A-20220819_R_SST_10min.nc")
    
    sst_tao = tao.SST.data
    nans = np.isnan(sst_tao)
    sst_tao[nans] = np.interp(
        np.flatnonzero(nans), np.flatnonzero(~nans), sst_tao[~nans]
    )
    sst_tao = sst_tao + 273.15
    time_tao = tao.TIME.data
    time_tao = np.array([view.dt64todt(dt64) for dt64 in time_tao])
    time_tao_h = []
    sst_tao_h = []
    for i in range(len(time_tao)):
        if time_tao[i].minute == 0:
            time_tao_h.append(time_tao[i])
            sst_tao_h.append(sst_tao[i])

    time_era5 = era5.time.data
    time_era5 = np.array([view.dt64todt(dt64) for dt64 in time_era5])
    sst_era5 = era5.sst.data[:, 0, 0]
    for i in range(len(time_tao_h)):
        if i == 0:
            j = 0
        while j < len(time_era5):
            if time_tao_h[i].day == time_era5[j].day and \
                time_tao_h[i].hour == time_era5[j].hour:
                sst_era5[j] = sst_tao_h[i][0]
                break
            j += 1
    plt.plot(time_era5, sst_era5, "-")
    plt.plot(time_tao_h, sst_tao_h, "-")
    plt.legend(["ERA5", "TAO"])
    era5["sst"] = (["time", "latitude", "longitude"], sst_era5[:, None, None])
    era5.to_netcdf("data\other\era5_with_tao_sst.nc")

def compare_sst():
    tao = xr.open_dataset(R"data/other/TAO_T8N165E_DM445A-20220819_R_SST_10min.nc")
    era5 = xr.open_dataset(R"data/other/ERA5_2022_2023.nc")

    # TAO data
    sst_tao = tao.SST.data
    nans = np.isnan(sst_tao)
    sst_tao[nans] = np.interp(
        np.flatnonzero(nans), np.flatnonzero(~nans), sst_tao[~nans]
    )
    time_tao = tao.TIME.data
    time_tao = np.array([view.dt64todt(dt64) for dt64 in time_tao])

    daily_change = []
    for i in range(0, len(sst_tao), 144):
        daily_change.append(
            np.max(sst_tao[i:i+144]) - np.min(sst_tao[i:i+144])
        )
    plt.figure(figsize=(10, 5))
    plt.plot(time_tao[0::144], daily_change, "-o")
    plt.xlabel('Date', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.ylabel('Diurnal Variation of SST (℃)', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.xlim([datetime(2022, 8, 1), datetime(2023, 9, 1)])
    date_range = pd.date_range(
        start='2022-08-01',
        end='2023-09-01',
        freq='3M'
    )
    plt.xticks(date_range, fontsize=18, fontname='Times New Roman')
    plt.ylim((0, 1.5))
    plt.yticks([0, 0.5, 1, 1.5], fontsize=18, fontname='Times New Roman')
    # plt.savefig("imgs/sst_diurnal_variation.jpg", dpi=1000)

    sst_era5_1 = era5.sst.data[5539:14173, 0, 0, 0] - 273.15
    sst_era5_1 = np.nan_to_num(sst_era5_1)
    sst_era5_2 = era5.sst.data[5539:14173, 1, 0, 0] - 273.15
    sst_era5_2 = np.nan_to_num(sst_era5_2)
    sst_era5 = sst_era5_1 + sst_era5_2
    time_era5 = era5.time.data[5539:14173]
    time_era5 = np.array([view.dt64todt(dt64) for dt64 in time_era5])

    MSE = []
    MAX = []
    ratio = []
    for h in range(24):
        print(f"====={h}=====")
        time_tao_h = []
        sst_tao_h = []
        for i in range(len(time_tao)):
            if time_tao[i].hour == h \
                and time_tao[i].minute == 0:
                time_tao_h.append(time_tao[i])
                sst_tao_h.append(sst_tao[i])
   
        sst_era5_h = []
        for i in range(len(time_tao_h)):
            if i == 0:
                j = 0
            while j < len(time_era5):
                if time_tao_h[i].day == time_era5[j].day and \
                    time_tao_h[i].hour == time_era5[j].hour:
                    sst_era5_h.append(sst_era5[j])
                    break
                j += 1
        sst_era5_h = torch.tensor(sst_era5_h)[None, :]
        sst_tao_h = torch.tensor(sst_tao_h).T
        mse = mse_loss(sst_era5_h, sst_tao_h)
        max_error = torch.max(torch.abs(sst_era5_h - sst_tao_h))
        print(f"MSE: {mse}")
        print(f"MAX: {max_error}")
        MSE.append(mse)  
        MAX.append(max_error)
        ratio.append(torch.sum(torch.abs(sst_era5_h - sst_tao_h) > 0.5)*100 / sst_era5_h.shape[1])
    
    plt.figure(figsize=(10, 5))
    plt.bar(np.linspace(0, 23, 24), ratio, color="#4c72b2")
    plt.xlabel('Hour', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.ylabel('Ratio (%)', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.xlim([-1, 24])
    plt.xticks([0, 6, 12, 18, 23], fontsize=18, fontname='Times New Roman')
    plt.ylim((0, 5.1))
    plt.yticks([1, 2, 3, 4, 5], fontsize=18, fontname='Times New Roman')
    # plt.savefig("imgs/sst_ratio.jpg", dpi=1000)

    _, ax1 = plt.subplots(figsize=(10, 5))
    l1 = ax1.plot(MSE, "-*", color='red', markersize=10)
    plt.xlabel('Hour', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.ylabel('MSE (℃)', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.xlim([-1, 24])
    plt.xticks([0, 6, 12, 18, 23], fontsize=18, fontname='Times New Roman')
    plt.ylim((0.005, 0.06))
    plt.yticks([0.01, 0.025, 0.04, 0.055], fontsize=18, fontname='Times New Roman')

    ax2 = ax1.twinx()
    l2 = ax2.plot(MAX, "-o", color='orange', markersize=10)
    plt.ylabel('MAX (℃)', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.ylim((0.45, 1.2))
    plt.yticks([0.5, 0.7, 0.9, 1.1], fontsize=18, fontname='Times New Roman')

    lines = l1 + l2  
    font = FontProperties(family='Times New Roman', size=18, weight='normal') 
    ax1.legend(lines[:1], ["MSE"], loc='upper left', prop=font)
    ax2.legend(lines[1:], ["MAX"], loc='upper right', prop=font) 
    # plt.savefig("imgs/mse_max.jpg", dpi=1000)

def compare_edh():
    era5 = xr.open_dataset("data\other\edh_with_era5.nc")
    tao = xr.open_dataset("data\other\edh_with_tao.nc")
    edh_era5 = torch.tensor(era5.EDH.data[:, 0, 0])
    edh_tao = torch.tensor(tao.EDH.data[:, 0, 0])
    plt.plot(view.to_datetime(era5.time), edh_era5)
    plt.plot(view.to_datetime(tao.time), edh_tao)
    print(f"MSE: {mse_loss(edh_era5, edh_tao)}")
    print(f"MAX: {torch.max(torch.abs(edh_era5 - edh_tao))}")
    print(f"ratio: {torch.sum(torch.abs(edh_era5 - edh_tao) > 5)*100 / edh_tao.shape[0]}")

if __name__=="__main__":
    print("pause")