import json
import os
import sys

import numpy as np
import torch

sys.path.append("./")

import xarray as xr
from einops import rearrange


def save_era5(data, lon, lat, t, path):
    '''
    data: (c, t, h, w)
    '''
    with open('.cache/dist_static.json', 'r') as f:  
        dist = json.load(f)
    era5_new = xr.Dataset(coords={
        "longitude": lon,
        "latitude": lat,
        "time": t
    })
    era5_new["u10"] = (
        ["time", "latitude", "longitude"], 
        data[0] * dist["u10"]["std"] + dist["u10"]["mean"]
    )
    era5_new["v10"] = (
        ["time", "latitude", "longitude"], 
        data[1] * dist["v10"]["std"] + dist["v10"]["mean"]
    )
    era5_new["t2m"] = (
        ["time", "latitude", "longitude"], 
        data[2] * dist["t2m"]["std"] + dist["t2m"]["mean"]
    )
    era5_new["msl"] = (
        ["time", "latitude", "longitude"], 
        data[3] * dist["msl"]["std"] + dist["msl"]["mean"]
    )
    era5_new["sst"] = (
        ["time", "latitude", "longitude"], 
        data[4] * dist["sst"]["std"] + dist["sst"]["mean"]
    )
    era5_new["q2m"] = (
        ["time", "latitude", "longitude"], 
        data[5] * dist["q2m"]["std"] + dist["q2m"]["mean"]
    )
    era5_new.to_netcdf(path)

def save_edh(data, lon, lat, t, path):
    '''
    data: (c, t, h, w)
    '''
    edh_new = xr.Dataset(coords={
        "longitude": lon,
        "latitude": lat,
        "time": t
    })
    edh_new["EDH"] = (
        ["time", "latitude", "longitude"], 
        data[0]
    )
    edh_new.to_netcdf(path)

def save_nc(data:list, lon, lat, t, path, save_fun:callable):
    '''
    data[i]: (b, c, t, h, w)
    '''
    data = torch.cat(data, dim=0)
    data = rearrange(data, "b c t h w -> c (b t) h w")
    save_fun(data, lon, lat, t, path)

def load_edh_from_nc(edh_path):
    edh = []
    for root, _, files in os.walk(edh_path):  
        for filename in files:
            edh.append(
                xr.open_dataset(os.path.join(root, filename)).EDH
            )

    edh = sorted(
        edh, key=lambda edh: edh.time.data[0]
    )
    edh = [rearrange(edh.data, "(b t) h w -> t b h w", t=16) for edh in edh]
    edh = np.concatenate(edh, axis=1)
    edh = torch.tensor(edh)
    return edh