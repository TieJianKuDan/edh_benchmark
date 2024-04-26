import gc
import json
import os

import numpy as np
import xarray as xr
from torch.utils.data import Dataset


class TD(Dataset):
    """
    abstract class
    """
    def __init__(self) -> None:
        super(TD, self).__init__()

    def normlize(self, data:np.ndarray, name:str):
        # handle nan
        data_ = data[~np.isnan(data)]
        if self.flag == "fit":
            self.dist[name] = {
                "mean": float(data_.mean()),
                "std": float(data_.std())
            }
        mean = self.dist[name]["mean"]
        std = self.dist[name]["std"]
        data = np.nan_to_num(data, nan=mean)
        return (data - mean) / std

    def _handle_edh(self, _edh:list):
        edh = [None] * len(_edh)
        time = [None] * len(_edh)
        for i in range(len(_edh)):
            edh[i] = _edh[i].data
            time[i] = _edh[i].time.data
        edh = np.concatenate(edh, axis=0)
        # edh = edh[:, 36, 36][:, None, None]
        edh = self.normlize(edh, "edh")
        time = np.concatenate(time, axis=0)
        return (edh, time)

    def _handle_u10(self, _era5:list):
        u10 = [None] * len(_era5)
        for i in range(len(_era5)):
            u10[i] = _era5[i].u10.data
        u10 = np.concatenate(u10, axis=0)
        # u10 = u10[:, 36, 36][:, None, None]
        # delete NaN and Normalize
        u10 = self.normlize(u10, "u10")
        return u10

    def _handle_v10(self, _era5:list):
        v10 = [None] * len(_era5)
        for i in range(len(_era5)):
            v10[i] = _era5[i].v10.data
        v10 = np.concatenate(v10, axis=0)
        # v10 = v10[:, 36, 36][:, None, None]
        # delete NaN and Normalize
        v10 = self.normlize(v10, "v10")
        return v10

    def _handle_t2m(self, _era5:list):
        t2m = [None] * len(_era5)
        for i in range(len(_era5)):
            t2m[i] = _era5[i].t2m.data
        t2m = np.concatenate(t2m, axis=0)
        # t2m = t2m[:, 36, 36][:, None, None]
        # delete NaN and Normalize
        t2m = self.normlize(t2m, "t2m")
        return t2m

    def _handle_msl(self, _era5:list):
        msl = [None] * len(_era5)
        for i in range(len(_era5)):
            msl[i] = _era5[i].msl.data
        msl = np.concatenate(msl, axis=0)
        # msl = msl[:, 36, 36][:, None, None]
        # delete NaN and Normalize
        msl = self.normlize(msl, "msl")
        return msl

    def _handle_sst(self, _era5:list):
        sst = [None] * len(_era5)
        for i in range(len(_era5)):
            sst[i] = _era5[i].sst.data
        sst = np.concatenate(sst, axis=0)
        # sst = sst[:, 36, 36][:, None, None]
        # delete NaN and Normalize
        sst = self.normlize(sst, "sst")
        return sst

    def _handle_q2m(self, _era5:list):
        q2m = [None] * len(_era5)
        for i in range(len(_era5)):
            q2m[i] = _era5[i].q2m.data
        q2m = np.concatenate(q2m, axis=0)
        # q2m = q2m[:, 36, 36][:, None, None]
        # delete NaN and Normalize
        q2m = self.normlize(q2m, "q2m")
        return q2m

class EDHDataset(TD):

    def __init__(self, root, seq_len) -> None:
        super(EDHDataset, self).__init__()
        # load data
        _edh = []
        for root, _, files in os.walk(root):  
            for filename in files:
                _edh.append(
                    xr.open_dataset(os.path.join(root, filename)).EDH
                )
        _edh = sorted(
            _edh, key=lambda _edh: _edh.time.data[0]
        )
        self.lon = _edh[0].longitude.data
        self.lat = _edh[0].latitude.data
        edh = [None] * len(_edh)
        time = [None] * len(_edh)
        for i in range(len(edh)):
            edh[i] = _edh[i].data
            time[i] = _edh[i].time.data

        edh = np.concatenate(edh, axis=0)
        edh = edh[:, 36, 36][:, None, None]
        edh = np.expand_dims(edh, -1)
        time = np.concatenate(time, axis=0)

        # slide window
        self.edh_seq = []
        self.time_seq = []
        self.seq_len = seq_len
        for i in range(edh.shape[0] - self.seq_len + 1):
            self.edh_seq.append(edh[i : i + self.seq_len])
            self.time_seq.append(time[i : i + self.seq_len])

    def __len__(self) -> int:
        return len(self.time_seq)

    def __getitem__(self, index: int):
        return self.edh_seq[index]

class ERA5Dataset(TD):

    def __init__(self, edh, era5, seq_len, flag="fit"):
        super().__init__()
        self.flag = flag
        if flag == "fit":
            self.dist = {}
        else:
            # load mean and std
            with open('.cache/dist_static.json', 'r') as f:  
                self.dist = json.load(f)
        # load edh(time, lat, lon)
        _edh = []
        for root, _, files in os.walk(edh):  
            for filename in files:
                _edh.append(
                    xr.load_dataset(os.path.join(root, filename)).EDH
                )
        _edh = sorted(
            _edh, key=lambda _edh: _edh.time.data[0]
        )
        self.lon = _edh[0].longitude.data
        self.lat = _edh[0].latitude.data
        edh, time = self._handle_edh(_edh)
        gc.collect()
        # slide window
        self.edh_seq = [None] * (edh.shape[0] - seq_len + 1)
        self.time_seq = [None] * len(self.edh_seq)
        for i in range(len(self.edh_seq)):
            self.edh_seq[i] = edh[i : i + seq_len]
            self.time_seq[i] = time[i : i + seq_len]

        # load era5=[time, lat, lon]
        _era5 = []
        for root, _, files in os.walk(era5):  
            for filename in files:
                _era5.append(
                    xr.load_dataset(os.path.join(root, filename))
                )
        _era5 = sorted(
            _era5, key=lambda _era5: _era5.time.data[0]
        )
        u10 = self._handle_u10(_era5)
        gc.collect()
        v10 = self._handle_v10(_era5)
        gc.collect()
        t2m = self._handle_t2m(_era5)
        gc.collect()
        msl = self._handle_msl(_era5)
        gc.collect()
        sst = self._handle_sst(_era5)
        gc.collect()
        q2m = self._handle_q2m(_era5)
        gc.collect()
        self.u10_seq = [None] * len(self.edh_seq)
        self.v10_seq = [None] * len(self.edh_seq)
        self.t2m_seq = [None] * len(self.edh_seq)
        self.msl_seq = [None] * len(self.edh_seq)
        self.sst_seq = [None] * len(self.edh_seq)
        self.q2m_seq = [None] * len(self.edh_seq)
        for i in range(len(self.edh_seq)):
            self.u10_seq[i] = u10[i : i + seq_len]
            self.v10_seq[i] = v10[i : i + seq_len]
            self.t2m_seq[i] = t2m[i : i + seq_len]
            self.msl_seq[i] = msl[i : i + seq_len]
            self.sst_seq[i] = sst[i : i + seq_len]
            self.q2m_seq[i] = q2m[i : i + seq_len]
        
        if flag == "fit":
            with open('.cache/dist.json', 'w') as f:  
                json.dump(self.dist, f)

    def __len__(self) -> int:
        return len(self.time_seq)

    def __getitem__(self, index: int):
        return (
            np.array(
                [
                    self.u10_seq[index], 
                    self.v10_seq[index], 
                    self.t2m_seq[index],
                    self.msl_seq[index], 
                    self.sst_seq[index], 
                    self.q2m_seq[index]
                ]
            ),
            np.expand_dims(self.edh_seq[index], axis=0)
        )

class DistriValDataset(TD):

    def __init__(self, train_path, test_path, seq_len):
        super(DistriValDataset, self).__init__()
        self.flag = "test"
        # load mean and std
        with open('.cache/dist.json', 'r') as f:  
            self.dist = json.load(f)

        # load era5=[time, lat, lon] train set
        era5_trian_seq = self.load_era5_seq(train_path, seq_len, is_train=1)

        # load era5=[time, lat, lon] test set
        era5_test_seq = self.load_era5_seq(test_path, seq_len, is_train=0)

        self.era5_seq = era5_trian_seq + era5_test_seq
        

    def load_era5_seq(self, era5_path, seq_len, is_train=1):
        # load era5=[time, lat, lon]
        _era5 = []
        for root, _, files in os.walk(era5_path):  
            for filename in files:
                _era5.append(
                    xr.load_dataset(os.path.join(root, filename))
                )
        _era5 = sorted(
            _era5, key=lambda _era5: _era5.time.data[0]
        )
        u10 = self._handle_u10(_era5)
        gc.collect()
        v10 = self._handle_v10(_era5)
        gc.collect()
        t2m = self._handle_t2m(_era5)
        gc.collect()
        msl = self._handle_msl(_era5)
        gc.collect()
        sst = self._handle_sst(_era5)
        gc.collect()
        q2m = self._handle_q2m(_era5)
        gc.collect()
        era5_seq = [None] * (u10.shape[0] - seq_len + 1)
        for i in range((u10.shape[0] - seq_len + 1)):
            era5_seq[i] = (
                np.array([
                    u10[i : i + seq_len],
                    v10[i : i + seq_len],
                    t2m[i : i + seq_len],
                    msl[i : i + seq_len],
                    sst[i : i + seq_len],
                    q2m[i : i + seq_len],
                ]), 
                is_train
            )
        return era5_seq

    def __len__(self) -> int:
        return len(self.era5_seq)

    def __getitem__(self, index: int):
        return self.era5_seq[index]

class SinDateset(Dataset):

    def __init__(self, start, step, len, seq_len) -> None:
        super(SinDateset, self).__init__()
        self.start = start
        self.step = step
        self.len = len
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int):
        return np.expand_dims(np.sin(
                np.linspace(
                    index * self.step + self.start,
                    (index + self.seq_len - 1) * self.step + self.start,
                    self.seq_len
                )
            ),
            axis=1
        ).astype(np.float32)
