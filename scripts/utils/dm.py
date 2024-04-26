import gc

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Subset
from .ds import *


class PLDM(pl.LightningDataModule):
    '''
    abstract class
    '''
    def __init__(self):
        super(PLDM, self).__init__()
    
    @property
    def train_sample_num(self):
        return len(self.train_set)
    
    @property
    def val_sample_num(self):
        return len(self.val_set)
    
    @property
    def test_sample_num(self):
        return len(self.test_set)

    def train_dataloader(self):
        dl = None
        if self.num_workers == 1:
            dl = DataLoader(
                self.train_set, 
                batch_size=self.batch_size, 
                shuffle=True,
                drop_last=True
            )
        else:
            dl = DataLoader(
                self.train_set, 
                batch_size=self.batch_size, 
                shuffle=True,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
            )
        return dl

    def val_dataloader(self):
        dl = None
        if self.num_workers == 1:
            dl = DataLoader(
                self.val_set, 
                batch_size=self.batch_size, 
                shuffle=False,
                drop_last=True
            )
        else:
            dl = DataLoader(
                self.val_set, 
                batch_size=self.batch_size, 
                shuffle=False,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers
            )
        return dl

    def test_dataloader(self):
        dl = None
        if self.num_workers == 1:
            dl = DataLoader(
                self.test_set,
                batch_size=self.batch_size, 
                shuffle=False,
                drop_last=True
            )
        else:
            dl = DataLoader(
                self.test_set,
                batch_size=self.batch_size, 
                shuffle=False,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers
            )
        return dl

class EDHPLDM(PLDM):

    def __init__(self, data_config, batch_size, seed):
        super(EDHPLDM, self).__init__()
        self.data_dir = data_config.data_dir
        self.val_ratio = data_config.val_ratio
        self.num_workers = data_config.num_workers
        self.persistent_workers = data_config.persistent_workers
        self.seq_len = data_config.seq_len
        self.batch_size = batch_size
        self.seed = seed
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage:str):
        datasets = None
        if stage == "fit" \
            and (
                self.train_set == None \
                or self.val_set == None
            ):
            print("prepare train_set and val_set")
            datasets = EDHDataset(self.data_dir, self.seq_len)
            self.train_set, self.val_set = random_split(
                datasets, 
                [
                    1 - self.val_ratio, 
                    self.val_ratio,
                ], 
                generator=torch.Generator().manual_seed(self.seed)
            )
        elif stage == "test" \
            and self.test_set == None:
            print("prepare test_set")
            datasets = EDHDataset(self.data_dir, self.seq_len)
            self.test_set = datasets
        else:
            print(stage)
            return
        self.lon = datasets.lon
        self.lat = datasets.lat
        self.time = datasets.time_seq

class ERA5PLDM(PLDM):
    """
    ouput shape: ((b, 6, len, h, w), (b, 1, len, h, w))
    """
    def __init__(self, data_config, batch_size, seed):
        super(ERA5PLDM, self).__init__()
        self.edh_dir = data_config.edh_dir
        self.era5_dir = data_config.era5_dir
        self.val_ratio = data_config.val_ratio
        self.seq_len = data_config.seq_len
        self.num_workers = data_config.num_workers
        self.persistent_workers = data_config.persistent_workers
        self.batch_size = batch_size
        self.seed = seed
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage:str):
        datasets = None
        if stage == "fit" \
            and (
                self.train_set == None \
                or self.val_set == None
            ):
            print("prepare train_set and val_set")
            datasets = ERA5Dataset(
                self.edh_dir, self.era5_dir, 
                self.seq_len, flag=stage
            )
            val_size = int(self.val_ratio * len(datasets))  
            train_size = len(datasets) - val_size 
            self.train_set = Subset(datasets, range(train_size))  
            self.val_set = Subset(datasets, range(train_size, len(datasets)))
        elif stage == "test" \
            and self.test_set == None:
            print("prepare test_set")
            datasets = ERA5Dataset(
                self.edh_dir, self.era5_dir, 
                self.seq_len, flag=stage
            )
            self.test_set = datasets
        else:
            print(stage)
            return
        self.lon = datasets.lon
        self.lat = datasets.lat
        self.time_seq = datasets.time_seq
        gc.collect()

class DistriPLDM(PLDM):
    """
    ouput shape: ((b, 6, len, h, w), class)
    """
    def __init__(self, data_config, batch_size, seed):
        super(DistriPLDM, self).__init__()
        self.train_path = data_config.train_path
        self.test_path = data_config.test_path
        self.val_ratio = data_config.val_ratio
        self.seq_len = data_config.seq_len
        self.num_workers = data_config.num_workers
        self.persistent_workers = data_config.persistent_workers
        self.batch_size = batch_size
        self.seed = seed
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage:str):
        datasets = None
        if stage == "fit" \
            and (
                self.train_set == None \
                or self.val_set == None
            ):
            print("prepare train_set and val_set")
            datasets = DistriValDataset(
                self.train_path,
                self.test_path,
                self.seq_len
            )
            self.train_set, self.val_set = random_split(
                datasets, 
                [
                    1 - self.val_ratio, 
                    self.val_ratio,
                ], 
                generator=torch.Generator().manual_seed(self.seed)
            )
        elif stage == "test" \
            and self.test_set == None:
            print("prepare test_set")
            datasets = DistriValDataset(
                self.train_path,
                self.test_path,
                self.seq_len
            )
            self.test_set = datasets
        else:
            print(stage)
            return
        gc.collect()

class SinPLDM(PLDM):

    def __init__(self, data_config, batch_size, seed) -> None:
        super(SinPLDM, self).__init__()
        self.start = data_config.start
        self.step = data_config.step
        self.len = data_config.len
        self.seq_len = data_config.seq_len
        self.num_workers = data_config.num_workers
        self.persistent_workers = data_config.persistent_workers
        self.batch_size = batch_size
        self.val_ratio = data_config.val_ratio
        self.seed = seed

    def setup(self, stage:str) -> None:
        datasets = SinDateset(
            self.start, self.step,
            self.len, self.seq_len
        )
        if stage == "fit":
            print("prepare train_set and val_set")
            self.train_set, self.val_set = random_split(
                datasets, 
                [
                    1 - self.val_ratio, 
                    self.val_ratio,
                ], 
                generator=torch.Generator().manual_seed(self.seed)
            )
        elif stage == "test":
            print("prepare test_set")
            self.test_set = datasets
        else:
            print(stage)
            return
