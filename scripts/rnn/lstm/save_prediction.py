import sys
import os

sys.path.append("./")

from einops import rearrange
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.functional import mse_loss

from core.rnn.lstm.lstm_pl import NStepLSTMPL
from scripts.utils.dm import ERA5PLDM
from scripts.utils.nc import *


def main():
    config = OmegaConf.load(open("scripts/rnn/lstm/config.yaml", "r"))
    torch.set_float32_matmul_precision(
        config.optim.float32_matmul_precision
    )
    # data
    dm = ERA5PLDM(
        config.data,
        config.optim.batch_size,
        config.seed
    )
    dm.setup("test")
    test_dl = dm.test_dataloader()

    # model
    net = NStepLSTMPL.load_from_checkpoint(
        f"autodl/scheme1/ckps/lstm/epoch=15-step=105168.ckpt"
    ).eval()

    # test
    model_name = "lstm"
    batch_size = config.optim.batch_size
    pred_len = config.model.pred_len
    cond_len = config.model.cond_len

    os.makedirs(
        name=f".cache/{model_name}/era5_pred",
        exist_ok=True
    )
    os.makedirs(
        name=f".cache/{model_name}/edh_true",
        exist_ok=True
    )
    times = 1
    for i, (era5, edh) in enumerate(test_dl):
        era5 = era5.to(net.device)
        edh = edh.to(net.device)
        if i > 36 * times:
            break
        if i == 0:
            era5_pred = []
            edh_true = []
            t = []
        elif i % 1 == 0:
            t = np.concatenate(t, axis=0)
            t = t[:, -pred_len:]
            t = rearrange(t, "b t -> (b t)")
            save_nc(
                era5_pred, dm.lon[36][None], dm.lat[36][None], t, 
                f".cache/{model_name}/era5_pred/era5_{model_name}_0_{i // 1}.nc",
                save_era5
            )
            save_nc(
                edh_true, dm.lon[36][None], dm.lat[36][None], t, 
                f".cache/{model_name}/edh_true/edh_{model_name}_1_{i // 1}.nc",
                save_edh
            )
            era5_pred = []
            edh_true = []
            t = []
  
        t.append(np.stack(dm.time_seq[i*batch_size : (i+1)*batch_size], axis=0))
        truth = era5[:, :, -pred_len:]
        cond = era5[:, :, 0:cond_len]
        pred = net.run(cond)
        pred = pred[:, :, -pred_len:]
        loss = mse_loss(
            pred, 
            truth
        )
        print(loss)
        era5_pred.append(pred.detach().cpu())
        edh_true.append(edh[:, :, -pred_len:].cpu())

    print("pause")

if __name__ == "__main__":
    main()
