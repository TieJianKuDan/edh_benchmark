import sys

import pandas as pd

sys.path.append("./")

import numpy as np
import torch
from matplotlib import pyplot as plt

from scripts.utils.metrics import CSI, MAE, MAPE, RMSE, SSIM
from scripts.utils.nc import load_edh_from_nc

model_names = {
    "ConvLSTM": "logs/convlstm_3/version_0/metrics.csv",
    "PredRNN": "logs/predrnn_3/version_0/metrics.csv",
    "CrevNet": "logs/crevnet_3/version_0/metrics.csv",
    "SA-ConvLSTM": "logs/samlstm_3/version_0/metrics.csv",
    "PredRNN-v2": "logs/predrnn_v2_3/version_0/metrics.csv",
    "SmaAt-UNet": "logs/smaat_3/version_0/metrics.csv",
    "SimVP": "logs/simvp_3/version_0/metrics.csv",
    "SimVP2": "logs/simvp2_3/version_0/metrics.csv",
    "TAU": "logs/tau_3/version_0/metrics.csv",
    "MMVP": "logs/mmvp_2/version_0/metrics.csv",
}

def calc_all_criteria(model_name):
    print(f"_____________{model_name}_____________")
    edh_pred_path = f".cache/archive/{model_name}/edh_pred"
    edh_true_path = f".cache/archive/{model_name}/edh_true"
    land = torch.load("data/other/land.pt")

    edh_pred = load_edh_from_nc(edh_pred_path)
    edh_pred = edh_pred * ~land[None, :, :]
    edh_true = load_edh_from_nc(edh_true_path)
    edh_true += 1e-6
    edh_pred += 1e-6

    rmse = RMSE(edh_pred, edh_true)
    mae = MAE(edh_pred, edh_true)
    mape = MAPE(edh_pred, edh_true)
    ssim = SSIM(edh_pred, edh_true, data_range=200)
    csi = CSI(edh_pred, edh_true)

    print(f"rmse: {rmse}")
    print(f"mae: {mae}")
    print(f"mape: {mape}")
    print(f"ssim: {ssim}")
    print(f"csi: {csi}")

    maes = [None] * 16
    for i in range(16):
        maes[i] = MAE(edh_pred[i], edh_true[i])

    return maes

def load_criteria(path):
    log = pd.read_csv(path)
    return log.iloc[0].values[2:]

if __name__ == "__main__":
    t = np.arange(1, 17, 1)
    plt.figure(figsize=(10, 5))
    plt.xlabel('Hour', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.ylabel('MAE', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.xlim([0, 17])
    plt.xticks(np.arange(1, 17, 3), fontsize=18, fontname='Times New Roman')
    plt.ylim((0.5, 4.5))
    plt.yticks(np.arange(1, 4.5, 1), fontsize=18, fontname='Times New Roman')
    for name in model_names:
        maes = load_criteria(model_names[name])
        plt.plot(t, maes, '-o')

    plt.legend(model_names, ncol=2)
    plt.grid()
    plt.tight_layout()
    print("pause")
    plt.savefig("imgs/mae_scheme3.jpg", dpi=1000)
    