import sys

sys.path.append("./")

import numpy as np
import torch
from torch.nn.functional import mse_loss

from core.rnn.crev.crev_pl import CrevNetPL
from scripts.utils.view import edh_gif, edh_subplot


def main():
    net = CrevNetPL.load_from_checkpoint(
        "ckps/crevnet_2/epoch=5-step=157758.ckpt"
    ).eval()

    era5 = torch.load("data/other/era5.pt")[16:]
    edh =  torch.load("data/other/edh.pt")[16:]
    lon = torch.load("data/other/lon.pt")
    lat =  torch.load("data/other/lat.pt")

    # (b c t h w)
    era5 = era5.to(net.device)
    edh = edh.to(net.device)
    truth = edh[:, :, 1:]

    pred = net.run(era5)
    pred = pred * (~torch.load("data/other/land.pt")[None, None, None, :].to(net.device))
    loss = mse_loss(
        pred[:, :, -16:], 
        truth[:, :, -16:]
    )
    print(loss)

    # save picture
    # edh_gif(
    #     "imgs/crevnet.gif",
    #     lon, lat,
    #     np.concatenate(
    #         (
    #             truth[7, 0, -16:].detach().cpu()[None, :], 
    #             pred[7, 0, -16:].detach().cpu()[None, :]
    #         ),
    #         axis=0
    #     ),
    #     fps=2
    # )
    fig = edh_subplot(
        lon, lat,
        pred[7, 0, -5:].detach().cpu(),
        1, 
        5
    )
    fig.savefig("imgs/crevnet.jpg", dpi=1000)

if __name__ == "__main__":
    main()
    print("pause")
