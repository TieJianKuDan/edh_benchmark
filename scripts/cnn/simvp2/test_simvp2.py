import sys

sys.path.append("./")

import numpy as np
import torch
from torch.nn.functional import mse_loss

from core.cnn.simvp2.simvp2_pl import SimVP2PL
from scripts.utils.view import edh_gif, edh_subplot


def main():
    net = SimVP2PL.load_from_checkpoint(
        "autodl/scheme2/ckps/simvp2/epoch=17-step=118314.ckpt"
    ).eval()

    era5 = torch.load("data/other/era5.pt")
    edh =  torch.load("data/other/edh.pt")
    lon = torch.load("data/other/lon.pt")
    lat =  torch.load("data/other/lat.pt")

    # (b c t h w)
    era5 = era5.to(net.device)
    edh = edh.to(net.device)
    cond = era5[:, :, 0:16]
    truth = edh[:, :, -16:]

    pred = net(cond)
    pred = pred * (~torch.load("data/other/land.pt")[None, None, None, :].to(net.device))
    loss = mse_loss(
        pred,
        truth
    )
    print(loss)

    # save picture
    # edh_gif(
    #     "imgs/simvp2.gif",
    #     lon, lat,
    #     np.concatenate(
    #         (
    #             truth[23, 0].detach().cpu()[None, :], 
    #             pred[23, 0].detach().cpu()[None, :]
    #         ),
    #         axis=0
    #     ),
    #     fps=2
    # )
    fig = edh_subplot(
        lon, lat,
        pred[23, 0, -5:].detach().cpu(),
        1, 
        5
    )
    fig.savefig("imgs/simvp2.jpg", dpi=1000)

if __name__ == "__main__":
    main()
    print("pause")