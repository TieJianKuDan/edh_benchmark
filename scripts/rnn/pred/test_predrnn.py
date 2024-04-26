import sys

import torch

sys.path.append("./")

import numpy as np
from einops import rearrange
from torch.nn.functional import mse_loss

from core.rnn.pred.predrnn_pl import PredRNNPL
from scripts.utils.view import edh_gif, edh_subplot


def main():
    net = PredRNNPL.load_from_checkpoint(
        "autodl/scheme2/ckps/predrnn/epoch=4-step=32865.ckpt"
    ).eval()

    era5 = torch.load("data/other/era5.pt")
    edh =  torch.load("data/other/edh.pt")
    lon = torch.load("data/other/lon.pt")
    lat =  torch.load("data/other/lat.pt")

    era5 = era5.to(net.device)
    edh = edh.to(net.device)
    era5 = rearrange(era5, "b c t h w -> b t h w c")
    edh = rearrange(edh, "b c t h w -> b t h w c")
    truth = edh[:, 1:]

    pred = net.run(era5)
    pred = pred * (~torch.load("data/other/land.pt")[None, None, :, :, None].to(net.device))
    loss = mse_loss(
        pred[:, -16:], 
        truth[:, -16:]
    )
    print(loss)

    # save picture
    # edh_gif(
    #     "imgs/predrnn_v1.gif",
    #     lon, lat, 
    #     np.concatenate(
    #         (
    #             truth[23, -16:, :, :, 0].detach().cpu()[None, :], 
    #             pred[23, -16:, :, :, 0].detach().cpu()[None, :]
    #         ),
    #         axis=0
    #     ),
    #     fps=2
    # )
    fig = edh_subplot(
        lon, lat,
        truth[23, -5:, :, :, 0].detach().cpu(), 
        1, 
        5
    )
    fig.savefig("imgs/truth.jpg", dpi=1000)
    fig = edh_subplot(
        lon, lat,
        pred[23, -5:, :, :, 0].detach().cpu(),
        1, 
        5
    )
    fig.savefig("imgs/predrnn_v1.jpg", dpi=1000)


if __name__ == "__main__":
    main()