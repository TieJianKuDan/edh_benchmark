import sys

sys.path.append("./")

import torch
from omegaconf import OmegaConf
from torch.nn.functional import mse_loss

from core.rnn.lstm.lstm_pl import NStepLSTMPL
from scripts.utils.dm import EDHPLDM


def main():
    config = OmegaConf.load(open("scripts/rnn/lstm/config.yaml", "r"))
    torch.set_float32_matmul_precision(
        config.optim.float32_matmul_precision
    )
    # data
    dm = EDHPLDM(
        config.data,
        config.optim.batch_size,
        config.seed
    )
    dm.setup("test")
    test_dl = dm.test_dataloader()

    # model
    net = NStepLSTMPL.load_from_checkpoint(
        f"ckps/lstm/epoch=110-step=5994.ckpt"
    ).eval()

    edh = next(iter(test_dl)).to(net.device)
    cond = edh[:, 0:config.model.cond_len]
    truth = edh[:, 1:]
    pred = net.run(cond)
    loss = mse_loss(pred, truth)
    print(loss)


if __name__ == "__main__":
    main()