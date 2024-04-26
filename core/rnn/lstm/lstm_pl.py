from einops import rearrange
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from ...utils.optim import warmup_lambda
from .nlstm import NStepLSTM

   
class NStepLSTMPL(pl.LightningModule):
    """
    input: (b, t, f)
    """
    def __init__(self, model_config, optim_config) -> None:
        super(NStepLSTMPL, self).__init__()
        self.h, self.w = model_config.h, model_config.w
        col = [None] * self.h
        for i in range(self.h):
            row = [None] * self.w
            for j in range(self.w):
                row[j] = nn.Sequential(
                    NStepLSTM(
                        model_config.input_channel,
                        model_config.hidden_channels,
                        model_config.input_channel,
                        model_config.pred_len,
                        model_config.dropout
                    )
                )
            row_nets = nn.Sequential(
                *row
            )
            col[i] = row_nets
        self.models = nn.Sequential(*col)
        self.loss = nn.MSELoss()
        self.optim_config = optim_config
        self.cond_len = model_config.cond_len
        self.pred_len = model_config.pred_len
        self.input_channel = model_config.input_channel
        self.output_channel = model_config.output_channel
        self.save_hyperparameters()

    def forward(self, cond, h, w):
        cond = rearrange(cond, "b f t -> b t f")
        model = self.models[h][w]
        output = model[0](cond)
        output = rearrange(output, "b t f -> b f t")
        return output
    
    def run(self, conds):
        '''
        shape: (b, c, t, h, w)
        '''
        b = conds.shape[0]
        c = conds.shape[1]
        t = conds.shape[2]
        h = conds.shape[3]
        w = conds.shape[4]
        assert t == self.cond_len
        assert h == self.h
        assert w == self.w
        assert c == self.input_channel
        preds = torch.zeros(
            (b, self.output_channel,
             self.cond_len + self.pred_len - 1, 
             h, w),
            device=self.device
        )
        for i in range(self.h):
            for j in range(self.w):
                cond = conds[:, :, :, i, j]
                pred = self(cond, i, j)
                preds[:, :, :, i, j] = pred
        return preds

    def training_step(self, batch, batch_idx):
        era5, _ = batch
        assert self.cond_len != 1
        assert self.h == era5.shape[3]
        assert self.w == era5.shape[4]
        conds = era5[:, :, 0:self.cond_len]
        truth = era5[:, :, 1:]
        preds = torch.zeros_like(truth)

        for i in range(self.h):
            for j in range(self.w):
                cond = conds[:, :, :, i, j]
                pred = self(cond, i, j)
                preds[:, :, :, i, j] = pred
        l = self.loss(preds, truth)
        self.log(
            "train/loss", l, prog_bar=True,
            logger=True, on_step=True, on_epoch=True
        )
        return l

    def validation_step(self, batch, batch_idx):
        era5, _ = batch
        assert self.cond_len != 1
        assert self.h == era5.shape[3]
        assert self.w == era5.shape[4]
        conds = era5[:, :, 0:self.cond_len]
        truth = era5[:, :, 1:]
        preds = torch.zeros_like(truth)

        for i in range(self.h):
            for j in range(self.w):
                cond = conds[:, :, :, i, j]
                pred = self(cond, i, j)
                preds[:, :, :, i, j] = pred
        l = self.loss(preds, truth)
        self.log(
            "val/loss", l, prog_bar=True,
            logger=True, on_step=False, on_epoch=True
        )

    def test_step(self, batch, batch_idx):
        era5, _ = batch
        assert self.cond_len != 1
        assert self.h == era5.shape[3]
        assert self.w == era5.shape[4]
        conds = era5[:, :, 0:self.cond_len]
        truth = era5[:, :, 1:]
        preds = torch.zeros_like(truth)

        for i in range(self.h):
            for j in range(self.w):
                cond = conds[:, :, :, i, j]
                pred = self(cond, i, j)
                preds[:, :, :, i, j] = pred
        l = self.loss(preds, truth)
        self.log(
            "test/loss", l, prog_bar=True,
            logger=True, on_step=False, on_epoch=True
        )

    def configure_optimizers(self):
        lr = self.optim_config.lr
        betas = self.optim_config.betas
        weight_decay = self.optim_config.weight_decay
        total_num_steps = self.optim_config.total_num_steps
        opt = torch.optim.Adam(
            params=self.models.parameters(),
            lr=lr, 
            betas=betas,
            weight_decay=weight_decay
        )

        warmup_iter = int(
            np.round(
                self.optim_config.warmup_percentage * total_num_steps)
        )
        if self.optim_config.lr_scheduler_mode == 'none':
            return opt
        else:
            if self.optim_config.lr_scheduler_mode == 'cosine':
                warmup_scheduler = LambdaLR(
                    opt,
                    lr_lambda=warmup_lambda(
                        warmup_steps=warmup_iter,
                        min_lr_ratio=self.optim_config.warmup_min_lr_ratio
                    )
                )
                cosine_scheduler = CosineAnnealingLR(
                    opt,
                    T_max=(total_num_steps - warmup_iter),
                    eta_min=self.optim_config.min_lr_ratio * self.optim_config.lr
                )
                lr_scheduler = SequentialLR(
                    opt,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_iter]
                )
                lr_scheduler_config = {
                    'scheduler': lr_scheduler,
                    'interval': 'step',
                    'frequency': 1, 
                }
            else:
                raise NotImplementedError
            return {
                    "optimizer": opt, 
                    "lr_scheduler": lr_scheduler_config
            }
