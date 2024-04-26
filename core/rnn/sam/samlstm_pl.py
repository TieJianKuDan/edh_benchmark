import json

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.nn import Conv2d, MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from scripts.utils.metrics import MAE, RMSE

from ...utils.optim import warmup_lambda
from .samlstm import SAMConvLSTM


class SAMConvLSTMPL(pl.LightningModule):

    def __init__(self, model_config, optim_config) -> None:
        super(SAMConvLSTMPL, self).__init__()
        self.model = SAMConvLSTM(model_config)
        self.conv = Conv2d(
            in_channels=model_config.in_channel,
            out_channels=model_config.out_channel,
            kernel_size=1
        )
        self.loss = MSELoss()
        self.optim_config = optim_config
        self.save_hyperparameters()

    def forward(self, conds, teacher_forcing_rate=0):
        '''
        conds: (b, c, t, h, w)
        '''
        conds = rearrange(conds, "b c t h w -> b t c h w")
        preds = self.model(conds, teacher_forcing_rate)
        preds = rearrange(preds, "b t c h w -> b c t h w")
        return preds

    def training_step(self, batch, batch_idx):
        era5, _ = batch
        preds = self(era5, self.optim_config.teacher_forcing_rate)
        l = self.loss(preds, era5[:, :, 1:])

        self.log(
            "train/loss", l, 
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True
        )

        return l

    def validation_step(self, batch, batch_idx):
        era5, _ = batch
        preds = self(era5)
        l = self.loss(preds, era5[:, :, 1:])

        self.log(
            "val/loss", l, 
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True
        )

    def inverse_norm(self, data, name):
        return data * self.dist[name]["std"] + self.dist[name]["mean"]

    def eval_era5(self, preds, truth, name):
        rmse = RMSE(preds, truth)
        mae = MAE(preds, truth)

        return {
            f"{name}/rmse": rmse,
            f"{name}/mae": mae
        }

    def log_era5(self, era5, preds):
        lookup = {
            "u10": 0,
            "v10": 1,
            "t2m": 2,
            "msl": 3,
            "sst": 4,
            "q2m": 5
        }
        for name, index in lookup.items():
            u10_true = self.inverse_norm(
                era5[:, index, -16:][:, None, :],
                name=name
            )
            u10_pred = self.inverse_norm(
                preds[:, index, -16:][:, None, :],
                name=name
            )
            criteria = self.eval_era5(
                preds=rearrange(u10_pred[:, :, -16:], "b c t h w -> (b t) c h w"),
                truth=rearrange(u10_true[:, :, -16:], "b c t h w -> (b t) c h w"),
                name=name
            )
            self.log_dict(
                criteria,
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True
            )

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            with open('.cache/dist_static.json', 'r') as f:  
                self.dist = json.load(f)
        era5, _ = batch
        preds = self(era5)
        
        self.log_era5(
            era5=era5,
            preds=preds
        )

    def configure_optimizers(self):
        lr = self.optim_config.lr
        betas = self.optim_config.betas
        weight_decay = self.optim_config.weight_decay
        total_num_steps = self.optim_config.total_num_steps
        opt = torch.optim.Adam(
            params=list(self.model.parameters()) \
                  +list(self.conv.parameters()),
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
        