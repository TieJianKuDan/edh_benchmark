import json
import random

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.nn import Conv2d, MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from scripts.utils.metrics import MAE, RMSE

from ...utils.optim import warmup_lambda
from .layers import AutoEncoder, ZigRevPredictor


class CrevNetPL(pl.LightningModule):

    def __init__(self, model_config, optim_config) -> None:
        super(CrevNetPL, self).__init__()
        self.auto_encoder = AutoEncoder(
            nBlocks=[4, 5, 3],
            nStrides=[1, 2, 2],
            nChannels=None,
            init_ds=2,
            dropout_rate=0., 
            affineBN=True, 
            in_shape=[
                model_config.in_channel,
                model_config.width, 
                model_config.width
            ],
            mult=2
        )
        self.rnn = ZigRevPredictor(
            input_size=model_config.hidden_channel,
            hidden_size=model_config.hidden_channel, 
            output_size=model_config.hidden_channel, 
            n_layers=model_config.n_layers,
            temp=model_config.depth
        )
        self.conv = Conv2d(
            model_config.in_channel,
            model_config.out_channel,
            1
        )
        self.pred_len = model_config.pred_len
        self.cond_len = model_config.cond_len
        self.depth = model_config.depth
        self.in_channel = model_config.in_channel
        self.hidden_channel = model_config.hidden_channel
        self.loss = MSELoss()
        self.optim_config = optim_config
        self.save_hyperparameters()

    def expand_time_depth(self, batch, depth):
        '''
        batch shape: (b, c, t, h, w)
        '''
        ret = []
        for j in range(batch.shape[2] - depth + 1):
            ele = [None] * depth
            for i in range(depth):
                ele[i] = batch[:, :, j + i]
            ret.append(
                torch.stack(
                    ele, 
                    2
                )
            )
        return torch.stack(ret, dim=0)

    def forward(self, conds, teacher_forcing_rate=0):
        '''
        conds: (t, b, c, d, h, w)
        '''
        batch_size = conds[0].shape[0]
        depth = conds[0].shape[2]
        width = conds[0].shape[4]
        self.rnn.init_hidden(
            batch_size=batch_size,
            device=self.device
        )
        memo = torch.zeros(
            batch_size,
            self.hidden_channel,
            depth,
            width // 8, 
            width // 8
        ).to(self.device)

        preds = []
        for t in range(1, self.cond_len + self.pred_len):
            if t <= self.cond_len \
                or random.random() < teacher_forcing_rate:
                cond = conds[t - 1]
            else:
                cond = pred
            h = self.auto_encoder(cond, True)
            h_pred, memo = self.rnn((h, memo))
            pred = self.auto_encoder(h_pred, False)
            preds.append(pred)
        return torch.stack(preds, dim=0)

    def training_step(self, batch, batch_idx):
        era5, _ = batch
        conds = self.expand_time_depth(era5, depth=self.depth)
        truth = conds[1:]
        preds = self(conds, self.optim_config.teacher_forcing_rate)
        
        l = self.loss(preds, truth)

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
        conds = self.expand_time_depth(era5, depth=self.depth)
        truth = conds[1:]
        preds = self(conds)
        l = self.loss(preds, truth)

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
        conds = self.expand_time_depth(era5, depth=self.depth)
        preds = self(conds)
        # (t, b, c, d, h, w)
        preds = rearrange(preds, "t b c d h w -> (b d) c t h w")
        conds = rearrange(conds, "t b c d h w -> (b d) c t h w")
        
        self.log_era5(
            era5=conds,
            preds=preds
        )

    def run(self, conds):
        conds = self.expand_time_depth(conds, depth=self.depth)
        preds = self(conds)
        preds = rearrange(preds[:, :, :, 0], "t b c h w -> b c t h w")
        return preds


    def configure_optimizers(self):
        lr = self.optim_config.lr
        betas = self.optim_config.betas
        weight_decay = self.optim_config.weight_decay
        total_num_steps = self.optim_config.total_num_steps
        opt = torch.optim.Adam(
            params=list(self.auto_encoder.parameters()) \
                  +list(self.rnn.parameters()) \
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
                # generator
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
        