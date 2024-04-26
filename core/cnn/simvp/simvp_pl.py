import json
from einops import rearrange
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import MSELoss, Conv2d
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from ...utils.optim import warmup_lambda
from .models import SimVP
from scripts.utils.metrics import MAPE, RMSE, MAE, SSIM, CSI


class SimVPPL(pl.LightningModule):

    def __init__(self, model_config, optim_config) -> None:
        super(SimVPPL, self).__init__()
        self.model = SimVP(
            model_config.shape_in,
            model_config.hid_S,
            model_config.hid_T,
            model_config.N_S,
            model_config.N_T,
            groups=model_config.groups
        )
        self.conv = Conv2d(
            in_channels=model_config.in_channel,
            out_channels=model_config.out_channel,
            kernel_size=1
        )
        self.cond_len = model_config.cond_len
        self.pred_len = model_config.pred_len
        self.loss = MSELoss()
        self.optim_config = optim_config
        self.save_hyperparameters()

    def forward(self, conds):
        '''
        conds: (b, c, t, h, w)
        '''
        conds = rearrange(conds, "b c t h w -> b t c h w")
        preds = self.model(conds)
        preds = rearrange(preds, "b t c h w -> b c t h w")
        return preds

    def training_step(self, batch, batch_idx):
        era5, edh = batch
        era5 = torch.cat((era5, edh), dim=1)
        cond = era5[:, :, 0:self.cond_len]
        truth = era5[:, :, -self.pred_len:]
        preds = self(cond)
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
        era5, edh = batch
        era5 = torch.cat((era5, edh), dim=1)
        cond = era5[:, :, 0:self.cond_len]
        truth = era5[:, :, -self.pred_len:]
        preds = self(cond)
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

    def eval_edh(self, preds, truth, name):
        rmse = RMSE(preds, truth)
        mae = MAE(preds, truth)
        mape = MAPE(preds, truth)
        ssim = SSIM(preds, truth, data_range=200)
        csi = CSI(preds, truth)

        return {
            f"{name}/rmse": rmse,
            f"{name}/mae": mae,
            f"{name}/mape": mape,
            f"{name}/ssim": ssim,
            f"{name}/csi": csi
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

    def log_edh(self, edh, preds):
        '''
        edh: (b, c, t, h, w)
        '''
        edh = self.inverse_norm(edh, "edh")
        preds = self.inverse_norm(preds, "edh")
        preds = preds * ~self.land[None, None, None, :, :]
        edh += 1e-6
        preds += 1e-6
        criteria = self.eval_edh(
            rearrange(preds[:, :, -16:], "b c t h w -> (b t) c h w"),
            rearrange(edh[:, :, -16:], "b c t h w -> (b t) c h w"), 
            name="edh"
        )
        self.log_dict(
            criteria,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )

    def log_edh_everytime(self, edh, preds):
        '''
        edh: (b, c, t, h, w)
        '''
        edh = self.inverse_norm(edh, "edh")
        preds = self.inverse_norm(preds, "edh")
        preds = preds * ~self.land[None, None, None, :, :]
        edh += 1e-6
        preds += 1e-6
        edh = rearrange(edh[:, :, -16:], "b c t h w -> t b c h w")
        preds = rearrange(preds[:, :, -16:], "b c t h w -> t b c h w")

        for i in range(16):
            mae = MAE(preds[i], edh[i])
            self.log(
                f"t{i+16}",
                mae,
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True
            )

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.land = torch.load("data/other/land.pt").to(self.device)
            with open('.cache/dist_static.json', 'r') as f:  
                self.dist = json.load(f)
        era5, edh = batch
        era5 = torch.cat((era5, edh), dim=1)
        preds = self(era5[:, :, 0:self.cond_len])

        # self.log_era5(
        #     era5=era5[:, 0:6],
        #     preds=preds[:, 0:6]
        # )
        # self.log_edh(
        #     edh=edh,
        #     preds=preds[:, 6][:, None, :]
        # )
        self.log_edh_everytime(
            edh=edh,
            preds=preds[:, 6][:, None, :]
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
        