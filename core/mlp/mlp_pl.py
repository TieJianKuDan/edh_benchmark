from einops import rearrange
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from ..utils.optim import warmup_lambda


class Regress(pl.LightningModule):
    
    def __init__(self, model_config, optim_config) -> None:
        super(Regress, self).__init__()
        self.hidden_channels = model_config.hidden_channels
        self.model = nn.Sequential()
        for i in range(len(model_config.hidden_channels)):
            if i == 0:
                self.model.append(
                    nn.Linear(
                        model_config.input_channel,
                        model_config.hidden_channels[i]
                    )
                )
                self.model.append(
                    nn.ReLU()
                )
            else:
                self.model.append(
                    nn.Linear(
                        model_config.hidden_channels[i - 1],
                        model_config.hidden_channels[i]
                    )
                )
                self.model.append(
                    nn.ReLU()
                )
        self.model.append(
            nn.Linear(
                model_config.hidden_channels[-1],
                model_config.output_channel
            )
        )
        self.loss = nn.MSELoss()
        self.optim_config = optim_config
        self.save_hyperparameters()

    def forward(self, batch) -> torch.Tensor:
        return self.model(batch)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = rearrange(x, "b c t h w -> (b t h w) c")
        y = rearrange(y, "b c t h w -> (b t h w) c")
        y_h = self(x)
        l = self.loss(y_h, y)
        self.log_dict(
            {
                "train/loss": l,
            }, 
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True
        )
        return l

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = rearrange(x, "b c t h w -> (b t h w) c")
        y = rearrange(y, "b c t h w -> (b t h w) c")
        y_h = self(x)
        l = self.loss(y_h, y)
        self.log_dict(
            {
                "val/loss": l,
            }, 
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True
        )
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = rearrange(x, "b c t h w -> (b t h w) c")
        y_h = self(x)
        y_h = rearrange(
            y_h, "(b t h w) c -> b c t h w",
            b=y.shape[0],
            t=y.shape[2],
            h=y.shape[3],
            w=y.shape[4]
        )
        land = torch.load("data/land.pt").to(self.device)
        y_h = y_h * (~land[None, None, None, :])
        l = self.loss(y_h, y)
        self.log_dict(
            {
                "test/loss": l,
            }, 
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True
        )

    def configure_optimizers(self):
        lr = self.optim_config.lr
        betas = self.optim_config.betas
        total_num_steps = self.optim_config.total_num_steps
        opt = torch.optim.Adam(
            params=self.model.parameters(),
            lr=lr, 
            betas=betas
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
        