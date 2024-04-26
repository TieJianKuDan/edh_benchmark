import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from sklearn import metrics
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from ..utils.optim import warmup_lambda


class AlexNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_channels)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AlexNetPL(pl.LightningModule):
    
    def __init__(self, model_config, optim_config) -> None:
        super(AlexNetPL, self).__init__()
        self.model = AlexNet(
            model_config.in_channels,
            model_config.out_channels
        )
        self.loss = nn.CrossEntropyLoss()
        self.optim_config = optim_config
        self.save_hyperparameters()

    def forward(self, batch) -> torch.Tensor:
        return self.model(batch)

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
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = rearrange(x, "b c t h w -> b (c t) h w")
        y_h = self(x)
        l = self.loss(y_h, y)
        y_h = torch.argmax(y_h, dim=1)
        f1 = metrics.f1_score(
            y.detach().cpu().numpy(),
            y_h.detach().cpu().numpy(),
        )
        self.log_dict(
            {
                "train/loss": l,
                "train/f1": f1
            }, 
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True
        )
        return l

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = rearrange(x, "b c t h w -> b (c t) h w")
        y_h = self(x)
        l = self.loss(y_h, y)
        y_h = torch.argmax(y_h, dim=1)
        f1 = metrics.f1_score(
            y.detach().cpu().numpy(),
            y_h.detach().cpu().numpy(),
        )
        self.log_dict(
            {
                "val/loss": l,
                "val/f1": f1
            }, 
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True
        )
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = rearrange(x, "b c t h w -> b (c t) h w")
        y_h = self(x)
        l = self.loss(y_h, y)
        y_h = torch.argmax(y_h, dim=1)
        f1 = metrics.f1_score(
            y.detach().cpu().numpy(),
            y_h.detach().cpu().numpy(),
        )
        self.log_dict(
            {
                "test/loss": l,
                "test/f1": f1
            }, 
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True
        )