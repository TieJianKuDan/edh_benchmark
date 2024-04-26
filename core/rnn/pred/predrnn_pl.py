import json
import math

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch.nn import Conv2d, MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from scripts.utils.metrics import MAE, RMSE

from ...utils.optim import warmup_lambda
from . import predrnn, predrnn_v2


class AbsPredRNN(pl.LightningModule):
    '''
    abstract class for PredRNN famliy
    '''
    def __init__(self, model_config, optim_config) -> None:
        super(AbsPredRNN, self).__init__()
        self.config = OmegaConf.merge(model_config, optim_config)
        self.eta = self.config.sampling_start_value
        self.conv = Conv2d(
            in_channels=model_config.img_channel,
            out_channels=model_config.out_channel,
            kernel_size=1
        )
        self.loss = MSELoss()

    def reserve_schedule_sampling_exp(self, itr):
        if itr < self.config.r_sampling_step_1:
            r_eta = 0.5
        elif itr < self.config.r_sampling_step_2:
            r_eta = 1.0 - 0.5 * math.exp(-float(itr - self.config.r_sampling_step_1) / self.config.r_exp_alpha)
        else:
            r_eta = 1.0

        if itr < self.config.r_sampling_step_1:
            eta = 0.5
        elif itr < self.config.r_sampling_step_2:
            eta = 0.5 - (0.5 / (self.config.r_sampling_step_2 - self.config.r_sampling_step_1)) * (itr - self.config.r_sampling_step_1)
        else:
            eta = 0.0

        r_random_flip = np.random.random_sample((
            self.config.batch_size, self.config.input_length - 1))
        r_true_token = (r_random_flip < r_eta)

        random_flip = np.random.random_sample((
            self.config.batch_size, 
            self.config.total_length - self.config.input_length - 1
        ))
        true_token = (random_flip < eta)

        ones = np.ones((
            self.config.img_height // self.config.patch_size,
            self.config.img_width // self.config.patch_size,
            self.config.patch_size ** 2 * self.config.img_channel
        ))
        zeros = np.zeros((
            self.config.img_height // self.config.patch_size,
            self.config.img_width // self.config.patch_size,
            self.config.patch_size ** 2 * self.config.img_channel
        ))

        mask_true = []
        for i in range(self.config.batch_size):
            for j in range(self.config.total_length - 2):
                if j < self.config.input_length - 1:
                    if r_true_token[i, j]:
                        mask_true.append(ones)
                    else:
                        mask_true.append(zeros)
                else:
                    if true_token[i, j - (self.config.input_length - 1)]:
                        mask_true.append(ones)
                    else:
                        mask_true.append(zeros)

        mask_true = np.array(mask_true)
        mask_true = np.reshape(
            mask_true,
            (
                self.config.batch_size,
                self.config.total_length - 2,
                self.config.img_height // self.config.patch_size,
                self.config.img_width // self.config.patch_size,
                self.config.patch_size ** 2 * self.config.img_channel
            )
        )
        return mask_true

    def schedule_sampling(self, eta, itr):
        zeros = np.zeros((
            self.config.batch_size,
            self.config.total_length - self.config.input_length - 1,
            self.config.img_height // self.config.patch_size,
            self.config.img_width // self.config.patch_size,
            self.config.patch_size ** 2 * self.config.img_channel
        ))
        if not self.config.scheduled_sampling:
            return 0.0, zeros

        if itr < self.config.sampling_stop_iter:
            eta -= self.config.sampling_changing_rate
        else:
            eta = 0.0
        random_flip = np.random.random_sample((
            self.config.batch_size, 
            self.config.total_length - self.config.input_length - 1
        ))
        true_token = (random_flip < eta)
        ones = np.ones((
            self.config.img_height // self.config.patch_size,
            self.config.img_width // self.config.patch_size,
            self.config.patch_size ** 2 * self.config.img_channel
        ))
        zeros = np.zeros((
            self.config.img_height // self.config.patch_size,
            self.config.img_width // self.config.patch_size,
            self.config.patch_size ** 2 * self.config.img_channel
        ))
        mask_true = []
        for i in range(self.config.batch_size):
            for j in range(
                self.config.total_length - self.config.input_length - 1
            ):
                if true_token[i, j]:
                    mask_true.append(ones)
                else:
                    mask_true.append(zeros)
        mask_true = np.array(mask_true)
        mask_true = np.reshape(
            mask_true,
            (
                self.config.batch_size,
                self.config.total_length - self.config.input_length - 1,
                self.config.img_height // self.config.patch_size,
                self.config.img_width // self.config.patch_size,
                self.config.patch_size ** 2 * self.config.img_channel
            )
        )
        return eta, mask_true

    def reshape_patch(self, img_tensor, patch_size):
        assert 5 == img_tensor.ndim
        batch_size = np.shape(img_tensor)[0]
        seq_length = np.shape(img_tensor)[1]
        img_height = np.shape(img_tensor)[2]
        img_width = np.shape(img_tensor)[3]
        num_channels = np.shape(img_tensor)[4]
        a = torch.reshape(img_tensor, [batch_size, seq_length,
                                    img_height//patch_size, patch_size,
                                    img_width//patch_size, patch_size,
                                    num_channels])
        b = torch.permute(a, [0,1,2,4,3,5,6])
        patch_tensor = torch.reshape(b, [batch_size, seq_length,
                                    img_height//patch_size,
                                    img_width//patch_size,
                                    patch_size*patch_size*num_channels])
        return patch_tensor

    def reshape_patch_back(self, patch_tensor, patch_size):
        assert 5 == patch_tensor.ndim
        batch_size = np.shape(patch_tensor)[0]
        seq_length = np.shape(patch_tensor)[1]
        patch_height = np.shape(patch_tensor)[2]
        patch_width = np.shape(patch_tensor)[3]
        channels = np.shape(patch_tensor)[4]
        img_channels = channels // (patch_size*patch_size)
        a = torch.reshape(patch_tensor, [batch_size, seq_length,
                                    patch_height, patch_width,
                                    patch_size, patch_size,
                                    img_channels])
        b = torch.permute(a, [0,1,2,4,3,5,6])
        img_tensor = torch.reshape(b, [batch_size, seq_length,
                                    patch_height * patch_size,
                                    patch_width * patch_size,
                                    img_channels])
        return img_tensor

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

    def configure_optimizers(self):
        lr = self.config.lr
        betas = self.config.betas
        weight_decay = self.config.weight_decay
        total_num_steps = self.config.total_num_steps
        opt = torch.optim.Adam(
            params=list(self.rnn.parameters()) \
                  +list(self.conv.parameters()),
            lr=lr, 
            betas=betas,
            weight_decay=weight_decay
        )

        warmup_iter = int(
            np.round(
                self.config.warmup_percentage * total_num_steps)
        )
        if self.config.lr_scheduler_mode == 'none':
            return opt
        else:
            if self.config.lr_scheduler_mode == 'cosine':
                warmup_scheduler = LambdaLR(
                    opt,
                    lr_lambda=warmup_lambda(
                        warmup_steps=warmup_iter,
                        min_lr_ratio=self.config.warmup_min_lr_ratio
                    )
                )
                cosine_scheduler = CosineAnnealingLR(
                    opt,
                    T_max=(total_num_steps - warmup_iter),
                    eta_min=self.config.min_lr_ratio * self.config.lr
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


class PredRNNPL(AbsPredRNN):
    '''
    Pytorch lightning module for PredRNN
    '''
    def __init__(self, model_config, optim_config) -> None:
        super(PredRNNPL, self).__init__(
            model_config, optim_config
        )
        self.rnn = predrnn.RNN(
            len(model_config.num_hidden),
            model_config.num_hidden,
            self.config
        )
        self.save_hyperparameters()

    def forward(self, frames_tensor, mask_true):
        input_patch = self.reshape_patch(
            frames_tensor, self.config.patch_size
        )
        next_frames = self.rnn(input_patch, mask_true)
        next_frames = self.reshape_patch_back(
            next_frames, self.config.patch_size
        )
        return next_frames

    def run(self, batch):
        # (b, t, h, w, c)
        temp = self.config.reverse_scheduled_sampling
        self.config.reverse_scheduled_sampling = 0
        mask_input = self.config.input_length
        mask_true = np.zeros((
            batch.shape[0],
            self.config.total_length - mask_input - 1,
            self.config.img_height // self.config.patch_size,
            self.config.img_width // self.config.patch_size,
            self.config.patch_size ** 2 * self.config.img_channel
        ))
        mask_true = torch.FloatTensor(
            mask_true).to(self.config.device)
        next_frames = self(batch, mask_true)
        self.config.reverse_scheduled_sampling = temp
        return next_frames

    def training_step(self, batch, batch_idx):
        input_tensor, _ = batch
        input_tensor = rearrange(input_tensor, "b c t h w -> b t h w c")

        if self.config.reverse_scheduled_sampling == 1:
            mask_true = self.reserve_schedule_sampling_exp(
                self.global_step
            )
        else:
            self.eta, mask_true = self.schedule_sampling(
                self.eta, self.global_step
            )
        # (b, t, h, w, c)
        mask_true = torch.FloatTensor(
            mask_true).to(self.config.device) 

        next_frames = self(input_tensor, mask_true)
        l = self.loss(next_frames, input_tensor[:, 1:])

        self.log(
            "train/loss", l, prog_bar=True,
            logger=True, on_step=True, on_epoch=True
        )
        return l

    def validation_step(self, batch, batch_idx):
        input_tensor, _ = batch
        input_tensor = rearrange(input_tensor, "b c t h w -> b t h w c")

        temp = self.config.reverse_scheduled_sampling
        self.config.reverse_scheduled_sampling = 0
        mask_input = self.config.input_length
        mask_true = np.zeros((
            input_tensor.shape[0],
            self.config.total_length - mask_input - 1,
            self.config.img_height // self.config.patch_size,
            self.config.img_width // self.config.patch_size,
            self.config.patch_size ** 2 * self.config.img_channel
        ))
        mask_true = torch.FloatTensor(
            mask_true).to(self.config.device)
        next_frames = self(input_tensor, mask_true)
        self.config.reverse_scheduled_sampling = temp

        l = self.loss(next_frames, input_tensor[:, 1:])
        self.log(
            "val/loss", l, prog_bar=True,
            logger=True, on_step=False, on_epoch=True
        )

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            with open('.cache/dist_static.json', 'r') as f:  
                self.dist = json.load(f)
                
        input_tensor, _ = batch
        input_tensor = rearrange(input_tensor, "b c t h w -> b t h w c")
        
        temp = self.config.reverse_scheduled_sampling
        self.config.reverse_scheduled_sampling = 0
        mask_input = self.config.input_length
        mask_true = np.zeros((
            input_tensor.shape[0],
            self.config.total_length - mask_input - 1,
            self.config.img_height // self.config.patch_size,
            self.config.img_width // self.config.patch_size,
            self.config.patch_size ** 2 * self.config.img_channel
        ))
        mask_true = torch.FloatTensor(
            mask_true).to(self.config.device)
        next_frames = self(input_tensor, mask_true)
        self.config.reverse_scheduled_sampling = temp        

        self.log_era5(
            era5=input_tensor,
            preds=next_frames
        )


class PredRNNV2PL(AbsPredRNN):
    '''
    Pytorch lightning module for PredRNN
    '''
    def __init__(self, model_config, optim_config) -> None:
        super(PredRNNV2PL, self).__init__(
            model_config, optim_config
        )
        self.rnn = predrnn_v2.RNN(
            len(model_config.num_hidden),
            model_config.num_hidden,
            self.config
        )
        self.save_hyperparameters()

    def forward(self, frames_tensor, mask_true):
        input_patch = self.reshape_patch(
            frames_tensor, self.config.patch_size
        )
        next_frames, d_loss = self.rnn(input_patch, mask_true)
        next_frames = self.reshape_patch_back(
            next_frames, self.config.patch_size
        )
        return next_frames, d_loss

    def run(self, batch):
        # (b, t, h, w, c)
        temp = self.config.reverse_scheduled_sampling
        self.config.reverse_scheduled_sampling = 0
        mask_input = self.config.input_length
        mask_true = np.zeros((
            batch.shape[0],
            self.config.total_length - mask_input - 1,
            self.config.img_height // self.config.patch_size,
            self.config.img_width // self.config.patch_size,
            self.config.patch_size ** 2 * self.config.img_channel
        ))
        mask_true = torch.FloatTensor(
            mask_true).to(self.config.device)
        next_frames, _ = self(batch, mask_true)
        self.config.reverse_scheduled_sampling = temp
        return next_frames

    def training_step(self, batch, batch_idx):
        input_tensor, _ = batch
        input_tensor = rearrange(input_tensor, "b c t h w -> b t h w c")

        if self.config.reverse_scheduled_sampling == 1:
            mask_true = self.reserve_schedule_sampling_exp(
                self.global_step
            )
        else:
            self.eta, mask_true = self.schedule_sampling(
                self.eta, self.global_step
            )
        mask_true = torch.FloatTensor(
            mask_true).to(self.config.device) 

        next_frames, d_loss = self(input_tensor, mask_true)
        mse_loss = self.loss(next_frames, input_tensor[:, 1:])
        total_loss = mse_loss + d_loss

        self.log_dict(
            {
                "train/d_loss": d_loss,
                "train/mse_loss": mse_loss
            },
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True
        )
        self.log(
            "train/total_loss", total_loss, prog_bar=True,
            logger=True, on_step=True, on_epoch=True
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        input_tensor, _ = batch
        input_tensor = rearrange(input_tensor, "b c t h w -> b t h w c")

        temp = self.config.reverse_scheduled_sampling
        self.config.reverse_scheduled_sampling = 0
        mask_input = self.config.input_length
        mask_true = np.zeros((
            input_tensor.shape[0],
            self.config.total_length - mask_input - 1,
            self.config.img_height // self.config.patch_size,
            self.config.img_width // self.config.patch_size,
            self.config.patch_size ** 2 * self.config.img_channel
        ))
        mask_true = torch.FloatTensor(
            mask_true).to(self.config.device)
        next_frames, d_loss = self(input_tensor, mask_true)
        self.config.reverse_scheduled_sampling = temp

        mse_loss = self.loss(next_frames, input_tensor[:, 1:])
        total_loss = mse_loss + d_loss

        self.log_dict(
            {
                "val/d_loss": d_loss,
                "val/mse_loss": mse_loss
            },
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )
        self.log(
            "val/total_loss", total_loss, prog_bar=True,
            logger=True, on_step=False, on_epoch=True
        )

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            with open('.cache/dist_static.json', 'r') as f:  
                self.dist = json.load(f)

        input_tensor, _ = batch
        input_tensor = rearrange(input_tensor, "b c t h w -> b t h w c")
        
        temp = self.config.reverse_scheduled_sampling
        self.config.reverse_scheduled_sampling = 0
        mask_input = self.config.input_length
        mask_true = np.zeros((
            input_tensor.shape[0],
            self.config.total_length - mask_input - 1,
            self.config.img_height // self.config.patch_size,
            self.config.img_width // self.config.patch_size,
            self.config.patch_size ** 2 * self.config.img_channel
        ))
        mask_true = torch.FloatTensor(
            mask_true).to(self.config.device)
        next_frames, d_loss = self(input_tensor, mask_true)
        self.config.reverse_scheduled_sampling = temp
        
        self.log_era5(
            era5=input_tensor,
            preds=next_frames
        )
