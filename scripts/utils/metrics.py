from einops import rearrange
import torch
from torch.nn.functional import mse_loss, l1_loss
from sklearn.metrics import mean_squared_log_error, median_absolute_error, r2_score
from torchmetrics.image import StructuralSimilarityIndexMeasure

def RMSE(x_hat:torch.Tensor, x:torch.Tensor):
    return torch.sqrt(mse_loss(x_hat, x, reduction="mean"))

def MAE(x_hat:torch.Tensor, x:torch.Tensor):
    return l1_loss(x_hat, x, reduction="mean")

def MAPE(x_hat:torch.Tensor, x:torch.Tensor):
    return l1_loss(torch.ones_like(x), x_hat / x, reduction="mean")

def CSI(x_hat:torch.Tensor, x:torch.Tensor):
    """
    shape: [b, h, w]
    """
    levels = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70]

    x_hat[x_hat <= levels[0]] = 0
    x[x <= levels[0]] = 0

    for i in range(len(levels) - 1):
        x_hat[(x_hat > levels[i]) & (x_hat <= levels[i + 1])] = levels[i + 1]
        x[(x > levels[i]) & (x <= levels[i + 1])] = levels[i + 1]

    x_hat[x_hat > levels[-1]] = levels[-1] + 5
    x[x > levels[-1]] = levels[-1] + 5

    return torch.sum(x_hat == x) / torch.sum(x == x)

def MSLE(x_hat:torch.Tensor, x:torch.Tensor):
    '''
    shape: (b, c, h, w)
    '''
    x_hat = rearrange(x_hat, "b c h w -> (b h w) c")
    x = rearrange(x, "b c h w -> (b h w) c")
    return mean_squared_log_error(
        x.detach().numpy(), x_hat.detach().numpy())

def MedAE(x_hat:torch.Tensor, x:torch.Tensor):
    x_hat = rearrange(x_hat, "b c h w -> (b h w) c")
    x = rearrange(x, "b c h w -> (b h w) c")
    return median_absolute_error(
        x.detach().numpy(), x_hat.detach().numpy())

def RS(x_hat:torch.Tensor, x:torch.Tensor):
    '''
    better when it is close to 1 ↑
    '''
    x_hat = rearrange(x_hat, "b c h w -> (b h w) c")
    x = rearrange(x, "b c h w -> (b h w) c")
    return r2_score(
        x.detach().numpy(), x_hat.detach().numpy())

def SSIM(x_hat:torch.Tensor, x:torch.Tensor, data_range):
    ssim = StructuralSimilarityIndexMeasure(
        data_range=data_range
    ).to(x_hat.device)
    return ssim(x_hat, x)
