import torch
from omegaconf import OmegaConf
from pytorch_lightning import (LightningModule, Trainer, callbacks, loggers,
                               seed_everything)

from .dm import PLDM


class ClassicConf():
    def __init__(
            self, dm_class:PLDM, model_class:LightningModule, 
            config_dir:str, log_dir:str, log_name:str, ckp_dir:str
        ):
        self.config = OmegaConf.load(
            open(config_dir, "r")
        )
        seed_everything(
            seed=self.config.seed, workers=True
        )
        torch.set_float32_matmul_precision(
            self.config.optim.float32_matmul_precision
        )
        self.dm = dm_class(
            self.config.data, 
            self.config.optim.batch_size,
            self.config.seed
        )
        self.model_class = model_class
        self.trainer = Trainer(
            max_epochs=self.config.optim.max_epochs,
            accelerator=self.config.optim.accelerator,
            accumulate_grad_batches=self.config.optim.accumulate,
            logger=[
                loggers.TensorBoardLogger(
                    log_dir,
                    name=log_name,
                ),
                loggers.CSVLogger(
                    log_dir,
                    name=log_name
                )
            ],
            precision=self.config.optim.precision,
            enable_checkpointing=True,
            callbacks=[
                callbacks.ModelCheckpoint(
                    dirpath=ckp_dir,
                    monitor=self.config.optim.monitor,
                    save_last=True,
                    save_top_k=3
                ),
                callbacks.EarlyStopping(
                    monitor=self.config.optim.monitor,
                    patience=self.config.optim.patience,
                )
            ],
            # deterministic="warn"
        )


    def train(self, ckpt=None):
        self.dm.setup("fit")
        total_num_steps = (
            self.dm.train_sample_num * self.config.optim.max_epochs
        ) / (
            self.config.optim.batch_size * self.config.optim.accumulate
        )
        self.config.optim.total_num_steps = total_num_steps
        model = self.model_class(
            self.config.model, self.config.optim
        )
        self.trainer.fit(
            model, self.dm, ckpt_path=ckpt
        )

    def test(self, ckpt):
        self.dm.setup("test")
        model = self.model_class.load_from_checkpoint(ckpt)
        self.trainer.test(
            model, self.dm
        )

def load_model_data(model_class, data_class, conf_path, ckp_path):
    config = OmegaConf.load(open(conf_path, "r"))
    torch.set_float32_matmul_precision(
        config.optim.float32_matmul_precision
    )
    seed_everything(
        seed=config.seed, workers=True
    )
    
    # data
    dm = data_class(
        config.data,
        config.optim.batch_size,
        config.seed
    )
    dm.setup("test")
    test_dl = dm.test_dataloader()

    # model
    net = model_class.load_from_checkpoint(ckp_path).eval()

    return (net, dm, test_dl, config)