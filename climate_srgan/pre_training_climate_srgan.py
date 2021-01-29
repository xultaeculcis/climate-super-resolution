import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

from models import Generator


class PreTrainingClimateSRGanModule(pl.LightningModule):
    """
    LightningModule for pre-training the Climate SRGAN.
    """

    def __init__(
            self,
            **kwargs
    ):
        super(PreTrainingClimateSRGanModule, self).__init__()

        # store parameters
        self.save_hyperparameters()

        # networks
        self.net_G = Generator(upscale_factor=self.hparams.scaling_factor)

        # training criterion
        self.loss = nn.L1Loss()

    def forward(self, input):
        return self.net_G(input)

    def training_step(self, batch, batch_nb):
        lr, hr = batch

        # Generating fake high resolution images from real low resolution images.
        sr = self(lr)

        # The MSE of the generated fake high-resolution image and real high-resolution image is calculated.
        l1_loss = self.loss(sr, hr)

        self.log("train_l1_loss", l1_loss, on_step=True)

        return {
            "loss": l1_loss,
            "prog": {
                "train/g_loss": l1_loss,
            }
        }

    def on_epoch_end(self) -> None:
        with torch.no_grad():
            lr, hr = next(iter(self.trainer.datamodule.val_dataloader()))
            lr = lr.to(self.device)

            grid = torchvision.utils.make_grid(hr, nrow=4)
            self.logger.experiment.add_images('hr_images', hr, self.current_epoch)

            grid = torchvision.utils.make_grid(lr, nrow=4)
            self.logger.experiment.add_images('lr_images', lr, self.current_epoch)

            # log SR images
            sr = self(lr)

            grid = torchvision.utils.make_grid(sr, nrow=4)

            self.logger.experiment.add_images('sr_images', sr, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net_G.parameters(), lr=self.hparams.lr, betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, gamma=0.5)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent):
        parser = argparse.ArgumentParser(parents=[parent], add_help=False, conflict_handler="resolve")
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--step_size", type=int, default=int(1e+5))

        return parser
