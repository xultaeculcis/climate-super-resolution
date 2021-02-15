import argparse

import pytorch_lightning as pl
import torch
import pytorch_lightning.metrics as pl_metrics

from models import Generator


class PreTrainingClimateSRGanModule(pl.LightningModule):
    """
    LightningModule for pre-training the Climate RFB-ESRGAN Generator.
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

        # metrics
        self.train_loss = pl_metrics.MeanAbsoluteError()
        self.val_loss = pl_metrics.MeanAbsoluteError()

    def forward(self, input):
        return self.net_G(input)

    def training_step(self, batch, batch_nb):
        lr, hr, _ = batch

        # Generating fake high resolution images from real low resolution images.
        sr = self(lr)

        # The L1 of the generated fake high-resolution image and real high-resolution image is calculated.
        loss = self.train_loss(sr, hr)
        self.log("train/l1", self.train_loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_nb):
        lr, hr, sr_bicubic = batch

        # Generating fake high resolution images from real low resolution images.
        sr = self(lr)

        # The L1 of the generated fake high-resolution image and real high-resolution image is calculated.
        self.val_loss(sr, hr)
        self.log("val/l1", self.val_loss, on_step=False, on_epoch=True)
        self.log("hp_metric", self.val_loss)

    def on_epoch_end(self) -> None:
        with torch.no_grad():
            lr, hr, sr_bicubic = next(iter(self.trainer.datamodule.val_dataloader()))
            lr = lr.to(self.device)

            self.logger.experiment.add_images('hr_images', hr, self.current_epoch)
            self.logger.experiment.add_images('lr_images', lr, self.current_epoch)
            self.logger.experiment.add_images('sr_bicubic', sr_bicubic, self.current_epoch)

            sr = self(lr)
            self.logger.experiment.add_images('sr_images', sr, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net_G.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, gamma=0.5)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent):
        parser = argparse.ArgumentParser(parents=[parent], add_help=False, conflict_handler="resolve")
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--step_size", type=int, default=3)

        return parser
