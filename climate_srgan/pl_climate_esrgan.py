import argparse
from collections import OrderedDict
from typing import Any

import pytorch_lightning as pl
import torch
import pytorch_lightning.metrics as pl_metrics

from loss import VGGLoss
from models import Generator, Discriminator


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

        if self.hparams.train_fresh:
            state_dict = torch.load(self.hparams.pretrained_gen_model)["state_dict"]
            gen_state_dict = {}

            for key in state_dict.keys():
                if key.startswith("net_G"):
                    gen_state_dict[key.replace("net_G.", "")] = state_dict[key]

            self.net_G.load_state_dict(gen_state_dict)

        self.net_D = Discriminator()

        # metrics
        # We use vgg34 as our feature extraction method by default.
        self.vgg_criterion = VGGLoss()
        # Loss = 10 * mse_loss + vgg_loss + 0.005 * l1_loss
        self.content_criterion = torch.nn.L1Loss()
        self.adversarial_criterion = torch.nn.BCELoss()

    def forward(self, input):
        return self.net_G(input)

    def training_step(self, batch, batch_idx, optimizer_idx):
        lr, hr, _ = batch
        real_label = torch.full((self.hparams.batch_size, 1), 1, dtype=lr.dtype, device=self.device)
        fake_label = torch.full((self.hparams.batch_size, 1), 0, dtype=lr.dtype, device=self.device)

        # Generating fake high resolution images from real low resolution images.
        sr = self(lr)

        # train discriminator
        if optimizer_idx == 0:
            self.net_D.zero_grad()
            # Train with real high resolution image.
            hr_output = self.net_D(hr)  # Train real image.
            sr_output = self.net_D(sr.detach())  # No train fake image.
            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_hr = self.adversarial_criterion(hr_output - torch.mean(sr_output), real_label)
            loss_sr = self.adversarial_criterion(sr_output - torch.mean(hr_output), fake_label)
            loss_d = (loss_sr + loss_hr) / 2

            self.log("train/loss_D", loss_d)

            return loss_d

        # train generator
        if optimizer_idx == 1:
            # According to the feature map, the root mean square error is regarded as the content loss.
            vgg_loss = self.vgg_criterion(sr, hr)
            # Train with fake high resolution image.
            hr_output = self.net_D(hr.detach())  # No train real fake image.
            sr_output = self.net_D(sr)  # Train fake image.
            # Adversarial loss (relativistic average GAN)
            adversarial_loss = self.adversarial_criterion(sr_output - torch.mean(hr_output), real_label)
            # Pixel level loss between two images.
            l1_loss = self.content_criterion(sr, hr)
            loss_g = 10 * l1_loss + vgg_loss + 0.005 * adversarial_loss

            self.log("train/loss_G", loss_g)

            return loss_g

    def validation_step(self, batch, batch_idx):
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
        optimizerD = torch.optim.Adam(self.net_D.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))
        optimizerG = torch.optim.Adam(self.net_G.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))
        schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=self.hparams.step_size, gamma=0.5)
        schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=self.hparams.step_size, gamma=0.5)

        return [optimizerD, optimizerG], [schedulerD, schedulerG]

    @staticmethod
    def add_model_specific_args(parent):
        parser = argparse.ArgumentParser(parents=[parent], add_help=False, conflict_handler="resolve")
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--step_size", type=int, default=3)

        return parser
