import argparse
from math import sqrt, ceil

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.metrics import SSIM
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import pytorch_lightning as pl

from losses import GANLoss, TVLoss, VGGLoss, PSNR
from models import RRDB, Discriminator, SRResNet


class CSRGAN(pl.LightningModule):
    """
    LightningModule for CSRGAN.
    """

    def __init__(
            self,
            **kwargs
    ):
        super(CSRGAN, self).__init__()

        # store parameters
        self.save_hyperparameters()

        # networks
        self.net_G = SRResNet(self.hparams.scale_factor, self.hparams.ngf, self.hparams.n_blocks)
        self.net_D = Discriminator(self.hparams.ndf)

        # training criterions
        self.criterion_MSE = nn.MSELoss()
        self.criterion_VGG = VGGLoss(net_type=self.hparams.vgg_type, layer=self.hparams.vgg_layer)
        self.criterion_GAN = GANLoss(gan_mode=self.hparams.gan_mode)
        self.criterion_TV = TVLoss()

        # validation metrics
        self.criterion_PSNR = PSNR()
        self.criterion_SSIM = SSIM(kernel_size=(11, 11), reduction="mean")

    def forward(self, input):
        return self.net_G(input)

    def training_step(self, batch, batch_nb, optimizer_idx):
        img_lr = batch["lr"]  # \in [0, 1]
        img_hr = batch["hr"]  # \in [0, 1]

        if optimizer_idx == 0:  # train discriminator
            self.img_sr = self.forward(img_lr)  # \in [0, 1]

            # for real image
            d_out_real = self.net_D(img_hr)
            d_loss_real = self.criterion_GAN(d_out_real, True)
            # for fake image
            d_out_fake = self.net_D(self.img_sr.detach())
            d_loss_fake = self.criterion_GAN(d_out_fake, False)

            # combined discriminator loss
            d_loss = 1 + d_loss_real + d_loss_fake

            return {"loss": d_loss, "prog": {"train/d_loss": d_loss}}

        elif optimizer_idx == 1:  # train generator
            # content loss
            mse_loss = self.criterion_MSE(self.img_sr * 2 - 1, img_hr * 2 - 1)
            vgg_loss = self.criterion_VGG(self.img_sr, img_hr)
            content_loss = (vgg_loss + mse_loss) / 2
            # adversarial loss
            adv_loss = self.criterion_GAN(self.net_D(self.img_sr), True)
            # tv loss
            tv_loss = self.criterion_TV(self.img_sr)

            # combined generator loss
            g_loss = content_loss + 1e-3 * adv_loss + 2e-8 * tv_loss

            if self.global_step % self.trainer.log_every_n_steps == 0:
                nrow = ceil(sqrt(self.hparams.batch_size))
                self.logger.experiment.add_image(
                    tag="train/lr_img",
                    img_tensor=make_grid(img_lr, nrow=nrow, padding=0),
                    global_step=self.global_step
                )
                self.logger.experiment.add_image(
                    tag="train/hr_img",
                    img_tensor=make_grid(img_hr, nrow=nrow, padding=0),
                    global_step=self.global_step
                )
                self.logger.experiment.add_image(
                    tag="train/sr_img",
                    img_tensor=make_grid(self.img_sr, nrow=nrow, padding=0),
                    global_step=self.global_step
                )

            return {
                "loss": g_loss,
                "prog": {
                    "train/g_loss": g_loss,
                    "train/content_loss": content_loss,
                    "train/adv_loss": adv_loss,
                    "train/tv_loss": tv_loss
                }
            }

    def validation_step(self, batch, batch_nb):
        with torch.no_grad():
            img_lr = batch["lr"]
            img_hr = batch["hr"]
            img_sr = self.forward(img_lr)

            img_hr_ = TF.to_grayscale(img_hr)
            img_sr_ = TF.to_grayscale(img_sr)

            psnr = self.criterion_PSNR(img_sr_, img_hr_)
            ssim = 1 - self.criterion_SSIM(img_sr_, img_hr_)  # invert

        return {"psnr": psnr, "ssim": ssim}

    def validation_epoch_end(self, outputs):
        val_psnr_mean = 0
        val_ssim_mean = 0
        for output in outputs:
            val_psnr_mean += output["psnr"]
            val_ssim_mean += output["ssim"]
        val_psnr_mean /= len(outputs)
        val_ssim_mean /= len(outputs)
        return {"val/psnr": val_psnr_mean.item(),
                "val/ssim": val_ssim_mean.item()}

    def configure_optimizers(self):
        optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.hparams.lr)
        optimizer_D = optim.Adam(self.net_D.parameters(), lr=self.hparams.lr)
        scheduler_G = StepLR(optimizer_G, step_size=self.hparams.step_size, gamma=self.hparams.step_lr_gamma)
        scheduler_D = StepLR(optimizer_D, step_size=self.hparams.step_size, gamma=self.hparams.step_lr_gamma)
        return [optimizer_D, optimizer_G], [scheduler_D, scheduler_G]

    @staticmethod
    def add_model_specific_args(parent):
        parser = argparse.ArgumentParser(parents=[parent], add_help=False, conflict_handler="resolve")
        parser.add_argument("--ngf", type=int, default=64)
        parser.add_argument("--ndf", type=int, default=64)
        parser.add_argument("--n_blocks", type=int, default=16)
        parser.add_argument("--vgg_type", type=str, default="vgg19")
        parser.add_argument("--vgg_layer", type=str, default="relu5_4")
        parser.add_argument("--gan_mode", type=str, default="wgangp")
        parser.add_argument("--step_lr_gamma", type=float, default=0.1)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--step_size", type=int, default=int(1e+5))

        return parser
