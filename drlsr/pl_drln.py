import argparse
from typing import List, Any

import pytorch_lightning as pl


class DRLNLightningModule(pl.LightningModule):
    def __init__(self, kwargs):
        super(DRLNLightningModule, self).__init__()

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs: List[Any]) -> None:
        pass

    def validation_step(self, *args, **kwargs):
        pass

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        pass

    def configure_optimizers(self):
        pass

    def add_model_specific_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        pass