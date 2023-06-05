import pytorch_lightning as pl
import torch
from einops import rearrange
from pytorch_fid.fid_score import calculate_frechet_distance
from torch import nn as nn
from torch.nn import functional as F
from torchmetrics import Accuracy
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import CLIPTextModel
from typing import Callable


class SegmentationModule(pl.LightningModule):
    def __init__(
        self,
        segmentation_model: nn.Module,
        learning_rate: float = 1e-4,
        loss: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = F.binary_cross_entropy,
    ) -> None:
        super().__init__()
        self.model = segmentation_model
        self.lr = learning_rate
        self.loss = loss
        self.accuracy = Accuracy(task="binary")

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.lr)

    def _common_step(self, batch, task: str) -> torch.Tensor:
        x, y = batch
        preds = self.model(x)
        loss = self.loss(preds, y)
        accuracy = self.accuracy(preds, y)
        self.log(f"{task}/loss", loss, on_epoch=True, on_step=True)
        self.log(f"{task}/acc", accuracy, on_epoch=True, on_step=True)
        return loss

    def training_step(self, batch, batch_idx: int):
        return self._common_step(batch=batch, task="train")

    def validation_step(self, batch, batch_idx: int):
        return self._common_step(batch=batch, task="val")
