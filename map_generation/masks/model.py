from typing import Callable

import pytorch_lightning as pl
import torch
from torch import nn as nn
from torchmetrics import Accuracy


class Unet(nn.Module):
    def __init__(self, channels, out_labels, dropout=0.0):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for in_channel, out_channel in zip([3] + channels, channels):
            self.conv_layers.append(
                torch.nn.Conv2d(in_channel, out_channel, 3, padding=1)
            )
        self.last_conv = torch.nn.Conv2d(channels[-1], channels[-1], 3, padding=1)

        channels_without_last = ([out_labels] + channels)[:-1]
        self.up_conv_layers = nn.ModuleList()
        for out_channel, in_channel in zip(channels_without_last[::-1], channels[::-1]):
            self.up_conv_layers.append(
                torch.nn.ConvTranspose2d(2 * in_channel, out_channel, 2, stride=2)
            )
        self.pool = torch.nn.MaxPool2d(2)
        self.act = torch.nn.ReLU()
        self.bnorm = torch.nn.BatchNorm2d(channels[-1])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        shorted = []
        for conv_layer in self.conv_layers:
            x = self.dropout(x)
            x = self.act(conv_layer(x))
            x = self.pool(x)
            shorted.append(x)

        x = self.bnorm(x)
        x = self.last_conv(x)

        for up_layer, sh in zip(self.up_conv_layers, shorted[::-1]):
            x = self.dropout(x)
            x = up_layer(torch.cat((self.act(x), sh), dim=1))
        return {"out": x}


class SegmentationModel(nn.Module):
    def __init__(self, segmentation_base: nn.Module, out_conv_in=21) -> None:
        super().__init__()
        self.model = segmentation_base
        self.out_layer = (
            nn.Conv2d(out_conv_in, 1, 1) if out_conv_in is not None else nn.Identity()
        )

    def forward(self, x):
        out = self.model(x)["out"]
        return self.out_layer(out)


class SegmentationModule(pl.LightningModule):
    def __init__(
        self,
        segmentation_model: nn.Module,
        learning_rate: float = 1e-4,
        loss: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = nn.BCEWithLogitsLoss(),
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
        self.log(f"loss/{task}", loss, on_epoch=True, on_step=True)
        self.log(f"acc/{task}", accuracy, on_epoch=True, on_step=True)
        return loss

    def training_step(self, batch, batch_idx: int):
        return self._common_step(batch=batch, task="train")

    def validation_step(self, batch, batch_idx: int):
        return self._common_step(batch=batch, task="val")
