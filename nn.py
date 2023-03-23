import os
from typing import Union
import pytorch_lightning as pl
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from einops import rearrange
from pytorch_fid.fid_score import calculate_frechet_distance
from torch import nn as nn
from torch.nn import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import CLIPTextModel


@torch.no_grad()
def calculate_activation_statistics(samples, inception_v3):
    features = inception_v3(samples)[0]
    features = rearrange(features, "... 1 1 -> ...")

    mu = torch.mean(features, dim=0).cpu()
    sigma = torch.cov(features).cpu()
    return mu, sigma


def fid_score(real_samples, fake_samples, inception_v3):
    min_batch = min(real_samples.shape[0], fake_samples.shape[0])
    real_samples, fake_samples = map(
        lambda t: t[:min_batch], (real_samples, fake_samples)
    )

    m1, s1 = calculate_activation_statistics(real_samples, inception_v3)
    m2, s2 = calculate_activation_statistics(fake_samples, inception_v3)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


class Diffusion(pl.LightningModule):
    def __init__(
        self,
        diffusion_model: nn.Module,
        learning_rate: float = 1e-3,
        count_fid: bool = True,
    ) -> None:
        super().__init__()
        self.model = diffusion_model
        self.lr = learning_rate
        self.fid = FrechetInceptionDistance(normalize=True)
        self.count_fid = count_fid

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr,
            # weight_decay=5e-4,
        )

    def training_step(self, batch, batch_idx: int):
        x, _ = batch
        loss = self.model(x)
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.count_fid is not None:
            with torch.no_grad():
                gen = self.model.sample(batch_size=32)

            self.fid.update(gen, real=False)
            fid_value = self.fid.compute()
            self.fid.reset()
            self.log(
                "test/fid",
                fid_value,
                on_epoch=True,
                on_step=False,
                prog_bar=True,
            )

    def validation_step(self, batch, batch_idx: int):
        x, _ = batch
        loss = self.model(x)
        self.fid.update(x, real=True)

        self.log("test/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss


class LightingStableDiffusion(pl.LightningModule):
    def __init__(self, model_name: str, learning_rate: float = 1e-5) -> None:
        super().__init__()
        self.model_name = model_name
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        )
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")

        self.vae = self.vae.requires_grad_(False)
        self.text_encoder = self.text_encoder.requires_grad_(False)
        self.lr = learning_rate

    def configure_optimizers(self):
        return torch.optim.SGD(self.unet.parameters(), lr=1e-05)

    def training_step(self, batch, batch_idx: int):
        loss = self.training_forward(batch)
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        return loss

    def training_forward(self, batch):
        with torch.no_grad():
            latents = self.calc_latents(batch)
            encoder_hidden_states = self.calc_hidden_state(batch)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.num_train_timesteps,
            (latents.shape[0],),
            device=self.device,
        ).long()
        timesteps, model_pred = self.sample(
            latents, encoder_hidden_states, noise, timesteps
        )
        loss = self.calc_loss(latents, noise, timesteps, model_pred)
        return loss

    def calc_loss(self, latents, noise, timesteps, model_pred):
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss

    def sample(self, latents, encoder_hidden_states, noise, timesteps):
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return timesteps, model_pred

    def calc_hidden_state(self, batch):
        encoder_hidden_states = self.text_encoder(
            batch["text"].to(self.text_encoder.device)
        )[0]

        return encoder_hidden_states

    def calc_latents(self, batch):
        latents = self.vae.encode(batch["image"]).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    def validation_step(self, batch, batch_idx: int):
        loss = self.training_forward(batch)
        self.log("test/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def save_pipeline(self, checkpoint_dir: Union[str, os.PathLike]):
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_name, text_encoder=self.text_encoder, vae=self.vae, unet=self.unet
        )
        pipeline.save_pretrained(checkpoint_dir)
