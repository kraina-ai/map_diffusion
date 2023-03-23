import numpy as np
import torch.nn as nn
import torch
from torch import logsumexp
import torch.nn.functional as F
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    linear_beta_schedule,
    cosine_beta_schedule,
    sigmoid_beta_schedule,
)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def index_to_log_onehot(x, num_classes=-1):
    x_onehot = F.one_hot(x, num_classes)
    return torch.log(torch.clamp(x_onehot, 1e-40))


def log_onehot_to_index(log_x):
    return torch.argmax(log_x, 1)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


class MultiGaussianDiffusion(nn.Module):
    def __init__(
        self, num_classes, decode_network, timesteps=1000, beta_schedule="sigmoid"
    ) -> None:
        super().__init__()
        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")
        alphas = np.sqrt(1 - beta_schedule_fn(timesteps))
        self.log_alphas = torch.log(alphas)
        self.log_cumprod_alphas = torch.cumsum(self.log_alphas, dim=0)
        self.log_betas = log_1_min_a(self.log_alphas)
        self.log_cumprod_betas = log_1_min_a(self.log_cumprod_alphas)
        self.num_classes = num_classes
        self.nn = decode_network

    def q_pred_one_timestep(self, log_x_t, t):
        # q(xt|xt−1) = alpha_t * E[xt] + (1 - alpha_t ) 1 / K
        log_alphas_t = extract(self.log_alphas, t, log_x_t.shape)
        log_betas_t = extract(self.log_betas, t, log_x_t.shape)
        return log_add_exp(
            log_x_t + log_alphas_t,
            log_betas_t - np.log(self.num_classes),
        )

    def q_pred(self, log_x_0, t):
        # q(xt|x0)
        return log_add_exp(
            log_x_0 + extract(self.log_cumprod_alphas, t, log_x_0.shape),
            extract(self.log_cumprod_betas, t, log_x_0.shape)
            - np.log(self.num_classes),
        )

    def q_posterior(self, log_x0, log_x_t, t):
        # Kronecker delta peak for q(x0 | x1 , x0 )

        log_probs_x_t_min = self.q_pred(log_x0, torch.where(t - 1 < 0, 0, t - 1))
        unnormed_log_probs = log_probs_x_t_min + self.q_pred_one_timestep(log_x_t, t)
        log_probs_posterior = unnormed_log_probs = logsumexp(unnormed_log_probs, dim=1)
        return log_probs_posterior

    def p_pred(self, log_x_t, t):
        # p(xt−1|xt)
        x_t = log_onehot_to_index(log_x_t)
        log_x_recon = F.log_softmax(self.nn(x_t, t), dim=1)
        log_model_pred = self.q_posterior(log_x_recon, log_x_t, t)
        return log_model_pred

    def categorical_kl(self, log_prob_a, log_prob_b):
        return (log_prob_a.exp() * (log_prob_a - log_prob_b)).sum(dim=1)

    def loss(self, log_x_0, log_x_t, t):
        log_true_prob = self.q_posterior(log_x_0, log_x_t, t)
        log_model_prob = self.p_pred(log_x_t, t)
        kl = self.categorical_kl(log_true_prob, log_model_prob)
        return sum_except_batch(kl)


    def forward(self, x):
        device = x.device
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=device).long()


if __name__ == "__main__":
    print(log_add_exp(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])))
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([1, 2, 3])
    st = torch.stack((a, b))
    tf = logsumexp(st, dim=(0,))
    print(tf)
