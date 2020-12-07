from typing import *
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class VanillaVAE(nn.Module):
    def __init__(self, input_size: int, hidden_dimensions: List[int] = None):
        super(VanillaVAE, self).__init__()

        if hidden_dimensions is None:
            hidden_dimensions = [512, 256, 128, 64, 32]
        hidden_dimensions = [input_size] + hidden_dimensions

        print(f"Creating auto encoder with dimensions {hidden_dimensions}")

        # input_size -> 512 -> 256 -> 128 -> 64 -> 32
        encoding_steps = []
        for step_input_size, step_output_size in zip(hidden_dimensions[:-1], hidden_dimensions[1:-1]):
            print(f"{step_input_size} -> {step_output_size}")
            encoding_steps += [
                nn.Linear(step_input_size, step_output_size),
                nn.GELU()
            ]
        self.encoder = nn.Sequential(*encoding_steps)
        print(f"fc_mu = {hidden_dimensions[-2]} -> {hidden_dimensions[-1]}")
        self.fc_mu = nn.Linear(hidden_dimensions[-2], hidden_dimensions[-1])
        self.fc_var = nn.Linear(hidden_dimensions[-2], hidden_dimensions[-1])

        # 32 -> 64 -> 128 -> 256 -> 512 -(linear)-> input_size
        hidden_dimensions.reverse()
        decoding_steps = []
        for step_input_size, step_output_size in zip(hidden_dimensions[:-1], hidden_dimensions[1:-1]):
            print(f"{step_input_size} -> {step_output_size}")
            decoding_steps += [
                nn.Linear(step_input_size, step_output_size),
                nn.GELU()
            ]
        decoding_steps += [
            nn.Linear(hidden_dimensions[-2], hidden_dimensions[-1])
        ]
        print(f"{hidden_dimensions[-2]} -> {hidden_dimensions[-1]} (linear)")
        self.decoder = nn.Sequential(*decoding_steps)

    def encode(self, x: Tensor):
        x = self.encoder(x)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var]

    def decode(self, x: Tensor):
        return self.decoder(x)

    def sample(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param log_var: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: Tensor):
        mu, log_var = self.encode(x)
        encoded = self.sample(mu, log_var)
        decoded = self.decode(encoded)
        return [decoded, mu, log_var]

    # TODO: Double check this fella
    def loss_function(self, recons, input, mu, log_var, kld_weight) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """
        reconstruction_loss = F.mse_loss(recons, input)

        # KLD loss = KL divergences
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = reconstruction_loss + kld_weight * kld_loss
        return {'loss': loss, 'reconstruction_loss': reconstruction_loss, 'kld': -kld_loss}
