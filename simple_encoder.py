from typing import List

import torch
from torch import nn
import torch.nn.functional as F

# VAE: https://github.com/AntixK/PyTorch-VAE/blob/8700d245a9735640dda458db4cf40708caf2e77f/models/vanilla_vae.py#L8
class SimpleAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_dimensions: List[int] = None):
        super(SimpleAutoEncoder, self).__init__()

        if hidden_dimensions is None:
            hidden_dimensions = [512, 256, 128, 64, 32]
        hidden_dimensions = [input_size] + hidden_dimensions

        print(f"Creating auto encoder with dimensions {hidden_dimensions}")

        # input_size -> 512 -> 256 -> 128 -> 64 -> 32
        encoding_steps = []
        for step_input_size, step_output_size in zip(hidden_dimensions, hidden_dimensions[1:]):
            encoding_steps += [
                nn.Linear(step_input_size, step_output_size),
                nn.GELU()
            ]
        self.encoder = nn.Sequential(*encoding_steps)

        # 32 -> 64 -> 128 -> 256 -> 512 -(linear)-> input_size
        hidden_dimensions.reverse()
        decoding_steps = []
        for step_input_size, step_output_size in zip(hidden_dimensions[:-1], hidden_dimensions[1:-1]):
            decoding_steps += [
                nn.Linear(step_input_size, step_output_size),
                nn.GELU()
            ]
        decoding_steps += [
            nn.Linear(hidden_dimensions[-2], hidden_dimensions[-1])
        ]
        self.decoder = nn.Sequential(*decoding_steps)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
