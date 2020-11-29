import torch
from torch import nn
import torch.nn.functional as F


class SimpleAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, encoded_size=10):
        super(SimpleAutoEncoder, self).__init__()

        self.encoder_hidden = nn.Linear(input_size, hidden_size)
        self.encoder_output = nn.Linear(hidden_size, encoded_size)
        self.decoder_hidden = nn.Linear(encoded_size, hidden_size)
        self.decoder_output = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        x = F.gelu(self.encoder_hidden(x))
        x = F.gelu(self.encoder_output(x))
        return x

    def decode(self, x):
        x = F.gelu(self.decoder_hidden(x))
        x = F.relu(self.decoder_output(x))
        # x = self.decoder_output(x)  # kinda works for non-normalized input (although normalizing input is best)
        return x

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
