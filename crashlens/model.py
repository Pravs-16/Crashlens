"""Dense autoencoder for pointwise telemetry reconstruction."""

import torch
from torch import nn

from . import config


class DenseAutoencoder(nn.Module):
    """38 -> 24 -> 8 -> 24 -> 38 fully connected autoencoder.

    Pointwise (one timestep per sample): the bottleneck forces the model to
    learn the joint distribution of the 38 metrics under normal operation,
    so correlated behaviour (e.g. high CPU is only normal alongside high
    network traffic) is captured without sequence modelling. The output
    layer is linear because scaled inputs may legitimately fall outside
    [0, 1] at test time.
    """

    def __init__(self, n_features=config.N_FEATURES, hidden=24, bottleneck=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, bottleneck), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
