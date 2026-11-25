"""Ready-made networks for the two modules.

Any ``torch.nn.Module`` with the same call signature can be used instead:

* quantile network ``h(x, z, y) -> tau``, values in (0, 1), shape ``(n, 1)`` or ``(n, y_dim)``;
* outcome network ``g(x, z, tau) -> y_hat``, same shape as ``y``.

``z`` is ``None`` when the data have no covariates.
"""

import math

import torch
import torch.nn as nn


def kaiming_init(module):
    """Kaiming-normal weights and zero biases for every linear and convolution layer."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.modules.conv._ConvNd)):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _select_arm(outputs, x):
    """Pick, for every sample, the output of the head that matches its (integer) treatment."""
    outputs = torch.stack(outputs, dim=1)
    rows = torch.arange(len(x), device=x.device)
    return outputs[rows, x.reshape(-1).long()]


class ResidualBlock(nn.Module):
    """Two linear layers with layer norm and SiLU, added to the input."""

    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, h):
        out = self.act(self.norm1(self.linear1(h)))
        out = self.act(self.norm2(self.linear2(out)))
        return (h + out) / math.sqrt(2)


class MLPQuantileNet(nn.Module):
    """Quantile network ``h(x, z, y) -> tau`` for vector data.

    A linear layer on ``[z, x, y]``, ``num_layers - 1`` residual blocks and a sigmoid output.
    With ``num_arms`` set, ``x`` is an integer treatment and each arm gets its own blocks and
    output layer after the shared first layer.
    """

    def __init__(self, x_dim, y_dim, z_dim=0, hidden_dim=200, num_layers=3, tau_dim=None, num_arms=None):
        super().__init__()
        tau_dim = tau_dim or y_dim
        self.input_layer = nn.Sequential(nn.Linear(z_dim + x_dim + y_dim, hidden_dim), nn.SiLU())
        self.num_arms = num_arms
        if num_arms is None:
            self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_layers - 1)])
            self.final = nn.Linear(hidden_dim, tau_dim)
        else:
            self.heads = nn.ModuleList([
                nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_layers - 1)],
                              nn.Linear(hidden_dim, tau_dim), nn.Sigmoid())
                for _ in range(num_arms)
            ])

    def forward(self, x, z, y):
        h = self.input_layer(torch.cat([x, y] if z is None else [z, x, y], dim=1))
        if self.num_arms is not None:
            return _select_arm([head(h) for head in self.heads], x)
        for block in self.blocks:
            h = block(h)
        return torch.sigmoid(self.final(h))


class MLPOutcomeNet(nn.Module):
    """Outcome network ``g(x, z, tau) -> y_hat`` for vector data.

    ``z``, ``x`` and ``tau`` are each embedded by a linear layer, then joined by a linear layer
    and passed through ``num_layers - 1`` residual blocks. With ``num_arms`` set, ``x`` is an
    integer treatment and each arm gets its own blocks and output layer. Weights use Kaiming
    initialisation.
    """

    def __init__(self, x_dim, y_dim, z_dim=0, tau_dim=None, hidden_dim=200, num_layers=1, num_arms=None):
        super().__init__()
        tau_dim = tau_dim or y_dim
        self.z_input = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.SiLU()) if z_dim > 0 else None
        self.x_input = nn.Sequential(nn.Linear(x_dim, hidden_dim), nn.SiLU())
        self.tau_input = nn.Sequential(nn.Linear(tau_dim, hidden_dim), nn.SiLU())
        num_inputs = 3 if z_dim > 0 else 2
        self.input_layer = nn.Sequential(nn.Linear(num_inputs * hidden_dim, hidden_dim), nn.SiLU())
        self.num_arms = num_arms
        if num_arms is None:
            self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_layers - 1)])
            self.final = nn.Linear(hidden_dim, y_dim)
        else:
            self.heads = nn.ModuleList([
                nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_layers - 1)],
                              nn.Linear(hidden_dim, y_dim))
                for _ in range(num_arms)
            ])
        kaiming_init(self)

    def forward(self, x, z, tau):
        parts = [self.x_input(x), self.tau_input(tau)]
        if self.z_input is not None:
            parts.insert(0, self.z_input(z))
        h = self.input_layer(torch.cat(parts, dim=1))
        if self.num_arms is not None:
            return _select_arm([head(h) for head in self.heads], x)
        for block in self.blocks:
            h = block(h)
        return self.final(h)


class ConvQuantileNet(nn.Module):
    """Quantile network for images: ``z`` and ``y`` are images, ``x`` a vector.

    ``x`` is mapped to an extra image channel; strided convolutions reduce the image to one
    value per quantile level, followed by a sigmoid.
    """

    def __init__(self, image_size=32, z_channels=3, y_channels=3, x_dim=1, hidden_dim=32, tau_dim=1):
        super().__init__()
        self.image_size = image_size
        self.tau_dim = tau_dim
        self.x_map = nn.Linear(x_dim, image_size ** 2)
        layers = [nn.Conv2d(z_channels + 1 + y_channels, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.1)]
        dim = hidden_dim
        for _ in range(int(math.log2(image_size)) - 3):
            layers += [nn.Conv2d(dim, dim * 2, 4, 2, 1), nn.LeakyReLU(0.1)]
            dim *= 2
        layers += [nn.Conv2d(dim, tau_dim, 4, 1, 0), nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, x, z, y):
        s = self.image_size
        x_map = self.x_map(x).view(len(x), 1, s, s)
        return self.model(torch.cat([z, x_map, y], dim=1)).view(len(x), self.tau_dim)


class ConvOutcomeNet(nn.Module):
    """Outcome network for images: an encoder-decoder on ``z`` plus one channel each for ``x`` and ``tau``.

    Weights use Kaiming initialisation.
    """

    def __init__(self, image_size=32, z_channels=3, y_channels=3, x_dim=1, tau_dim=1, hidden_dim=128):
        super().__init__()
        self.image_size = image_size
        self.x_map = nn.Linear(x_dim, image_size ** 2)
        self.tau_map = nn.Linear(tau_dim, image_size ** 2)
        layers = [nn.Conv2d(z_channels + 2, hidden_dim, 4, 2, 1), nn.LeakyReLU(0.1)]
        dim = hidden_dim
        depth = int(math.log2(image_size)) - 3
        for _ in range(depth):
            layers += [nn.Conv2d(dim, dim * 2, 4, 2, 1), nn.LeakyReLU(0.1)]
            dim *= 2
        for _ in range(depth):
            layers += [nn.ConvTranspose2d(dim, dim // 2, 4, 2, 1), nn.LeakyReLU(0.1)]
            dim //= 2
        layers += [nn.ConvTranspose2d(dim, y_channels, 4, 2, 1)]
        self.model = nn.Sequential(*layers)
        kaiming_init(self)

    def forward(self, x, z, tau):
        s = self.image_size
        x_map = self.x_map(x).view(len(x), 1, s, s)
        tau_map = self.tau_map(tau).view(len(x), 1, s, s)
        return self.model(torch.cat([z, x_map, tau_map], dim=1))
