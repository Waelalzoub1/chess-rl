"""Small AlphaZero-style policy + value network, sized for CPU self-play."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import N_PLANES, POLICY_SIZE


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, channels=64, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 64, POLICY_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.blocks(self.stem(x))
        return self.policy_head(h), self.value_head(h).squeeze(-1)


def load_model(ckpt_path=None):
    """Builds the net, loading weights from a checkpoint if it exists.

    Returns (model, iteration)."""
    model = PolicyValueNet()
    iteration = 0
    if ckpt_path is not None:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"])
            iteration = ckpt.get("iter", 0)
        except FileNotFoundError:
            pass
    model.eval()
    return model, iteration
