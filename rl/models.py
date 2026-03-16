from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class DQN(nn.Module):
    """Simple DQN model for flattened state"""

    def __init__(self, state_dim: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

