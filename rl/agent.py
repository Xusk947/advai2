from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch
from torch import nn, optim

from .models import DQN


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


def load_state_dict_compat(module: nn.Module, state_dict: dict, device: torch.device) -> None:
    """Load checkpoint into model, допускает меньший входной размер.

    - Если размер тензора совпадает — просто копируем.
    - Для первого слоя `net.0.weight` с меньшим количеством входов:
      копируем веса в первые N входов, остальные оставляем как есть.
    - Остальные несовпадающие тензоры оставляем из текущей инициализации.
    """
    current = module.state_dict()
    merged: dict[str, torch.Tensor] = {}
    for k, cur in current.items():
        if k not in state_dict:
            merged[k] = cur.clone()
            continue
        ckpt = state_dict[k]
        if ckpt.shape == cur.shape:
            merged[k] = ckpt.to(device)
        elif (
            k == "net.0.weight"
            and ckpt.shape[0] == cur.shape[0]
            and ckpt.shape[1] < cur.shape[1]
        ):
            # Копируем только совпадающую часть входов
            merged[k] = cur.clone()
            merged[k][:, : ckpt.shape[1]] = ckpt.to(device)
        else:
            merged[k] = cur.clone()
    module.load_state_dict(merged, strict=True)


@dataclass
class AgentConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    min_buffer_size: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 10_000
    target_update_freq: int = 1000


class DQNAgent:
    def __init__(self, state_dim: int, num_actions: int, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.num_actions = num_actions
        self.steps_done = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, num_actions).to(self.device)
        self.target_net = DQN(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)
        self.replay_buffer: Deque[Transition] = deque(maxlen=self.config.buffer_size)

    def select_action(
        self,
        state: np.ndarray,
        eval_mode: bool = False,
        eval_epsilon: float | None = None,
    ) -> int:
        """eval_epsilon: если задан, то с этой вероятностью случайное действие (для разнообразия при одних весах)."""
        epsilon = self._current_epsilon()
        self.steps_done += 1
        use_epsilon = eval_epsilon if eval_epsilon is not None else (epsilon if not eval_mode else 0.0)
        if random.random() < use_epsilon:
            return random.randrange(self.num_actions)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.policy_net(state_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

    def store(self, transition: Transition) -> None:
        self.replay_buffer.append(transition)

    def can_update(self) -> bool:
        return len(self.replay_buffer) >= self.config.min_buffer_size

    def update(self) -> float | None:
        if not self.can_update():
            return None

        batch = random.sample(self.replay_buffer, self.config.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1, keepdim=True).values
            target_q = rewards_t + self.config.gamma * (1.0 - dones_t) * next_q_values

        loss = nn.functional.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def _current_epsilon(self) -> float:
        frac = min(1.0, self.steps_done / self.config.epsilon_decay)
        return self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start)

