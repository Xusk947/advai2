"""Load Pacman and ghost agents from checkpoints."""

from __future__ import annotations

from typing import Dict

import torch

from env.pacman_env import PacmanEnv
from rl.agent import AgentConfig, DQNAgent, load_state_dict_compat

from .config import (
    CHECKPOINT_DIR,
    EPISODE_NUM_FILE,
    GHOST_CHECKPOINT_PATTERN,
    NUM_ACTIONS,
    PACMAN_CHECKPOINT,
)


def load_pacman_agent(state_dim: int) -> DQNAgent | None:
    agent = DQNAgent(state_dim=state_dim, num_actions=NUM_ACTIONS, config=AgentConfig())
    if not PACMAN_CHECKPOINT.exists():
        return agent
    try:
        state_dict = torch.load(
            PACMAN_CHECKPOINT, map_location=agent.device, weights_only=True
        )
        load_state_dict_compat(agent.policy_net, state_dict, agent.device)
        load_state_dict_compat(agent.target_net, state_dict, agent.device)
    except (RuntimeError, OSError, EOFError) as e:
        print(f"Checkpoint read failed (file busy or truncated?): {e}")
        return None
    return agent


def get_ghost_agents(env: PacmanEnv) -> Dict[str, DQNAgent]:
    c, h, w = env.shape
    ghost_state_dim = c * h * w + 16  # one-hot 4 + scalars 12 (incl. valid_moves 4)
    out: Dict[str, DQNAgent] = {}
    for key in env.ghosts:
        out[key] = DQNAgent(
            state_dim=ghost_state_dim,
            num_actions=NUM_ACTIONS,
            config=AgentConfig(),
        )
        path = CHECKPOINT_DIR / GHOST_CHECKPOINT_PATTERN.format(key)
        if path.exists():
            try:
                state_dict = torch.load(
                    path, map_location=out[key].device, weights_only=True
                )
                load_state_dict_compat(
                    out[key].policy_net, state_dict, out[key].device
                )
                out[key].target_net.load_state_dict(
                    out[key].policy_net.state_dict()
                )
            except (RuntimeError, OSError, EOFError):
                pass
    return out


def read_training_episode_num() -> int:
    if not EPISODE_NUM_FILE.exists():
        return 0
    try:
        return int(EPISODE_NUM_FILE.read_text().strip())
    except (ValueError, OSError):
        return 0
