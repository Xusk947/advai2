from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from env.pacman_env import PacmanEnv
from game.pygame_renderer import PygameRenderer, SPEED_PRESETS
from rl.agent import AgentConfig, DQNAgent


CHECKPOINT_DIR = Path("checkpoints")
PACMAN_CHECKPOINT = CHECKPOINT_DIR / "pacman_dqn.pt"
GHOST_CHECKPOINT_PATTERN = "ghost_{}_dqn.pt"
ANIMATION_FRAMES_PER_STEP = 6


def _load_or_init_pacman_agent(state_dim: int) -> DQNAgent:
    num_actions = 4
    agent = DQNAgent(state_dim=state_dim, num_actions=num_actions, config=AgentConfig())
    if PACMAN_CHECKPOINT.exists():
        state_dict = torch.load(PACMAN_CHECKPOINT, map_location=agent.device, weights_only=True)
        agent.policy_net.load_state_dict(state_dict)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print(f"Loaded Pacman from {PACMAN_CHECKPOINT}")
    else:
        print(f"No checkpoint at {PACMAN_CHECKPOINT}. Using untrained Pacman.")
    return agent


def _get_ghost_agents(env: PacmanEnv) -> Dict[str, DQNAgent]:
    """Always return ghost agents: load from checkpoint if exists, else untrained nets."""
    c, h, w = env.shape
    ghost_state_dim = c * h * w + 16  # one-hot 4 + scalars 12 (incl. valid_moves 4)
    num_actions = 4
    ghost_agents = {}
    for key in env.ghosts:
        ghost_agents[key] = DQNAgent(
            state_dim=ghost_state_dim, num_actions=num_actions, config=AgentConfig()
        )
        path = CHECKPOINT_DIR / GHOST_CHECKPOINT_PATTERN.format(key)
        if path.exists():
            state_dict = torch.load(path, map_location=ghost_agents[key].device, weights_only=True)
            ghost_agents[key].policy_net.load_state_dict(state_dict)
            ghost_agents[key].target_net.load_state_dict(ghost_agents[key].policy_net.state_dict())
    return ghost_agents


def main() -> None:
    env = PacmanEnv(level_image_path="assets/level1.png", max_steps=None)
    state = env.reset()

    agent = _load_or_init_pacman_agent(state.shape)
    ghost_agents = _get_ghost_agents(env)
    renderer = PygameRenderer(env)

    state_vec = state.reshape(-1)
    speed_index = 1  # Slow по умолчанию (30)
    fps = SPEED_PRESETS[speed_index][0]
    animation_frame = 0
    prev_pacman = (env.pacman.pos.x, env.pacman.pos.y)
    prev_ghosts = {k: (g.pos.x, g.pos.y) for k, g in env.ghosts.items()}

    try:
        while True:
            if animation_frame == 0:
                prev_pacman = (env.pacman.pos.x, env.pacman.pos.y)
                prev_ghosts = {k: (g.pos.x, g.pos.y) for k, g in env.ghosts.items()}
                action = agent.select_action(state_vec, eval_mode=True)
                ghost_actions = {
                    key: ghost_agents[key].select_action(
                        env.get_ghost_state(key), eval_mode=True
                    )
                    for key in env.ghosts
                }
                step_result = env.step(action, ghost_actions)
                state_vec = step_result.next_state
                if step_result.done:
                    state = env.reset()
                    state_vec = state
                animation_frame = 1

            t = animation_frame / ANIMATION_FRAMES_PER_STEP
            pacman_display = (
                prev_pacman[0] + t * (env.pacman.pos.x - prev_pacman[0]),
                prev_pacman[1] + t * (env.pacman.pos.y - prev_pacman[1]),
            )
            ghost_display = {
                k: (
                    prev_ghosts[k][0] + t * (env.ghosts[k].pos.x - prev_ghosts[k][0]),
                    prev_ghosts[k][1] + t * (env.ghosts[k].pos.y - prev_ghosts[k][1]),
                )
                for k in env.ghosts
            }
            result = renderer.render(
                fps=fps,
                pacman_display=pacman_display,
                ghost_display=ghost_display,
            )
            if result.get("restart"):
                agent = _load_or_init_pacman_agent(state.shape)
                ghost_agents = _get_ghost_agents(env)
                state = env.reset()
                state_vec = state.reshape(-1)
                animation_frame = 0
                prev_pacman = (env.pacman.pos.x, env.pacman.pos.y)
                prev_ghosts = {k: (g.pos.x, g.pos.y) for k, g in env.ghosts.items()}
                continue
            if result.get("speed_index") is not None:
                speed_index = result["speed_index"]
                fps = SPEED_PRESETS[speed_index][0]
            animation_frame += 1
            if animation_frame > ANIMATION_FRAMES_PER_STEP:
                animation_frame = 0
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
