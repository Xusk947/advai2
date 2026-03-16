from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from env.ghosts import GhostMode
from env.pacman_env import PacmanEnv
from rl.agent import AgentConfig, DQNAgent


CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PACMAN_CHECKPOINT = CHECKPOINT_DIR / "pacman_dqn.pt"
GHOST_CHECKPOINT_PATTERN = "ghost_{}_dqn.pt"
EPISODE_NUM_FILE = CHECKPOINT_DIR / "episode.txt"

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
EVAL_LOG_FILE = LOG_DIR / "eval_runs.jsonl"


def _read_start_episode() -> int:
    if not EPISODE_NUM_FILE.exists():
        return 0
    try:
        return int(EPISODE_NUM_FILE.read_text().strip())
    except (ValueError, OSError):
        return 0


# Curriculum: 64 -> 128 -> 256 -> 284 -> 512 по мере роста номера эпизода
MAX_STEPS_SCHEDULE = [
    (500, 64),
    (1000, 128),
    (1500, 256),
    (2000, 284),
    (2250, 512),
    (2500, 1024),
]


def _max_steps_for_episode(episode_num: int) -> int:
    for threshold, max_steps in MAX_STEPS_SCHEDULE:
        if episode_num < threshold:
            return max_steps
    return 512


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pacman and ghost DQN agents.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Start from scratch: do not load checkpoints, reset episode counter.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        metavar="N",
        help="Max episodes (default: unlimited, stop with Ctrl+C).",
    )
    args = parser.parse_args()

    env = PacmanEnv(level_image_path="assets/level1.png", max_steps=512)
    state = env.reset()

    state_dim = state.shape[0]
    c, h, w = env.shape
    ghost_state_dim = c * h * w + 16  # one-hot 4 + scalars 12 (incl. valid_moves 4)
    num_actions = 4

    agent = DQNAgent(state_dim=state_dim, num_actions=num_actions, config=AgentConfig())
    ghost_agents = {
        key: DQNAgent(state_dim=ghost_state_dim, num_actions=num_actions, config=AgentConfig())
        for key in env.ghosts
    }

    if args.clean:
        start_episode = 0
        if EPISODE_NUM_FILE.exists():
            EPISODE_NUM_FILE.unlink()
        if PACMAN_CHECKPOINT.exists():
            PACMAN_CHECKPOINT.unlink()
        for key in env.ghosts:
            path = CHECKPOINT_DIR / GHOST_CHECKPOINT_PATTERN.format(key)
            if path.exists():
                path.unlink()
        print("Clean start: training from scratch (all checkpoints removed).")
    else:
        start_episode = _read_start_episode()
        if PACMAN_CHECKPOINT.exists():
            agent.policy_net.load_state_dict(
                torch.load(PACMAN_CHECKPOINT, map_location=agent.device, weights_only=True)
            )
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(f"Loaded Pacman from {PACMAN_CHECKPOINT}")
        for key in env.ghosts:
            path = CHECKPOINT_DIR / GHOST_CHECKPOINT_PATTERN.format(key)
            if path.exists():
                ghost_agents[key].policy_net.load_state_dict(
                    torch.load(path, map_location=ghost_agents[key].device, weights_only=True)
                )
                ghost_agents[key].target_net.load_state_dict(
                    ghost_agents[key].policy_net.state_dict()
                )
        if start_episode > 0:
            print(f"Resuming from episode {start_episode}.")

    print(f"Using device: {agent.device}")
    if args.episodes is None:
        print("Training until Ctrl+C (unlimited episodes).")
    else:
        print(f"Max episodes: {args.episodes}.")

    i = 0
    while args.episodes is None or i < args.episodes:
        episode_num = start_episode + i + 1
        env.max_steps = _max_steps_for_episode(episode_num)
        state = env.reset()
        state_vec = state
        ghost_states = env.get_all_ghost_states()
        done = False
        episode_reward = 0.0
        ghost_episode_rewards = {key: 0.0 for key in env.ghosts}

        # === TRAINING EPISODE ===
        while not done:
            action = agent.select_action(state_vec)
            ghost_actions = {
                key: ghost_agents[key].select_action(ghost_states[key])
                for key in env.ghosts
            }
            ghost_modes_before = {key: g.mode for key, g in env.ghosts.items()}
            step_result = env.step(action, ghost_actions)

            next_state_vec = step_result.next_state
            next_ghost_states = env.get_all_ghost_states()

            agent.store(
                (
                    state_vec,
                    action,
                    step_result.reward,
                    next_state_vec,
                    step_result.done,
                )
            )
            for key in env.ghosts:
                r = step_result.info["ghost_rewards"][key]
                ghost_episode_rewards[key] += r
                if ghost_modes_before[key] is GhostMode.NORMAL:
                    ghost_agents[key].store(
                        (
                            ghost_states[key],
                            ghost_actions[key],
                            r,
                            next_ghost_states[key],
                            step_result.done,
                        )
                    )

            agent.update()
            for key in env.ghosts:
                ghost_agents[key].update()

            state_vec = next_state_vec
            ghost_states = next_ghost_states
            episode_reward += step_result.reward

            if step_result.done:
                break

        # Сохраняем чекпоинты каждые 5 эпизодов (меньше I/O); всегда сохраняем в последнем эпизоде
        is_last_episode = args.episodes is not None and i >= args.episodes - 1
        if episode_num % 5 == 0 or is_last_episode:
            torch.save(agent.policy_net.state_dict(), PACMAN_CHECKPOINT)
            for key in env.ghosts:
                path = CHECKPOINT_DIR / GHOST_CHECKPOINT_PATTERN.format(key)
                torch.save(ghost_agents[key].policy_net.state_dict(), path)
        EPISODE_NUM_FILE.write_text(str(episode_num))

        steps_done = env.steps_done
        print(f"Episode {episode_num}: steps={steps_done}, reward={episode_reward:.2f}")
        if env.pills_eaten > 0:
            print(f"  pills eaten: {env.pills_eaten}")
        if env.power_pills_eaten > 0:
            print(f"  power pills eaten: {env.power_pills_eaten}")
        if env.ghost_hits > 0:
            print(f"  ghost hits: {env.ghost_hits}")
        if env.ghosts_eaten > 0:
            print(f"  ghosts eaten: {env.ghosts_eaten}")
        ghost_reward_str = ", ".join(
            f"{key}={ghost_episode_rewards[key]:.2f}" for key in env.ghosts
        )
        print(f"  ghosts reward: {ghost_reward_str}")

        # === EVALUATION RUN (no learning, greedy policy with small noise) ===
        if episode_num % 10 == 0:
            eval_state = env.reset()
            eval_state_vec = eval_state
            eval_ghost_states = env.get_all_ghost_states()
            eval_done = False
            eval_reward = 0.0
            eval_steps = 0
            eval_ghost_hits = 0
            eval_ghost_rewards = {key: 0.0 for key in env.ghosts}
            eval_ghost_kills = {key: 0 for key in env.ghosts}
            eval_ghosts_eaten = {key: 0 for key in env.ghosts}
            while not eval_done:
                a = agent.select_action(
                    eval_state_vec,
                    eval_mode=True,
                    eval_epsilon=0.1,
                )
                eval_ghost_actions = {
                    key: ghost_agents[key].select_action(
                        eval_ghost_states[key],
                        eval_mode=True,
                        eval_epsilon=0.1,
                    )
                    for key in env.ghosts
                }
                eval_step = env.step(a, eval_ghost_actions)
                eval_state_vec = eval_step.next_state
                eval_ghost_states = env.get_all_ghost_states()
                eval_reward += eval_step.reward
                eval_steps += 1
                eval_ghost_hits = eval_step.info["ghost_hits"]
                for key in env.ghosts:
                    eval_ghost_rewards[key] += eval_step.info["ghost_rewards"][key]
                for key in eval_step.info.get("ghosts_hit_pacman_this_step", []):
                    if key in eval_ghost_kills:
                        eval_ghost_kills[key] += 1
                for key in eval_step.info.get("ghosts_eaten_this_step", []):
                    if key in eval_ghosts_eaten:
                        eval_ghosts_eaten[key] += 1
                eval_done = eval_step.done

            # Логируем данные eval-забега для последующего анализа / построения метрик.
            # Формат: одна JSON-строка на строку (JSONL), чтобы удобно читать и агрегировать.
            eval_log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "episode_num": episode_num,
                "run_type": "eval",
                "steps": eval_steps,
                "env": {
                    "max_steps": int(env.max_steps if env.max_steps is not None else -1),
                    "pills_eaten": int(env.pills_eaten),
                    "power_pills_eaten": int(env.power_pills_eaten),
                    "ghost_hits": int(eval_ghost_hits),
                    "ghosts_eaten": int(env.ghosts_eaten),
                },
                "pacman": {
                    "total_reward": float(eval_reward),
                },
                "ghosts": {
                    key: {
                        "total_reward": float(eval_ghost_rewards.get(key, 0.0)),
                        "kills_pacman": int(eval_ghost_kills.get(key, 0)),
                        "times_eaten": int(eval_ghosts_eaten.get(key, 0)),
                    }
                    for key in env.ghosts
                },
            }
            try:
                with EVAL_LOG_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(eval_log_entry, ensure_ascii=False) + "\n")
            except OSError:
                # Не ломаем обучение, если по какой-то причине не можем записать лог.
                pass

            print(
                f"  [eval] steps={eval_steps}, reward={eval_reward:.2f}, "
                f"pills={env.pills_eaten}, power_pills={env.power_pills_eaten}, "
                f"ghost_hits={eval_ghost_hits}, ghosts_eaten={env.ghosts_eaten}"
            )

        i += 1


if __name__ == "__main__":
    main()
