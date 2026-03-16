"""Main loop: reload agents, run episodes, handle input."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from env.ghosts import GhostMode
from env.pacman_env import PacmanEnv
from game.pygame_renderer import PygameRenderer, SPEED_PRESETS
from rl.agent import DQNAgent

from .agents import get_ghost_agents, load_pacman_agent, read_training_episode_num
from .config import (
    ANIMATION_FRAMES_PER_STEP,
    DEFAULT_SPEED_INDEX,
    END_SCREEN_FRAMES,
    EVAL_EPSILON,
    NO_AGENT_POLL_INTERVAL,
    PACMAN_CHECKPOINT,
)
from .popups import EventPopup, collect_step_events, tick_popups


LEVEL_PATHS = ["assets/level1.png", "assets/level2.png"]
MAX_STEPS_PRESETS = [64, 128, 256, 284, 512]
DEFAULT_MAX_STEPS_INDEX = 2  # 256

LOG_DIR = Path("logs")
EVAL_LOG_FILE = LOG_DIR / "eval_runs.jsonl"


@dataclass
class RenderResult:
    restart: bool
    speed_index: int | None
    toggle_popups: bool
    toggle_debug_rewards: bool
    max_steps_down: bool
    max_steps_up: bool
    new_max_steps_index: int | None
    switch_level: int | None


def _load_eval_history(max_points: int = 200) -> List[Tuple[int, float]]:
    """
    Читает последние max_points строк из лога eval-забегов и возвращает (episode_num, reward).
    Если файла ещё нет или формат битый — возвращает пустой список, не ломая live view.
    """
    if not EVAL_LOG_FILE.exists():
        return []
    try:
        lines = EVAL_LOG_FILE.read_text(encoding="utf-8").splitlines()[-max_points:]
    except OSError:
        return []

    history: List[Tuple[int, float]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        ep = obj.get("episode_num")
        r = obj.get("reward")
        if isinstance(ep, int) and isinstance(r, (int, float)):
            history.append((ep, float(r)))
    history.sort(key=lambda x: x[0])
    return history


def main() -> None:
    level_index = 0
    max_steps_index = DEFAULT_MAX_STEPS_INDEX
    max_steps = MAX_STEPS_PRESETS[max_steps_index]

    env = PacmanEnv(level_image_path=LEVEL_PATHS[level_index], max_steps=max_steps)
    state = env.reset()
    state_dim = state.shape[0]

    agent: DQNAgent | None = None
    ghost_agents: Dict[str, DQNAgent] = {}
    last_mtime = 0.0
    renderer = PygameRenderer(env)
    speed_index = DEFAULT_SPEED_INDEX
    fps = SPEED_PRESETS[speed_index][0]
    show_reward_popups = True
    show_debug_rewards = False

    try:
        while True:
            if PACMAN_CHECKPOINT.exists():
                mtime = PACMAN_CHECKPOINT.stat().st_mtime
                if mtime > last_mtime or agent is None:
                    new_agent = load_pacman_agent(state_dim)
                    if new_agent is not None:
                        agent = new_agent
                        last_mtime = mtime
                ghost_agents = get_ghost_agents(env)

            if agent is None:
                time.sleep(NO_AGENT_POLL_INTERVAL)
                continue

            episode_num = read_training_episode_num()
            eval_history = _load_eval_history()
            env.max_steps = MAX_STEPS_PRESETS[max_steps_index]
            state = env.reset()
            state_dim = state.shape[0]
            if agent is not None and agent.policy_net.net[0].in_features != state_dim:
                agent = None
            state_vec = state
            done = False
            animation_frame = 0
            level_switched = False
            prev_pacman = (env.pacman.pos.x, env.pacman.pos.y)
            prev_ghosts: Dict[str, Tuple[int, int]] = {}
            for k, g in env.ghosts.items():
                prev_ghosts[k] = (g.pos.x, g.pos.y)

            ghost_cumulative: Dict[str, float] = {k: 0.0 for k in env.ghosts}
            popups: List[EventPopup] = []
            last_info: Dict | None = None
            prev_ghost_modes: Dict[str, GhostMode] = {}

            while not done:
                if animation_frame == 0:
                    prev_pacman = (env.pacman.pos.x, env.pacman.pos.y)
                    for k, g in env.ghosts.items():
                        prev_ghosts[k] = (g.pos.x, g.pos.y)
                    prev_pills = env.pills_eaten
                    prev_power = env.power_pills_eaten
                    prev_ghost_hits = env.ghost_hits
                    prev_ghosts_eaten = env.ghosts_eaten
                    prev_ghost_modes = {k: g.mode for k, g in env.ghosts.items()}

                    action = agent.select_action(
                        state_vec, eval_mode=True, eval_epsilon=EVAL_EPSILON
                    )
                    ghost_actions = {}
                    for key in env.ghosts:
                        ghost_actions[key] = ghost_agents[key].select_action(
                            env.get_ghost_state(key),
                            eval_mode=True,
                            eval_epsilon=EVAL_EPSILON,
                        )
                    step_result = env.step(action, ghost_actions)
                    pacman_died = step_result.info["ghost_hits"] > prev_ghost_hits
                    for k in env.ghosts:
                        ghost_cumulative[k] += step_result.info["ghost_rewards"][k]

                    if last_info is not None:
                        for p in collect_step_events(
                            env,
                            step_result.info,
                            prev_pills,
                            prev_power,
                            prev_ghost_hits,
                            prev_ghosts_eaten,
                            prev_ghost_modes,
                            pacman_died,
                        ):
                            popups.append(p)
                    last_info = step_result.info

                    state_vec = step_result.next_state
                    done = step_result.done
                    animation_frame = 1

                active_popups, popups = tick_popups(popups)
                t = animation_frame / ANIMATION_FRAMES_PER_STEP
                pacman_display = (
                    prev_pacman[0] + t * (env.pacman.pos.x - prev_pacman[0]),
                    prev_pacman[1] + t * (env.pacman.pos.y - prev_pacman[1]),
                )
                ghost_display = {}
                for k in env.ghosts:
                    ghost_display[k] = (
                        prev_ghosts[k][0]
                        + t * (env.ghosts[k].pos.x - prev_ghosts[k][0]),
                        prev_ghosts[k][1]
                        + t * (env.ghosts[k].pos.y - prev_ghosts[k][1]),
                    )

                raw = renderer.render(
                    fps=fps,
                    episode_num=episode_num,
                    pacman_display=pacman_display,
                    ghost_display=ghost_display,
                    event_popups=active_popups if show_reward_popups else [],
                    show_reward_popups=show_reward_popups,
                    show_debug_rewards=show_debug_rewards,
                    pacman_reward=env.score,
                    ghost_rewards=ghost_cumulative,
                    max_steps=env.max_steps,
                    level_num=level_index + 1,
                    eval_history=eval_history,
                )
                result = RenderResult(
                    restart=raw.get("restart", False),
                    speed_index=raw.get("speed_index"),
                    toggle_popups=raw.get("toggle_popups", False),
                    toggle_debug_rewards=raw.get("toggle_debug_rewards", False),
                    max_steps_down=raw.get("max_steps_down", False),
                    max_steps_up=raw.get("max_steps_up", False),
                    new_max_steps_index=raw.get("new_max_steps_index"),
                    switch_level=raw.get("switch_level"),
                )

                if result.toggle_popups:
                    show_reward_popups = not show_reward_popups
                if result.toggle_debug_rewards:
                    show_debug_rewards = not show_debug_rewards
                if result.max_steps_down and max_steps_index > 0:
                    max_steps_index -= 1
                if result.max_steps_up and max_steps_index < len(MAX_STEPS_PRESETS) - 1:
                    max_steps_index += 1
                if result.new_max_steps_index is not None:
                    max_steps_index = result.new_max_steps_index
                    env.max_steps = MAX_STEPS_PRESETS[max_steps_index]
                    state = env.reset()
                    state_dim = state.shape[0]
                    if agent is not None and agent.policy_net.net[0].in_features != state_dim:
                        agent = None
                    state_vec = state
                    done = False
                    animation_frame = 0
                    prev_pacman = (env.pacman.pos.x, env.pacman.pos.y)
                    for k, g in env.ghosts.items():
                        prev_ghosts[k] = (g.pos.x, g.pos.y)
                    ghost_cumulative = {k: 0.0 for k in env.ghosts}
                    popups.clear()
                    last_info = None
                    continue
                if result.switch_level is not None:
                    new_idx = result.switch_level - 1
                    if 0 <= new_idx < len(LEVEL_PATHS) and new_idx != level_index:
                        level_index = new_idx
                        env = PacmanEnv(
                            level_image_path=LEVEL_PATHS[level_index],
                            max_steps=MAX_STEPS_PRESETS[max_steps_index],
                        )
                        state = env.reset()
                        state_dim = state.shape[0]
                        renderer.set_env(env)
                        ghost_agents = get_ghost_agents(env)
                        if agent is not None and agent.policy_net.net[0].in_features != state_dim:
                            agent = None
                        level_switched = True
                        break
                if result.restart:
                    new_agent = load_pacman_agent(state_dim)
                    if new_agent is not None:
                        agent = new_agent
                    ghost_agents = get_ghost_agents(env)
                    episode_num = read_training_episode_num()
                    state = env.reset()
                    state_dim = state.shape[0]
                    state_vec = state
                    done = False
                    animation_frame = 0
                    prev_pacman = (env.pacman.pos.x, env.pacman.pos.y)
                    for k, g in env.ghosts.items():
                        prev_ghosts[k] = (g.pos.x, g.pos.y)
                    ghost_cumulative = {k: 0.0 for k in env.ghosts}
                    popups.clear()
                    last_info = None
                    continue
                if result.speed_index is not None:
                    speed_index = result.speed_index
                    fps = SPEED_PRESETS[speed_index][0]

                animation_frame += 1
                if animation_frame > ANIMATION_FRAMES_PER_STEP:
                    animation_frame = 0

            if level_switched:
                continue

            steps_done = env.steps_done
            print(
                f"Run ended (agent from training ep. {episode_num}): steps={steps_done}, "
                f"pills={env.pills_eaten}, power_pills={env.power_pills_eaten}, "
                f"ghost_hits={env.ghost_hits}, reward={env.score:.1f}"
            )

            for _ in range(END_SCREEN_FRAMES):
                renderer.render(
                    fps=fps,
                    episode_num=episode_num,
                    episode_ended=True,
                    max_steps=MAX_STEPS_PRESETS[max_steps_index],
                    level_num=level_index + 1,
                    eval_history=_load_eval_history(),
                )

    finally:
        renderer.close()
