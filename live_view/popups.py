"""Event popups: collect step events and tick lifetime."""

from __future__ import annotations

from typing import Dict, List, Tuple

from env.ghosts import GhostMode
from env.pacman_env import PacmanEnv

from .config import POPUP_COLORS, POPUP_DURATION_FRAMES, RGB


def _format_reward(value: float) -> str:
    if value == int(value):
        return f"+{int(value)}" if value >= 0 else str(int(value))
    return f"+{value:.1f}" if value >= 0 else f"{value:.1f}"


class EventPopup:
    __slots__ = ("tile_x", "tile_y", "text", "color", "frames_left")

    def __init__(
        self,
        tile_x: float,
        tile_y: float,
        text: str,
        color: RGB,
        frames_left: int,
    ) -> None:
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.text = text
        self.color = color
        self.frames_left = frames_left


def collect_step_events(
    env: PacmanEnv,
    step_info: Dict,
    prev_pills: int,
    prev_power: int,
    prev_ghost_hits: int,
    prev_ghosts_eaten: int,
    prev_ghost_modes: Dict[str, GhostMode],
    pacman_died_this_step: bool,
) -> List[EventPopup]:
    new_popups: List[EventPopup] = []
    px = float(env.pacman.pos.x)
    py = float(env.pacman.pos.y)
    pacman_color = POPUP_COLORS["pacman"]

    if step_info["pills_eaten"] > prev_pills:
        new_popups.append(EventPopup(px, py, "+1", pacman_color, POPUP_DURATION_FRAMES))
    if step_info["power_pills_eaten"] > prev_power:
        new_popups.append(EventPopup(px, py, "+5", pacman_color, POPUP_DURATION_FRAMES))
    if step_info["ghost_hits"] > prev_ghost_hits:
        new_popups.append(
            EventPopup(px, py, "-25", pacman_color, POPUP_DURATION_FRAMES)
        )
    if step_info["ghosts_eaten"] > prev_ghosts_eaten:
        new_popups.append(
            EventPopup(px, py, "+10", pacman_color, POPUP_DURATION_FRAMES)
        )

    ghost_rewards = step_info.get("ghost_rewards", {})
    if pacman_died_this_step and ghost_rewards:
        for key, g in env.ghosts.items():
            r = ghost_rewards.get(key, 0.0)
            new_popups.append(
                EventPopup(
                    float(g.pos.x),
                    float(g.pos.y),
                    _format_reward(r),
                    POPUP_COLORS[f"ghost_{key}"],
                    POPUP_DURATION_FRAMES,
                )
            )

    for key, g in env.ghosts.items():
        if (
            g.mode is GhostMode.EATEN
            and prev_ghost_modes.get(key) is not GhostMode.EATEN
        ):
            new_popups.append(
                EventPopup(
                    float(g.pos.x),
                    float(g.pos.y),
                    "-10",
                    POPUP_COLORS[f"ghost_{key}"],
                    POPUP_DURATION_FRAMES,
                )
            )
        if (
            g.mode is GhostMode.NORMAL
            and g.pos == g.home_pos
            and prev_ghost_modes.get(key) is GhostMode.EATEN
        ):
            new_popups.append(
                EventPopup(
                    float(g.pos.x),
                    float(g.pos.y),
                    "+2",
                    POPUP_COLORS[f"ghost_{key}"],
                    POPUP_DURATION_FRAMES,
                )
            )
    return new_popups


def tick_popups(
    popups: List[EventPopup],
) -> Tuple[List[Tuple[float, float, str, RGB]], List[EventPopup]]:
    active: List[Tuple[float, float, str, RGB]] = []
    next_list: List[EventPopup] = []
    for p in popups:
        if p.frames_left > 0:
            active.append((p.tile_x, p.tile_y, p.text, p.color))
        if p.frames_left - 1 > 0:
            next_list.append(
                EventPopup(p.tile_x, p.tile_y, p.text, p.color, p.frames_left - 1)
            )
    return active, next_list
