"""Shaping награды для Pacman: поощрение приближения к пилюлям по лабиринту."""
from __future__ import annotations

from collections import deque
from typing import Deque, Set, Tuple, TYPE_CHECKING

from .level_loader import LevelDefinition
from .vec2 import Vec2Int
from .reward_config import PACMAN_PILL_DIST_COEFF

if TYPE_CHECKING:
    from .pacman_env import PacmanEnv


def _cell_walkable_for_pacman(level: LevelDefinition, pos: Vec2Int) -> bool:
    if pos.x < 0 or pos.y < 0 or pos.x >= level.width or pos.y >= level.height:
        return False
    return not level.walls[pos.y][pos.x] and not level.pacman_barrier[pos.y][pos.x]


def _maze_distance_to_nearest_pill(
    level: LevelDefinition,
    start: Vec2Int,
    max_dist: int = 80,
) -> int:
    """Путевая дистанция от start до ближайшей клетки с пилюлей или power pill (BFS)."""
    w, h = level.width, level.height
    q: Deque[Tuple[Vec2Int, int]] = deque()
    q.append((start, 0))
    visited: Set[Tuple[int, int]] = {(start.x, start.y)}

    while q:
        pos, dist = q.popleft()
        if dist >= max_dist:
            return max_dist
        if level.pills[pos.y][pos.x] or level.power_pills[pos.y][pos.x]:
            return dist
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = pos.x + dx, pos.y + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            if not _cell_walkable_for_pacman(level, Vec2Int(nx, ny)):
                continue
            if (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            q.append((Vec2Int(nx, ny), dist + 1))
    return max_dist


def get_pacman_pill_shaping(env: "PacmanEnv") -> float:
    """Бонус за то, что Pacman близко к ближайшей пилюле (по лабиринту). Возвращает отрицательное число при большой дистанции."""
    if env.level is None or env.pacman is None:
        return 0.0
    dist = _maze_distance_to_nearest_pill(env.level, env.pacman.pos)
    return -PACMAN_PILL_DIST_COEFF * dist
