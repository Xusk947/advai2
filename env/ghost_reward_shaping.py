"""Per-ghost reward shaping: different 'brains' via shaped rewards."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Deque, Set, Tuple

from .vec2 import Vec2Int
from .reward_config import GHOST_CLUSTER_PENALTY, GLOBAL_CHASE_COEFF

if TYPE_CHECKING:
    from .pacman_env import PacmanEnv


def _cluster_penalty(env: "PacmanEnv", ghost_key: str) -> float:
    """Штраф за то, что призрак толпится с другими на одной клетке."""
    ghost = env.ghosts.get(ghost_key)
    if ghost is None:
        return 0.0
    pos = ghost.pos
    same_tile = sum(1 for k, g in env.ghosts.items() if k != ghost_key and g.pos == pos)
    return same_tile * GHOST_CLUSTER_PENALTY


def _maze_distance_for_ghost(env: "PacmanEnv", start: Vec2Int, target: Vec2Int, max_dist: int = 50) -> int:
    """Путевая дистанция по клеткам с учётом стен (как минимум не хуже A* по качеству для маленькой карты).

    Используем обычный BFS по проходимым для призрака клеткам (level.walls == False).
    Если целевая клетка недостижима (или далеко), возвращаем max_dist.
    """
    level = env.level
    if level is None:
        return max_dist
    if start == target:
        return 0

    w, h = level.width, level.height
    from collections import deque

    q: Deque[Tuple[Vec2Int, int]] = deque()
    q.append((start, 0))
    visited: Set[Tuple[int, int]] = {(start.x, start.y)}

    while q:
        pos, dist = q.popleft()
        if dist >= max_dist:
            return max_dist
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = pos.x + dx, pos.y + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            if level.walls[ny][nx]:
                continue
            if (nx, ny) in visited:
                continue
            next_pos = Vec2Int(nx, ny)
            if next_pos == target:
                return dist + 1
            visited.add((nx, ny))
            q.append((next_pos, dist + 1))

    return max_dist


def ghost_shaping(env: "PacmanEnv", ghost_key: str) -> float:
    """Returns shaping term for the given ghost (add to shared reward)."""
    if env.pacman is None or env.level is None:
        return 0.0
    pacman_pos = env.pacman.pos
    level = env.level
    ghost = env.ghosts.get(ghost_key)
    if ghost is None:
        return 0.0
    ghost_pos = ghost.pos
    manhattan_dist = ghost_pos.manhattan_distance(pacman_pos)
    maze_dist = _maze_distance_for_ghost(env, ghost_pos, pacman_pos)

    total = _cluster_penalty(env, ghost_key)
    # Небольшой общий бонус за приближение к Pacman по реальной (лабиринтной) дистанции
    total += GLOBAL_CHASE_COEFF * maze_dist

    if ghost_key == "blinky":
        total += _blinky_shaping(pacman_pos, ghost_pos, manhattan_dist)
    elif ghost_key == "pinky":
        total += _pinky_shaping(env, ghost_pos, manhattan_dist)
    elif ghost_key == "inky":
        total += _inky_shaping(env, ghost_pos, manhattan_dist)
    elif ghost_key == "clyde":
        total += _clyde_shaping(pacman_pos, ghost_pos, manhattan_dist, level.height)

    return total


def _blinky_shaping(pacman_pos, ghost_pos, dist: int) -> float:
    """Hunter: reward for getting closer to Pacman."""
    return -0.02 * dist


def _pinky_shaping(env: "PacmanEnv", ghost_pos, dist: int) -> float:
    """Ambusher: reward for being ahead of Pacman along his direction."""
    pacman = env.pacman
    dx, dy = pacman.direction.delta
    ahead = Vec2Int(pacman.pos.x + dx * 4, pacman.pos.y + dy * 4)
    dist_ahead = ghost_pos.manhattan_distance(ahead)
    return -0.01 * dist_ahead


def _inky_shaping(env: "PacmanEnv", ghost_pos, dist: int) -> float:
    """Fickle: reward for being on the 'other side' from Blinky."""
    blinky = env.ghosts.get("blinky")
    if blinky is None:
        return -0.02 * dist
    pacman_pos = env.pacman.pos
    vx = pacman_pos.x - blinky.pos.x
    vy = pacman_pos.y - blinky.pos.y
    target = Vec2Int(pacman_pos.x + vx, pacman_pos.y + vy)
    d = ghost_pos.manhattan_distance(target)
    return -0.015 * d


def _clyde_shaping(pacman_pos, ghost_pos, dist: int, level_height: int) -> float:
    """Feigned ignorance: chase when far, retreat to corner when close.

    Если Pacman далеко — Clyde ведёт себя как охотник.
    Если близко — «трусится» и тянется в угол.
    """
    corner = Vec2Int(0, level_height - 1)
    if dist > 8:
        # Далеко от Pacman — гонимся за ним
        return -0.02 * dist
    # Близко к Pacman — уезжаем к углу
    return -0.02 * ghost_pos.manhattan_distance(corner)


def get_all_ghost_shaping(env: "PacmanEnv") -> Dict[str, float]:
    out = {}
    for key in env.ghosts:
        out[key] = ghost_shaping(env, key)
    return out
