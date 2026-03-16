from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from ..entities import Entity
from ..level_loader import LevelDefinition
from ..vec2 import Vec2Int

# Action index to delta: UP, DOWN, LEFT, RIGHT (same as Pacman)
ACTION_DELTAS = (
    Vec2Int(0, -1),
    Vec2Int(0, 1),
    Vec2Int(-1, 0),
    Vec2Int(1, 0),
)


class GhostMode(Enum):
    NORMAL = auto()
    FRIGHTENED = auto()
    EATEN = auto()


@dataclass
class Ghost(Entity):
    key: str
    home_pos: Vec2Int
    mode: GhostMode = GhostMode.NORMAL
    direction: int = 3  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT (для спрайтов глаз)

    def step_with_action(self, level: LevelDefinition, action: int) -> Vec2Int:
        """Move one cell in the given direction (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT). Used by RL."""
        if self.mode is GhostMode.EATEN:
            if self.pos == self.home_pos:
                self.mode = GhostMode.NORMAL
            else:
                old = self.pos
                self.pos = _step_towards_target(level, self.pos, self.home_pos)
                _update_direction_from_move(self, old, self.pos)
            return self.pos
        if self.mode is GhostMode.FRIGHTENED:
            old = self.pos
            self.pos = _step_towards_target(level, self.pos, None)
            _update_direction_from_move(self, old, self.pos)
            return self.pos
        if 0 <= action < len(ACTION_DELTAS):
            delta = ACTION_DELTAS[action]
            next_pos = self.pos + delta
            if _cell_walkable_for_ghost(level, next_pos):
                self.direction = action
                self.pos = next_pos
                return self.pos
            wrap = _get_ghost_wrap_position(level, next_pos)
            if wrap is not None:
                self.direction = action
                self.pos = wrap
                return self.pos
            for i in range(len(ACTION_DELTAS)):
                if i == action:
                    continue
                alt_pos = self.pos + ACTION_DELTAS[i]
                if _cell_walkable_for_ghost(level, alt_pos):
                    self.direction = i
                    self.pos = alt_pos
                    return self.pos
                alt_wrap = _get_ghost_wrap_position(level, alt_pos)
                if alt_wrap is not None:
                    self.direction = i
                    self.pos = alt_wrap
                    return self.pos
        return self.pos


def _update_direction_from_move(ghost: Ghost, old_pos: Vec2Int, new_pos: Vec2Int) -> None:
    """Обновляет ghost.direction по смещению (для EATEN/FRIGHTENED)."""
    dx = new_pos.x - old_pos.x
    dy = new_pos.y - old_pos.y
    if dy < 0:
        ghost.direction = 0  # UP
    elif dy > 0:
        ghost.direction = 1  # DOWN
    elif dx < 0:
        ghost.direction = 2  # LEFT
    elif dx > 0:
        ghost.direction = 3  # RIGHT


def _cell_walkable_for_ghost(level: LevelDefinition, pos: Vec2Int) -> bool:
    if pos.x < 0 or pos.y < 0 or pos.x >= level.width or pos.y >= level.height:
        return False
    return not level.walls[pos.y][pos.x]


def _get_ghost_wrap_position(
    level: LevelDefinition, desired_pos: Vec2Int
) -> Vec2Int | None:
    """If desired_pos is past the map edge, return wrap position on the opposite side (tunnels)."""
    w, h = level.width, level.height

    def try_candidates(candidates: list[tuple[int, int]]) -> Vec2Int | None:
        for wx, wy in candidates:
            if 0 <= wy < h and 0 <= wx < w and _cell_walkable_for_ghost(
                level, Vec2Int(wx, wy)
            ):
                return Vec2Int(wx, wy)
        return None

    def best_in_column(col: int, prefer_y: int) -> Vec2Int | None:
        passable = [
            (col, y)
            for y in range(h)
            if _cell_walkable_for_ghost(level, Vec2Int(col, y))
        ]
        if not passable:
            return None
        passable.sort(key=lambda c: (abs(c[1] - prefer_y), c[1]))
        return Vec2Int(passable[0][0], passable[0][1])

    def best_in_row(row: int, prefer_x: int) -> Vec2Int | None:
        passable = [
            (x, row)
            for x in range(w)
            if _cell_walkable_for_ghost(level, Vec2Int(x, row))
        ]
        if not passable:
            return None
        passable.sort(key=lambda c: (abs(c[0] - prefer_x), c[0]))
        return Vec2Int(passable[0][0], passable[0][1])

    if desired_pos.x < 0:
        y = max(0, min(desired_pos.y, h - 1))
        return try_candidates([(w - 1, y), (w - 1, y - 1), (w - 1, y + 1)]) or best_in_column(w - 1, y)
    if desired_pos.x >= w:
        y = max(0, min(desired_pos.y, h - 1))
        return try_candidates([(0, y), (0, y - 1), (0, y + 1)]) or best_in_column(0, y)
    if desired_pos.y < 0:
        x = max(0, min(desired_pos.x, w - 1))
        return try_candidates([(x, h - 1), (x - 1, h - 1), (x + 1, h - 1)]) or best_in_row(h - 1, x)
    if desired_pos.y >= h:
        x = max(0, min(desired_pos.x, w - 1))
        return try_candidates([(x, 0), (x - 1, 0), (x + 1, 0)]) or best_in_row(0, x)
    return None


def _step_towards_target(
    level: LevelDefinition,
    current_pos: Vec2Int,
    target: Vec2Int | None,
) -> Vec2Int:
    x, y = current_pos.x, current_pos.y

    if target is None:
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            cand = Vec2Int(nx, ny)
            if _cell_walkable_for_ghost(level, cand):
                return cand
            wrap = _get_ghost_wrap_position(level, cand)
            if wrap is not None:
                return wrap
        return current_pos

    tx, ty = target.x, target.y
    best_pos = current_pos
    best_dist = float("inf")

    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        nx, ny = x + dx, y + dy
        cand: Vec2Int | None = None
        if 0 <= nx < level.width and 0 <= ny < level.height:
            if not level.walls[ny][nx]:
                cand = Vec2Int(nx, ny)
        else:
            cand = _get_ghost_wrap_position(level, Vec2Int(nx, ny))
        if cand is None:
            continue
        dist = abs(tx - cand.x) + abs(ty - cand.y)
        if dist < best_dist:
            best_dist = dist
            best_pos = cand

    return best_pos
