from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Tuple

from ..entities import Entity
from ..level_loader import LevelDefinition
from ..vec2 import Vec2Int


class Direction(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()

    @property
    def delta(self) -> Tuple[int, int]:
        if self is Direction.UP:
            return (0, -1)
        if self is Direction.DOWN:
            return (0, 1)
        if self is Direction.LEFT:
            return (-1, 0)
        if self is Direction.RIGHT:
            return (1, 0)
        raise ValueError("Unknown direction")


ACTION_TO_DIRECTION: Dict[int, Direction] = {
    0: Direction.UP,
    1: Direction.DOWN,
    2: Direction.LEFT,
    3: Direction.RIGHT,
}


@dataclass
class Pacman(Entity):
    direction: Direction = Direction.LEFT
    powered_up_steps: int = 0

    def move(self, action: int, level: LevelDefinition) -> Vec2Int:
        desired_direction = self.direction
        if action in ACTION_TO_DIRECTION:
            desired_direction = ACTION_TO_DIRECTION[action]

        dx_desired, dy_desired = desired_direction.delta
        desired_pos = Vec2Int(self.pos.x + dx_desired, self.pos.y + dy_desired)

        wrap_pos = self._get_wrap_position(desired_pos, level)
        if wrap_pos is not None:
            self.direction = desired_direction
            self.pos = wrap_pos
        elif self._can_move_to(desired_pos, level):
            self.direction = desired_direction
            self.pos = desired_pos
        else:
            dx_current, dy_current = self.direction.delta
            current_pos = Vec2Int(self.pos.x + dx_current, self.pos.y + dy_current)
            if self._can_move_to(current_pos, level):
                self.pos = current_pos

        # Уменьшение powered_up_steps делается в env.step() после проверки столкновений,
        # иначе при последнем тике силы Pacman считался бы мёртвым при встрече с призраком.
        return self.pos

    @staticmethod
    def _get_wrap_position(desired_pos: Vec2Int, level: LevelDefinition) -> Vec2Int | None:
        """Если desired_pos за краем карты — телепорт на противоположный край.
        Сначала пробуем свою строку/столбец, потом все проходимые клетки противоположного края (тоннель может быть не по центру)."""
        w, h = level.width, level.height

        def try_candidates(candidates: list[tuple[int, int]]) -> Vec2Int | None:
            for wx, wy in candidates:
                if 0 <= wy < h and 0 <= wx < w and Pacman._can_move_to(Vec2Int(wx, wy), level):
                    return Vec2Int(wx, wy)
            return None

        def best_in_column(col: int, prefer_y: int) -> Vec2Int | None:
            """Любая проходимая клетка в колонке col; приоритет — строка prefer_y, затем по возрастанию расстояния."""
            passable = [
                (col, y) for y in range(h)
                if Pacman._can_move_to(Vec2Int(col, y), level)
            ]
            if not passable:
                return None
            passable.sort(key=lambda c: (abs(c[1] - prefer_y), c[1]))
            return Vec2Int(passable[0][0], passable[0][1])

        def best_in_row(row: int, prefer_x: int) -> Vec2Int | None:
            """Любая проходимая клетка в строке row; приоритет — столбец prefer_x."""
            passable = [
                (x, row) for x in range(w)
                if Pacman._can_move_to(Vec2Int(x, row), level)
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

    @staticmethod
    def _can_move_to(pos: Vec2Int, level: LevelDefinition) -> bool:
        if pos.x < 0 or pos.y < 0 or pos.x >= level.width or pos.y >= level.height:
            return False
        return not level.walls[pos.y][pos.x] and not level.pacman_barrier[pos.y][pos.x]

