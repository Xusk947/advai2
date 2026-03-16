from __future__ import annotations

from dataclasses import dataclass

from .vec2 import Vec2Int


@dataclass
class Entity:
    pos: Vec2Int

    def set_pos(self, pos: Vec2Int) -> None:
        self.pos = pos

    def move_by(self, delta: Vec2Int) -> None:
        self.pos = self.pos + delta

