from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass
class Vec2Int:
    x: int
    y: int

    def __add__(self, other: "Vec2Int") -> "Vec2Int":
        return Vec2Int(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2Int") -> "Vec2Int":
        return Vec2Int(self.x - other.x, self.y - other.y)

    def manhattan_distance(self, other: "Vec2Int") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def to_tuple(self) -> Tuple[int, int]:
        return self.x, self.y

    @classmethod
    def from_tuple(cls, data: Iterable[int]) -> "Vec2Int":
        x, y = data
        return cls(int(x), int(y))

    def neighbors4(self) -> Tuple["Vec2Int", "Vec2Int", "Vec2Int", "Vec2Int"]:
        return (
            Vec2Int(self.x, self.y - 1),
            Vec2Int(self.x, self.y + 1),
            Vec2Int(self.x - 1, self.y),
            Vec2Int(self.x + 1, self.y),
        )

