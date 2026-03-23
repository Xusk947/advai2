from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import PIL.Image as Image

from src.config import (
    WALL_RGB, NO_PILL_RGB, PACMAN_RGB, POWER_PILL_RGB, 
    BARRIER_RGB, GHOST_HEX_TO_NAME
)

GridPos = Tuple[int, int]

@dataclass
class LevelDefinition:
    width: int
    height: int
    walls: List[List[bool]]
    pacman_barrier: List[List[bool]]
    pills: List[List[bool]]
    power_pills: List[List[bool]]
    no_pill_zone: List[List[bool]]
    pacman_start: GridPos
    ghost_starts: Dict[str, GridPos]

class LevelLoader:
    def __init__(self, image_path: str | Path) -> None:
        self.image_path = str(image_path)

    def load(self) -> LevelDefinition:
        path = Path(self.image_path)
        if not path.exists():
            raise FileNotFoundError(path)

        image = Image.open(self.image_path).convert("RGBA")
        width, height = image.size
        pixels = image.load()

        walls = [[False for _ in range(width)] for _ in range(height)]
        pills = [[False for _ in range(width)] for _ in range(height)]
        pacman_barrier = [[False for _ in range(width)] for _ in range(height)]
        power_pills = [[False for _ in range(width)] for _ in range(height)]
        no_pill_zone = [[False for _ in range(width)] for _ in range(height)]

        pacman_start: GridPos | None = None
        ghost_starts: Dict[str, GridPos] = {}

        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]

                if a == 0:
                    pills[y][x] = True
                    continue

                rgb = (r, g, b)
                rgb_hex = f"{r:02X}{g:02X}{b:02X}"

                if rgb == WALL_RGB:
                    walls[y][x] = True
                elif rgb == PACMAN_RGB:
                    pacman_start = (x, y)
                elif rgb == POWER_PILL_RGB:
                    power_pills[y][x] = True
                elif rgb == NO_PILL_RGB:
                    no_pill_zone[y][x] = True
                elif rgb == BARRIER_RGB:
                    pacman_barrier[y][x] = True
                elif rgb_hex in GHOST_HEX_TO_NAME:
                    ghost_key = GHOST_HEX_TO_NAME[rgb_hex]
                    ghost_starts[ghost_key] = (x, y)
                else:
                    pills[y][x] = True

        if pacman_start is None:
            raise ValueError("Pacman start position not found")

        px, py = pacman_start
        pills[py][px] = False

        return LevelDefinition(
            width=width,
            height=height,
            walls=walls,
            pacman_barrier=pacman_barrier,
            pills=pills,
            power_pills=power_pills,
            no_pill_zone=no_pill_zone,
            pacman_start=pacman_start,
            ghost_starts=ghost_starts,
        )
