from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


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
    """
    Loads a level layout from a PNG image

    Color semantics for level1.png, hex without alpha:
    - 000000, walls
    - AC3131, play area without pills
    - FBF236, Pacman start cell
    - FFFFFF, power pills
    - D67BBA, pink ghost, Pinky
    - DF7026, orange ghost, Clyde
    - D85662, red ghost, Blinky
    - 5ECDE4, cyan ghost, Inky
    - 896E2F, Pacman barrier, ghosts may pass through
    - alpha equals zero, regular pill
    """

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

        WALL_RGB = (0x00, 0x00, 0x00)
        NO_PILL_RGB = (0xAC, 0x31, 0x31)
        PACMAN_RGB = (0xFB, 0xF2, 0x36)
        POWER_PILL_RGB = (0xFF, 0xFF, 0xFF)
        BARRIER_RGB = (0x89, 0x6E, 0x2F)

        ghost_hex_to_name: Dict[str, str] = {
            "D85662": "blinky",
            "D67BBA": "pinky",
            "5ECDE4": "inky",
            "DF7026": "clyde",
        }

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
                elif rgb_hex in ghost_hex_to_name:
                    ghost_key = ghost_hex_to_name[rgb_hex]
                    ghost_starts[ghost_key] = (x, y)
                else:
                    pills[y][x] = True

        if pacman_start is None:
            raise ValueError("Pacman start position not found in level image")

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
