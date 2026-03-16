from __future__ import annotations

from pathlib import Path

# (fps, label). Turtle is slow so you can follow the game
SPEED_PRESETS = [
    (10, "Turtle"),
    (30, "Slow"),
    (60, "Normal"),
    (120, "Fast"),
    (576, "Rabbit"),
]

UI_STRIP_HEIGHT = 68  # две строки кнопок

# repo root: .../game/renderer/constants.py -> parents[2] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"

GHOST_BODY_FRAMES = ("ghost_0001.png", "ghost_0002.png")
EYE_PLATFORM_IMAGE = "eye_platform.png"
EYE_PUPIL_IMAGE = "eye.png"

PACMAN_FRAMES = ("pacman_0001.png", "pacman_0002.png")

# UI button value for toggling reward popups (not a speed index)
POPUPS_BUTTON_VALUE = -2
# UI button value for toggling debug entity rewards under Pacman/ghosts
DEBUG_REWARDS_BUTTON_VALUE = -3
# Rounds: value 1000 + index (0..4) -> max_steps preset
ROUNDS_BUTTON_BASE = 1000
# Map: 2001 = level 1, 2002 = level 2
MAP_BUTTON_BASE = 2000
MAX_STEPS_PRESETS = [64, 128, 256, 284, 512]
