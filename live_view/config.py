"""Constants and types for the live view app."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

CHECKPOINT_DIR = Path("checkpoints")
PACMAN_CHECKPOINT = CHECKPOINT_DIR / "pacman_dqn.pt"
GHOST_CHECKPOINT_PATTERN = "ghost_{}_dqn.pt"
EPISODE_NUM_FILE = CHECKPOINT_DIR / "episode.txt"

NUM_ACTIONS = 4
EVAL_EPSILON = 0.3
ANIMATION_FRAMES_PER_STEP = 10
POPUP_DURATION_FRAMES = 40
DEFAULT_SPEED_INDEX = 3
END_SCREEN_FRAMES = 45
NO_AGENT_POLL_INTERVAL = 1.0

RGB = Tuple[int, int, int]
StateShape = Tuple[int, int, int]

POPUP_COLORS: Dict[str, RGB] = {
    "pacman": (255, 255, 0),
    "ghost_blinky": (255, 0, 0),
    "ghost_pinky": (255, 105, 180),
    "ghost_inky": (0, 255, 255),
    "ghost_clyde": (255, 165, 0),
}
