# Constants for the Pacman simulation

# Colors (RGB)
WALL_COLOR = (0, 0, 255)
PILL_COLOR = (255, 184, 151)
PACMAN_COLOR = (255, 255, 0)
BARRIER_COLOR = (100, 100, 100)
BG_COLOR = (0, 0, 0)

GHOST_COLORS = {
    "blinky": (255, 0, 0),
    "pinky": (255, 182, 193),
    "inky": (0, 255, 255),
    "clyde": (255, 165, 0)
}

# Rewards
REWARD_STEP = -1
REWARD_PILL = 10
REWARD_POWER_PILL = 50
REWARD_GHOST = 200
REWARD_DIE = -500
REWARD_WIN = 1000

# Level Parsing Colors (RGB)
WALL_RGB = (0x00, 0x00, 0x00)
NO_PILL_RGB = (0xAC, 0x31, 0x31)
PACMAN_RGB = (0xFB, 0xF2, 0x36)
POWER_PILL_RGB = (0xFF, 0xFF, 0xFF)
BARRIER_RGB = (0x89, 0x6E, 0x2F)

GHOST_HEX_TO_NAME = {
    "D85662": "blinky",
    "D67BBA": "pinky",
    "5ECDE4": "inky",
    "DF7026": "clyde",
}

# Directions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

DIR_OFFSETS = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0)
}

# RL Hyperparameters
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.9995
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 100000
TARGET_UPDATE_FREQ = 1000
TRAIN_START = 2000

# General
SCREEN_SIZE = 600
RENDER_FPS = 10
CLYDE_SCATTER_DISTANCE = 8
PINKY_TARGET_OFFSET = 4
INKY_TARGET_OFFSET = 2
POWER_PILL_DURATION = 50 # steps

# Telegram Config
import os
from dotenv import load_dotenv
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Device
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
