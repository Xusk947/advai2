import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import torch
from typing import Dict, List, Tuple, Any, Optional

from src.utils.loader import LevelLoader, LevelDefinition
from src.config import (
    UP, DOWN, LEFT, RIGHT, DIR_OFFSETS, 
    RENDER_FPS, WALL_COLOR, PILL_COLOR, 
    PACMAN_COLOR, GHOST_COLORS, BG_COLOR,
    REWARD_STEP, REWARD_PILL, REWARD_POWER_PILL, 
    REWARD_GHOST, REWARD_DIE, REWARD_WIN,
    BARRIER_COLOR, DEVICE
)

class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}

    def __init__(self, level_path: str, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.loader = LevelLoader(level_path)
        self.level: LevelDefinition = self.loader.load()
        
        self.render_mode = render_mode
        self.cell_size = 20
        self.width = self.level.width * self.cell_size
        self.height = self.level.height * self.cell_size
        
        # 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)
        
        # Grid-based observation
        self.observation_space = spaces.Box(
            low=0, high=5, 
            shape=(self.level.height, self.level.width), 
            dtype=np.int8
        )
        
        self.reset()
        
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.pacman_pos = list(self.level.pacman_start)
        self.pacman_dir = RIGHT
        self.ghost_positions = {name: list(pos) for name, pos in self.level.ghost_starts.items()}
        self.ghost_dirs = {name: UP for name in self.ghost_positions}
        # States: "alive", "frightened", "dead"
        self.ghost_states = {name: "alive" for name in self.ghost_positions}
        
        self.pills = [row[:] for row in self.level.pills]
        self.power_pills = [row[:] for row in self.level.power_pills]
        
        self.score = 0
        self.done = False
        self.frightened_timer = 0
        
        return self._get_obs(), self._get_info()

    def _get_obs(self) -> np.ndarray:
        # Construct grid observation
        # 0: Empty, 1: Wall, 2: Pill, 3: Power Pill, 4: Pacman, 5: Ghost, 6: Frightened, 7: Dead
        obs = np.zeros((self.level.height, self.level.width), dtype=np.int8)
        
        for y in range(self.level.height):
            for x in range(self.level.width):
                if self.level.walls[y][x]: 
                    obs[y][x] = 1
                elif self.pills[y][x]: 
                    obs[y][x] = 2
                elif self.power_pills[y][x]: 
                    obs[y][x] = 3
        
        obs[self.pacman_pos[1]][self.pacman_pos[0]] = 4
        
        for name, pos in self.ghost_positions.items():
            state = self.ghost_states[name]
            if state == "dead": val = 7
            elif self.frightened_timer > 0: val = 6
            else: val = 5
            obs[pos[1]][pos[0]] = val
            
        return obs

    def _get_info(self) -> Dict:
        return {
            "score": self.score,
            "pacman_pos": tuple(self.pacman_pos),
            "ghost_positions": self.ghost_positions,
            "pills": self.pills,
            "frightened": self.frightened_timer > 0
        }

    def step(self, actions: Dict[str, int]) -> Tuple[np.ndarray, Dict[str, float], bool, bool, Dict]:
        if self.done:
            return self._get_obs(), {k: 0.0 for k in actions}, True, False, self._get_info()

        # Tick frightened timer
        if self.frightened_timer > 0:
            self.frightened_timer -= 1

        # Update Pacman
        self.pacman_dir = self._move_entity(self.pacman_pos, actions["pacman"], self.pacman_dir, is_ghost=False)
        
        # Update Ghosts
        for name in self.ghost_positions:
            if self.ghost_states[name] == "dead":
                # Dead ghosts move automatically towards start
                target = self.level.ghost_starts[name]
                curr = self.ghost_positions[name]
                # Simple greedy movement
                best_action = self.ghost_dirs[name]
                min_dist = 1e9
                for act in [UP, DOWN, LEFT, RIGHT]:
                    dx, dy = DIR_OFFSETS[act]
                    nx, ny = (curr[0] + dx) % self.level.width, (curr[1] + dy) % self.level.height
                    if not self.level.walls[ny][nx]:
                        dist = abs(nx - target[0]) + abs(ny - target[1])
                        if dist < min_dist:
                            min_dist = dist
                            best_action = act
                self.ghost_dirs[name] = self._move_entity(self.ghost_positions[name], best_action, self.ghost_dirs[name], is_ghost=True)
                
                # Check if arrived
                if tuple(self.ghost_positions[name]) == target:
                    self.ghost_states[name] = "alive"
            elif name in actions:
                self.ghost_dirs[name] = self._move_entity(self.ghost_positions[name], actions[name], self.ghost_dirs[name], is_ghost=True)
        
        rewards = {name: float(REWARD_STEP) for name in actions}
        env_done = False
        
        pac_pos = tuple(self.pacman_pos)
        for name, g_pos in self.ghost_positions.items():
            if tuple(g_pos) == pac_pos and self.ghost_states[name] != "dead":
                if self.frightened_timer > 0:
                    # Pacman eats ghost
                    rewards["pacman"] += float(REWARD_GHOST)
                    rewards[name] -= 100.0
                    self.ghost_states[name] = "dead"
                else:
                    # Pacman dies
                    rewards["pacman"] = float(REWARD_DIE)
                    rewards[name] += 50.0  
                    env_done = True
                    self.done = True
                break
        
        if not self.done:
            # Eating items
            px, py = self.pacman_pos
            
            if self.pills[py][px]:
                self.pills[py][px] = False
                self.score += 10
                rewards["pacman"] += float(REWARD_PILL)
                
            elif self.power_pills[py][px]:
                self.power_pills[py][px] = False
                self.score += 50
                rewards["pacman"] += float(REWARD_POWER_PILL)
                self.frightened_timer = 50 # Original duration
            
            # Win condition
            if not any(any(row) for row in self.pills) and not any(any(row) for row in self.power_pills):
                env_done = True
                self.done = True
                rewards["pacman"] += float(REWARD_WIN)
                for name in self.ghost_positions:
                    rewards[name] -= 100.0 # Penalty for ghosts losing
        
        return self._get_obs(), rewards, env_done, False, self._get_info()

    def _move_entity(self, pos: List[int], action: int, current_dir: int, is_ghost: bool) -> int:
        # returns the actual direction moved
        dx, dy = DIR_OFFSETS[action]
        next_x = (pos[0] + dx) % self.level.width
        next_y = (pos[1] + dy) % self.level.height
        
        # Try new action
        if not self.level.walls[next_y][next_x]:
            if is_ghost or not self.level.pacman_barrier[next_y][next_x]:
                pos[0], pos[1] = next_x, next_y
                return action
        
        # If new action failed, try current momentum
        dx, dy = DIR_OFFSETS[current_dir]
        next_x = (pos[0] + dx) % self.level.width
        next_y = (pos[1] + dy) % self.level.height
        
        if not self.level.walls[next_y][next_x]:
            if is_ghost or not self.level.pacman_barrier[next_y][next_x]:
                pos[0], pos[1] = next_x, next_y
                return current_dir
                
        # If still stuck, stay but return current_dir (momentum preserved)
        return current_dir

    def render(self) -> None:
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
            
            self.screen.fill(BG_COLOR)
            self._draw_to_surface(self.screen)
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def render_array(self) -> np.ndarray:
        if self.render_mode != "rgb_array":
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        surface = pygame.Surface((self.width, self.height))
        self._draw_to_surface(surface)
        return pygame.surfarray.array3d(surface).transpose(1, 0, 2)

    def _draw_to_surface(self, surface: pygame.Surface) -> None:
        surface.fill(BG_COLOR)
        
        # Draw Walls and Pills
        for y in range(self.level.height):
            for x in range(self.level.width):
                rect = (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                
                if self.level.walls[y][x]:
                    pygame.draw.rect(surface, WALL_COLOR, rect)
                elif self.pills[y][x]:
                    center = (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2)
                    pygame.draw.circle(surface, PILL_COLOR, center, 2)
                elif self.power_pills[y][x]:
                    center = (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2)
                    pygame.draw.circle(surface, PILL_COLOR, center, 6)
                elif self.level.pacman_barrier[y][x]:
                    pygame.draw.rect(surface, BARRIER_COLOR, rect)

        # Draw Pacman
        px, py = self.pacman_pos
        p_center = (px * self.cell_size + self.cell_size // 2, py * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(surface, PACMAN_COLOR, p_center, self.cell_size // 2 - 2)

        # Draw Ghosts
        for name, pos in self.ghost_positions.items():
            state = self.ghost_states[name]
            gx, gy = pos
            
            if state == "dead":
                color = (255, 255, 255) # White Eyes
            elif self.frightened_timer > 0:
                color = (0, 0, 255) # Blue Frightened
            else:
                color = GHOST_COLORS.get(name, (200, 200, 200))
                
            g_rect = (gx * self.cell_size + 2, gy * self.cell_size + 2, self.cell_size - 4, self.cell_size - 4)
            pygame.draw.rect(surface, color, g_rect)

    def close(self) -> None:
        if self.render_mode == "human":
            pygame.quit()
