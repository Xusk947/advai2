import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
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

_ACTIONS = (UP, DOWN, LEFT, RIGHT)


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
            low=0, high=7, 
            shape=(self.level.height, self.level.width), 
            dtype=np.int8
        )

        # --- Pre-bake static numpy arrays from loader lists (done ONCE) ---
        lv = self.level
        self._walls_np = np.array(lv.walls, dtype=bool)
        self._barrier_np = np.array(lv.pacman_barrier, dtype=bool)
        self._pills_orig = np.array(lv.pills, dtype=bool)
        self._power_pills_orig = np.array(lv.power_pills, dtype=bool)

        # Static base layer: walls=1, barriers visible as 0 in obs but drawn separately
        self._static_obs = np.zeros((lv.height, lv.width), dtype=np.int8)
        self._static_obs[self._walls_np] = 1

        # Track remaining pills count for O(1) win-check
        self._total_pills = int(self._pills_orig.sum()) + int(self._power_pills_orig.sum())

        # Pre-build draw cache (wall/barrier rects, pill centers) – never changes
        self._wall_rects: List[Tuple] = []
        self._barrier_rects: List[Tuple] = []
        self._pill_centers: List[Tuple] = []
        self._power_pill_centers: List[Tuple] = []
        cs = self.cell_size
        for y in range(lv.height):
            for x in range(lv.width):
                rect = (x * cs, y * cs, cs, cs)
                center = (x * cs + cs // 2, y * cs + cs // 2)
                if self._walls_np[y, x]:
                    self._wall_rects.append(rect)
                elif self._barrier_np[y, x]:
                    self._barrier_rects.append(rect)
                elif self._pills_orig[y, x]:
                    self._pill_centers.append((center, (x, y)))
                elif self._power_pills_orig[y, x]:
                    self._power_pill_centers.append((center, (x, y)))

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

        # Use numpy arrays for pills (fast vectorised ops)
        self.pills_np = self._pills_orig.copy()
        self.power_pills_np = self._power_pills_orig.copy()
        self._remaining_pills = self._total_pills

        self.score = 0
        self.done = False
        self.frightened_timer = 0
        
        return self._get_obs(), self._get_info()

    def _get_obs(self) -> np.ndarray:
        # Start from static layer (walls already = 1, rest = 0)
        obs = self._static_obs.copy()

        # Overlay dynamic items using numpy fancy indexing
        obs[self.pills_np] = 2
        obs[self.power_pills_np] = 3

        obs[self.pacman_pos[1], self.pacman_pos[0]] = 4

        frightened = self.frightened_timer > 0
        for name, pos in self.ghost_positions.items():
            state = self.ghost_states[name]
            if state == "dead":
                val = 7
            elif frightened:
                val = 6
            else:
                val = 5
            obs[pos[1], pos[0]] = val
            
        return obs

    def _get_info(self) -> Dict:
        return {
            "score": self.score,
            "pacman_pos": tuple(self.pacman_pos),
            "ghost_positions": self.ghost_positions,
            "frightened": self.frightened_timer > 0,
            "remaining_pills": self._remaining_pills,
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
        walls = self._walls_np
        lv = self.level
        for name in self.ghost_positions:
            if self.ghost_states[name] == "dead":
                # Dead ghosts move automatically towards start
                target = lv.ghost_starts[name]
                curr = self.ghost_positions[name]
                best_action = self.ghost_dirs[name]
                min_dist = 999999
                for act in _ACTIONS:
                    dx, dy = DIR_OFFSETS[act]
                    nx = (curr[0] + dx) % lv.width
                    ny = (curr[1] + dy) % lv.height
                    if not walls[ny, nx]:
                        dist = abs(nx - target[0]) + abs(ny - target[1])
                        if dist < min_dist:
                            min_dist = dist
                            best_action = act
                self.ghost_dirs[name] = self._move_entity(
                    self.ghost_positions[name], best_action, self.ghost_dirs[name], is_ghost=True
                )
                # Check if arrived
                if self.ghost_positions[name][0] == target[0] and self.ghost_positions[name][1] == target[1]:
                    self.ghost_states[name] = "alive"
            elif name in actions:
                self.ghost_dirs[name] = self._move_entity(
                    self.ghost_positions[name], actions[name], self.ghost_dirs[name], is_ghost=True
                )
        
        rewards = {name: float(REWARD_STEP) for name in actions}
        env_done = False
        
        pac_x, pac_y = self.pacman_pos
        for name, g_pos in self.ghost_positions.items():
            if g_pos[0] == pac_x and g_pos[1] == pac_y and self.ghost_states[name] != "dead":
                if self.frightened_timer > 0:
                    rewards["pacman"] += float(REWARD_GHOST)
                    rewards[name] -= 100.0
                    self.ghost_states[name] = "dead"
                else:
                    rewards["pacman"] = float(REWARD_DIE)
                    rewards[name] += 50.0  
                    env_done = True
                    self.done = True
                break
        
        if not self.done:
            px, py = pac_x, pac_y
            
            if self.pills_np[py, px]:
                self.pills_np[py, px] = False
                self._remaining_pills -= 1
                self.score += 10
                rewards["pacman"] += float(REWARD_PILL)
                
            elif self.power_pills_np[py, px]:
                self.power_pills_np[py, px] = False
                self._remaining_pills -= 1
                self.score += 50
                rewards["pacman"] += float(REWARD_POWER_PILL)
                self.frightened_timer = 50

            # Win condition: O(1) counter check instead of scanning entire grid
            if self._remaining_pills == 0:
                env_done = True
                self.done = True
                rewards["pacman"] += float(REWARD_WIN)
                for name in self.ghost_positions:
                    rewards[name] -= 100.0
        
        return self._get_obs(), rewards, env_done, False, self._get_info()

    def _move_entity(self, pos: List[int], action: int, current_dir: int, is_ghost: bool) -> int:
        walls = self._walls_np
        barrier = self._barrier_np
        W = self.level.width
        H = self.level.height

        dx, dy = DIR_OFFSETS[action]
        next_x = (pos[0] + dx) % W
        next_y = (pos[1] + dy) % H
        
        if not walls[next_y, next_x]:
            if is_ghost or not barrier[next_y, next_x]:
                pos[0], pos[1] = next_x, next_y
                return action
        
        dx, dy = DIR_OFFSETS[current_dir]
        next_x = (pos[0] + dx) % W
        next_y = (pos[1] + dy) % H
        
        if not walls[next_y, next_x]:
            if is_ghost or not barrier[next_y, next_x]:
                pos[0], pos[1] = next_x, next_y
                return current_dir
                
        if is_ghost:
            for act in _ACTIONS:
                dx, dy = DIR_OFFSETS[act]
                next_x = (pos[0] + dx) % W
                next_y = (pos[1] + dy) % H
                if not walls[next_y, next_x]:
                    pos[0], pos[1] = next_x, next_y
                    return act
                    
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
        
        # Static elements: walls and barriers (pre-cached at init)
        draw_rect = pygame.draw.rect
        draw_circle = pygame.draw.circle
        for rect in self._wall_rects:
            draw_rect(surface, WALL_COLOR, rect)
        for rect in self._barrier_rects:
            draw_rect(surface, BARRIER_COLOR, rect)

        # Dynamic: pills (only still-present ones)
        for center, (x, y) in self._pill_centers:
            if self.pills_np[y, x]:
                draw_circle(surface, PILL_COLOR, center, 2)
        for center, (x, y) in self._power_pill_centers:
            if self.power_pills_np[y, x]:
                draw_circle(surface, PILL_COLOR, center, 6)

        # Draw Pacman
        cs = self.cell_size
        px, py = self.pacman_pos
        p_center = (px * cs + cs // 2, py * cs + cs // 2)
        draw_circle(surface, PACMAN_COLOR, p_center, cs // 2 - 2)

        # Draw Ghosts
        frightened = self.frightened_timer > 0
        for name, pos in self.ghost_positions.items():
            state = self.ghost_states[name]
            gx, gy = pos
            
            if state == "dead":
                color = (255, 255, 255)
            elif frightened:
                color = (0, 0, 255)
            else:
                color = GHOST_COLORS.get(name, (200, 200, 200))
                
            g_rect = (gx * cs + 2, gy * cs + 2, cs - 4, cs - 4)
            draw_rect(surface, color, g_rect)

    def close(self) -> None:
        if self.render_mode == "human":
            pygame.quit()
