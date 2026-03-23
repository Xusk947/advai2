from typing import Dict, List, Tuple

from src.agents.base import BaseAgent
from src.config import (
    UP, DOWN, LEFT, RIGHT, DIR_OFFSETS, 
    CLYDE_SCATTER_DISTANCE, PINKY_TARGET_OFFSET, 
    INKY_TARGET_OFFSET
)

GridPos = Tuple[int, int]

class GhostAgent(BaseAgent):
    def __init__(self, name: str, scatter_target: GridPos) -> None:
        super().__init__(name)
        self.scatter_target = scatter_target
        self.mode = "chase"
        self.current_direction = UP

    def get_action(self, state: Dict) -> int:
        pos = state["ghost_positions"][self.name]
        pacman_pos = state["pacman_pos"]
        pacman_dir = state["pacman_dir"]
        blinky_pos = state["ghost_positions"].get("blinky", pos)
        walls = state["walls"]

        target = self.get_target(pacman_pos, pacman_dir, blinky_pos)
        
        opposite = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
        possible_actions = []
        
        for action, (dx, dy) in DIR_OFFSETS.items():
            if action == opposite.get(self.current_direction, -1):
                continue
            
            next_pos = (pos[0] + dx, pos[1] + dy)
            
            if self._is_valid(next_pos, walls):
                dist = (next_pos[0] - target[0])**2 + (next_pos[1] - target[1])**2
                possible_actions.append((dist, action))
        
        if not possible_actions:
            rev = opposite.get(self.current_direction, UP)
            self.current_direction = rev
            return rev

        possible_actions.sort()
        best_action = possible_actions[0][1]
        self.current_direction = best_action
        
        return best_action

    def get_target(self, pacman_pos: GridPos, pacman_dir: int, blinky_pos: GridPos) -> GridPos:
        raise NotImplementedError

    def _is_valid(self, pos: GridPos, walls: List[List[bool]]) -> bool:
        x, y = pos
        if 0 <= y < len(walls) and 0 <= x < len(walls[0]):
            return not walls[y][x]
        return False

class BlinkyAgent(GhostAgent):
    def __init__(self) -> None:
        super().__init__("blinky", (25, 0))

    def get_target(self, pacman_pos: GridPos, pacman_dir: int, blinky_pos: GridPos) -> GridPos:
        return pacman_pos

class PinkyAgent(GhostAgent):
    def __init__(self) -> None:
        super().__init__("pinky", (2, 0))

    def get_target(self, pacman_pos: GridPos, pacman_dir: int, blinky_pos: GridPos) -> GridPos:
        dx, dy = DIR_OFFSETS.get(pacman_dir, (0, 0))
        return (pacman_pos[0] + dx * PINKY_TARGET_OFFSET, pacman_pos[1] + dy * PINKY_TARGET_OFFSET)

class InkyAgent(GhostAgent):
    def __init__(self) -> None:
        super().__init__("inky", (27, 30))

    def get_target(self, pacman_pos: GridPos, pacman_dir: int, blinky_pos: GridPos) -> GridPos:
        dx, dy = DIR_OFFSETS.get(pacman_dir, (0, 0))
        pivot = (pacman_pos[0] + dx * INKY_TARGET_OFFSET, pacman_pos[1] + dy * INKY_TARGET_OFFSET)
        
        vec = (pivot[0] - blinky_pos[0], pivot[1] - blinky_pos[1])
        return (blinky_pos[0] + 2 * vec[0], blinky_pos[1] + 2 * vec[1])

class ClydeAgent(GhostAgent):
    def __init__(self) -> None:
        super().__init__("clyde", (0, 30))

    def get_action(self, state: Dict) -> int:
        pos = state["ghost_positions"]["clyde"]
        pacman_pos = state["pacman_pos"]
        dist = abs(pos[0] - pacman_pos[0]) + abs(pos[1] - pacman_pos[1])
        
        if dist > CLYDE_SCATTER_DISTANCE:
            self.mode = "chase"
        else:
            self.mode = "scatter"
            
        return super().get_action(state)

    def get_target(self, pacman_pos: GridPos, pacman_dir: int, blinky_pos: GridPos) -> GridPos:
        if self.mode == "chase":
            return pacman_pos
        else:
            return self.scatter_target
