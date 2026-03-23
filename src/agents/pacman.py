import random
from typing import Dict, List, Optional, Tuple

from src.agents.base import BaseAgent
from src.config import UP, RIGHT, DIR_OFFSETS

GridPos = Tuple[int, int]

class PacmanAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("pacman")

    def get_action(self, state: Dict) -> int:
        pacman_pos = state["pacman_pos"]
        ghost_positions = state["ghost_positions"]
        pills = state["pills"]
        walls = state["walls"]
        
        target = self._find_nearest_pill(pacman_pos, pills, walls)
        
        if not target:
            return random.randint(UP, RIGHT)

        best_action = UP
        min_score = float("inf")
        
        for action, (dx, dy) in DIR_OFFSETS.items():
            next_pos = (pacman_pos[0] + dx, pacman_pos[1] + dy)
            
            if self._is_valid(next_pos, walls):
                dist_to_target = abs(next_pos[0] - target[0]) + abs(next_pos[1] - target[1])
                ghost_penalty = self._calculate_ghost_penalty(next_pos, ghost_positions)
                
                score = dist_to_target + ghost_penalty
                
                if score < min_score:
                    min_score = score
                    best_action = action
                    
        return best_action

    def _calculate_ghost_penalty(self, pos: GridPos, ghost_positions: Dict[str, GridPos]) -> int:
        penalty = 0
        
        for g_pos in ghost_positions.values():
            dist = abs(pos[0] - g_pos[0]) + abs(pos[1] - g_pos[1])
            
            if dist < 2:
                penalty += 1000
            elif dist < 4:
                penalty += 100
                
        return penalty

    def _is_valid(self, pos: GridPos, walls: List[List[bool]]) -> bool:
        x, y = pos
        
        if 0 <= y < len(walls) and 0 <= x < len(walls[0]):
            return not walls[y][x]
            
        return False

    def _find_nearest_pill(self, start: GridPos, pills: List[List[bool]], walls: List[List[bool]]) -> Optional[GridPos]:
        queue = [start]
        visited = {start}
        
        while queue:
            x, y = queue.pop(0)
            
            if pills[y][x]:
                return (x, y)
            
            for dx, dy in DIR_OFFSETS.values():
                next_pos = (x + dx, y + dy)
                
                if self._is_valid(next_pos, walls) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)
                    
        return None
