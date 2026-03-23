from typing import Dict, Tuple

GridPos = Tuple[int, int]

class BaseAgent:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_action(self, state: Dict) -> int:
        raise NotImplementedError
