from typing import Tuple

from agent_code.common.types import Position


class Player:
    name: str
    score: int
    is_bomb_possible: bool
    position: Position

    def __init__(self, state: Tuple[str, int, bool, Position]):
        self.name = state[0]
        self.score = state[1]
        self.is_bomb_possible = state[2]
        self.position = state[3]
