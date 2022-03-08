from typing import Tuple


class Player:
    name: str
    score: int
    is_bomb_possible: bool
    position: Tuple[int, int]

    def __init__(self, state: Tuple[str, int, bool, Tuple[int, int]]):
        self.name = state[0]
        self.score = state[1]
        self.is_bomb_possible = state[2]
        self.position = state[3]
