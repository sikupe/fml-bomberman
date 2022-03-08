from __future__ import annotations
from typing import Dict, Any, Tuple, List
import numpy as np

from player import Player


class GameState:
    round: int
    step: int
    field: np.ndarray
    bombs: List[Tuple[int, int], int]
    explosion_map: np.ndarray
    coins: List[Tuple[int, int]]
    self: Player
    others: List[Player]
    user_input: str | None

    def __init__(self, state: Dict[str, Any]):
        self.round: int = state['round']
        self.step: int = state['step']
        self.field: np.ndarray = state['field']
        self.bombs: List[Tuple[int, int], int] = state['bombs']
        self.explosion_map: np.ndarray = state['explosion_map']
        self.coins: List[Tuple[int, int]] = state['coins']
        self.self: Player = Player(state['self'])
        self.others: List[Player] = list(map(Player, state['others']))
        self.user_input: str | None = state['user_input']
