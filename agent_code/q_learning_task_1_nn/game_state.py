from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from agent_code.q_learning_task_1_nn.types import Position, Bomb
from agent_code.q_learning_task_1_nn.player import Player


@dataclass
class GameState:
    round: int
    step: int
    field: np.ndarray
    bombs: List[Bomb]
    explosion_map: np.ndarray
    coins: List[Position]
    self: Player
    others: List[Player]
    user_input: str | None
