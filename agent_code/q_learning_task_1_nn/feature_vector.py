from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from agent_code.common.neighborhood import Neighborhood


@dataclass
class FeatureVector:
    coin_distance: Neighborhood
    coin_exists: bool
    crate_distance: Neighborhood
    crate_exists: bool
    in_danger: bool
    can_move_in_direction: Neighborhood
    bomb_distance: Neighborhood
    bomb_exists: bool
    move_to_danger: Neighborhood

    @staticmethod
    def size() -> int:
        """
        Returns the needed size for 11 bit.

        coin_distance, coin_exists, can_move_in_direction
        """
        return 4 + 1 + 4

    def to_nn_state(self):
        """
        Layout: |xxxx|x|xxxx|
                |    | |
                |    | |- can move in direction
                |    |- coin exists
                |- coin distance

        """
        vector = np.array([*self.coin_distance.to_one_hot_encoding(), self.coin_exists,
                           *self.can_move_in_direction.to_nn_vector()]) * 2 - 1

        return torch.tensor(vector).double()
