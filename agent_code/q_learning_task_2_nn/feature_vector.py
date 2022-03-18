from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from agent_code.common.neighborhood import Neighborhood, Mirror


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

    def mirror(self, mirror: Mirror) -> FeatureVector:
        return FeatureVector(self.coin_distance.mirror(mirror), self.coin_exists, self.crate_distance.mirror(mirror),
                             self.crate_exists, self.in_danger, self.can_move_in_direction.mirror(mirror),
                             self.bomb_distance.mirror(mirror), self.bomb_exists, self.move_to_danger.mirror(mirror))

    @staticmethod
    def size() -> int:
        """
        Returns the needed size for 11 bit.

        in_danger, coin_distance, coin_exists, crate_distance, crate_exists,
        can_move_in_direction, move_to_danger
        """
        return 1 + 4 + 1 + 4 + 1 + 4 + 4 + 4

    def to_nn_state(self):
        """
        Layout: |x|xxxx|x|xxxx|x|xxxx|xxxx|
                | |    | |    | |    |
                | |    | |    | |    |- move to danger
                | |    | |    | |- can move in direction
                | |    | |    |- crate exists
                | |    | |- crate distance
                | |    |- coin exists
                | |- coin distance
                |- in danger

        """
        vector = np.array([self.in_danger, *self.coin_distance.to_one_hot_encoding(), self.coin_exists,
                           *self.crate_distance.to_one_hot_encoding(), self.crate_exists,
                           *self.can_move_in_direction.to_nn_vector(), *self.move_to_danger.to_nn_vector(),
                           *self.bomb_distance.to_nn_vector()]) * 2 - 1

        return torch.tensor(vector)
