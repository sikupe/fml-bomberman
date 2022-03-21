from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from agent_code.common.neighborhood import Neighborhood, Mirror
from agent_code.common.nn_feature_vector import NNFeatureVector


@dataclass
class FeatureVector(NNFeatureVector):
    coin_distance: Neighborhood
    coin_exists: bool
    crate_distance: Neighborhood
    crate_exists: bool
    in_danger: bool
    can_move_in_direction: Neighborhood
    shortest_path_to_safety: Neighborhood
    bomb_exists: bool
    move_to_danger: Neighborhood
    next_to_bomb_target: bool
    good_bomb: bool

    def mirror(self, mirror: Mirror) -> FeatureVector:
        return FeatureVector(self.coin_distance.mirror(mirror), self.coin_exists, self.crate_distance.mirror(mirror),
                             self.crate_exists, self.in_danger, self.can_move_in_direction.mirror(mirror),
                             self.shortest_path_to_safety.mirror(mirror), self.bomb_exists,
                             self.move_to_danger.mirror(mirror),
                             self.next_to_bomb_target, self.good_bomb)

    @staticmethod
    def size() -> int:
        """
        Returns the needed size for 11 bit.

        in_danger, shortest success path, coin_exists, crate_exists,
        can_move_in_direction, move_to_danger, next_to_bomb_target
        """
        return 1 + 4 + 1 + 1 + 4 + 4 + 1 + 1

    def to_nn_state(self):
        """
        Layout: |x|xxxx|x|xxxx|x|xxxx|xxxx|x|
                | |    | |    | |    |    |- next to bomb target
                | |    | |    | |    |- move to danger
                | |    | |    | |- can move in direction
                | |    | |    |- crate exists
                | |    | |- crate distance
                | |    |- coin exists
                | |- coin distance
                |- in danger

        """

        if self.in_danger:
            shortest_path = self.shortest_path_to_safety
        elif self.coin_exists:
            shortest_path = self.coin_distance
        else:
            shortest_path = self.crate_distance

        vector = np.array([self.in_danger, *shortest_path.to_one_hot_encoding(), self.coin_exists, self.crate_exists,
                           *self.can_move_in_direction.to_nn_vector(), *self.move_to_danger.to_nn_vector(),
                           self.next_to_bomb_target, self.good_bomb]) * 2 - 1

        return torch.tensor(vector)
