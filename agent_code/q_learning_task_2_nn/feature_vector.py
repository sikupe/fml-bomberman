from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from agent_code.common.feature_vector import BaseFeatureVector
from agent_code.common.nn_feature_vector import NNFeatureVector


@dataclass
class FeatureVector(NNFeatureVector, BaseFeatureVector):

    @staticmethod
    def size() -> int:
        """
        Returns the needed size for 11 bit.

        in_danger, shortest success path, coin_exists, crate_exists,
        can_move_in_direction, move_to_danger, next_to_bomb_target, shortest path to safety
        """
        return 1 + 4 + 1 + 4 + 1 + 4 + 4 + 1 + 1 + 4

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
        elif self.coin_distance.exists:
            shortest_path = self.coin_distance
        else:
            shortest_path = self.crate_distance

        vector = np.array(
            [self.in_danger, *self.coin_distance.to_nn_vector(), self.coin_distance.exists,
             *self.crate_distance.to_nn_vector(),
             self.crate_distance.exists,
             *self.can_move_in_direction.to_nn_vector(), *self.move_to_danger.to_nn_vector(),
             self.good_bomb, self.good_bomb, *self.shortest_path_to_safety.to_nn_vector()]) * 2 - 1

        return torch.tensor(vector).double()
