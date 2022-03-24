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

        coin_distance, coin_exists, can_move_in_direction
        """
        return 4 + 4 + 4 + 1

    def to_nn_state(self):
        """
        Layout: |xxxx|x|xxxx|
                |    | |
                |    | |- can move in direction
                |    |- coin exists
                |- coin distance

        """
        vector = np.array([
            *self.coin_distance.to_one_hot_encoding(),
            # self.coin_exists,
            # *self.can_move_in_direction.to_nn_vector(),
            *self.shortest_path_to_safety.to_one_hot_encoding(),
            *self.move_to_danger.to_nn_vector(),
            # self.in_danger,
            # *self.crate_distance.to_one_hot_encoding(),
            self.good_bomb,
            # self.crate_exists
        ]) * 2 - 1

        return torch.tensor(vector).double()
