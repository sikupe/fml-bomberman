from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent_code.common.neighborhood import Neighborhood


@dataclass
class FeatureVector:
    coin_distance: Neighborhood
    can_move_in_direction: Neighborhood

    @staticmethod
    def size():
        # coin distance + can move neighborhood
        return 4 * 2 ** 4

    def to_state(self) -> int:
        return self.coin_distance.to_shortest_binary_encoding() + (self.can_move_in_direction.to_binary_encoding() << 2)

    def to_feature_vector(self):
        return np.concatenate([self.coin_distance.to_one_hot_encoding(),
                               self.can_move_in_direction.to_feature_vector(0)
                               ])
