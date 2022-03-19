from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent_code.common.neighborhood import Neighborhood, Mirror
from agent_code.common.q_table_feature_vector import QTableFeatureVector


@dataclass
class FeatureVector(QTableFeatureVector):
    coin_distance: Neighborhood
    can_move_in_direction: Neighborhood

    def mirror(self, mirror: Mirror):
        return FeatureVector(self.coin_distance.mirror(mirror), self.can_move_in_direction.mirror(mirror))

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
