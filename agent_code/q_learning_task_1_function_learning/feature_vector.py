from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent_code.common.function_learning_feature_vector import FunctionLearningFeatureVector
from agent_code.common.neighborhood import Neighborhood, Mirror


@dataclass
class FeatureVector(FunctionLearningFeatureVector):
    coin_distance: Neighborhood
    can_move_in_direction: Neighborhood

    def mirror(self, mirror: Mirror):
        return FeatureVector(self.coin_distance.mirror(mirror), self.can_move_in_direction.mirror(mirror))

    @staticmethod
    def size():
        # coin distance + can move neighborhood
        return 8

    def to_state(self) -> np.array:
        return np.array([*self.coin_distance.to_one_hot_encoding(),
                         *self.can_move_in_direction.to_vector()
                         ], dtype=np.float)
