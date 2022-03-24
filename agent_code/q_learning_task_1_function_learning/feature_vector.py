from __future__ import annotations

import numpy as np

from agent_code.common.feature_vector import BaseFeatureVector
from agent_code.common.function_learning_feature_vector import FunctionLearningFeatureVector


class FeatureVector(FunctionLearningFeatureVector, BaseFeatureVector):

    @staticmethod
    def size():
        # coin distance + can move neighborhood
        return 8

    def to_state(self) -> np.array:
        return np.array([*self.coin_distance.to_one_hot_encoding(),
                         *self.can_move_in_direction.to_vector()
                         ], dtype=np.float)
