from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent_code.common.feature_vector import BaseFeatureVector
from agent_code.common.function_learning_feature_vector import FunctionLearningFeatureVector
from agent_code.common.neighborhood import Neighborhood, Mirror


@dataclass
class FeatureVector(FunctionLearningFeatureVector, BaseFeatureVector):
    @staticmethod
    def size():
        """
        Returns the bit size for the feature vector."""
        return 17

    def to_state(self) -> np.ndarray:
        return np.array([
            *self.shortest_path_to_safety.to_one_hot_encoding(),
            *self.move_to_danger.to_vector(),
            *self.coin_distance.to_one_hot_encoding(),
            *self.crate_distance.to_one_hot_encoding(),
            self.good_bomb,
            # *self.can_move_in_direction.to_vector()
        ])
