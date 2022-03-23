from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent_code.common.function_learning_feature_vector import FunctionLearningFeatureVector
from agent_code.common.neighborhood import Neighborhood, Mirror


@dataclass
class FeatureVector(FunctionLearningFeatureVector):
    coin_distance: Neighborhood
    coin_exists: bool
    crate_distance: Neighborhood
    crate_exists: bool
    in_danger: bool
    bomb_distance: Neighborhood
    bomb_exists: bool
    move_to_danger: Neighborhood
    bomb_drop_safe: bool
    good_bomb: bool
    shortest_path_to_safety: Neighborhood
    can_move_in_direction: Neighborhood

    def mirror(self, mirror: Mirror) -> FeatureVector:
        return FeatureVector(self.coin_distance.mirror(mirror), self.coin_exists, self.crate_distance.mirror(mirror),
                             self.crate_exists, self.in_danger, self.bomb_distance.mirror(mirror),
                             self.bomb_exists, self.move_to_danger.mirror(mirror), self.bomb_drop_safe, self.good_bomb,
                             self.shortest_path_to_safety.mirror(mirror), self.can_move_in_direction.mirror(mirror))

    @staticmethod
    def size():
        """
        Returns the bit size for the feature vector."""
        return 21

    def to_state(self) -> np.ndarray:
        return np.array([
            *self.shortest_path_to_safety.to_one_hot_encoding(),
            *self.move_to_danger.to_vector(),
            *self.coin_distance.to_one_hot_encoding(),
            *self.crate_distance.to_one_hot_encoding(),
            self.good_bomb,
            *self.can_move_in_direction.to_vector()
        ])
