import numpy as np

from agent_code.common.feature_vector import BaseFeatureVector
from agent_code.common.function_learning_feature_vector import FunctionLearningFeatureVector


class FeatureVector(FunctionLearningFeatureVector, BaseFeatureVector):

    @staticmethod
    def size():
        """
        Returns the bit size for the feature vector."""
        return 22

    def to_state(self) -> np.ndarray:
        return np.array([
            *self.shortest_path_to_safety.to_one_hot_encoding(),
            *self.move_to_danger.to_vector(),
            *self.coin_distance.to_one_hot_encoding(),
            *self.crate_distance.to_one_hot_encoding(),
            self.good_bomb,
            *self.opponent_distance.to_one_hot_encoding(),
            self.opponent_distance.exists
        ])
