from __future__ import annotations

from dataclasses import dataclass

from functools import reduce
import operator

from agent_code.common.feature_vector import BaseFeatureVector
from agent_code.common.neighborhood import Neighborhood, Mirror
from agent_code.common.q_table_feature_vector import QTableFeatureVector


class FeatureVector(BaseFeatureVector, QTableFeatureVector):

    @staticmethod
    def size():
        """
        Returns the needed size for FeatureVector.bits() bit.

        in_danger, coin_distance, coin_exists, crate_distance, crate_exists,
        bomb_distance, bomb_exists, move_to_danger, good bomb
        """
        return reduce(operator.mul, FeatureVector.encoding())

    @staticmethod
    def feature_encoding(unencoded_state: list[int], encoding: list[int]) -> int:
        result = [reduce(operator.mul, encoding[:idx + 1]) for idx in range(len(encoding))]
        feature_encoding = [1] + result[:-1]
        return sum([val * i for val, i in zip(unencoded_state, feature_encoding)])

    @staticmethod
    def encoding() -> list[int]:
        """
        Returns the bit size for the feature vector."""
        return [5, 5, 5, 5, 16, 2]

    def to_state(self) -> int:
        """
        Layout: 5 coins
                5 crates
                5 opponents
                5 shortest_path_to_safety -> exists:
                    -> in_danger
                16 move_into_danger
                2 good_bomb
        """
        unencoded_state = [
            self.coin_distance.to_feature_encoding(),       # 5
            self.crate_distance.to_feature_encoding(),      # 5
            self.opponent_distance.to_feature_encoding(),   # 5
            self.shortest_path_to_safety.to_feature_encoding(),   # 5
            self.move_to_danger.to_binary_encoding(),
            int(self.good_bomb)
        ]
        return self.feature_encoding(unencoded_state, self.encoding())
