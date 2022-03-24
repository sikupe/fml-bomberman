from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from agent_code.common.feature_vector import BaseFeatureVector
from agent_code.common.neighborhood import Neighborhood
from agent_code.common.q_table_feature_vector import QTableFeatureVector


@dataclass
class FeatureVector(QTableFeatureVector, BaseFeatureVector):
    @staticmethod
    def bits():
        """
        Returns the bit size for the feature vector.
        1 + 2 + 1 + 2 + 1 + 2 + 2 + 1 + 4 + 1

        in_danger                       1
        coin_distance | crate_distance  2
        coin_exists   | crate_exists    1
        shortest_path_to_safety         2
        opponent_distance               2
        has_opponents                   1
        move_to_danger                  4
        good_bomb                       1
        """
        return 14

    @staticmethod
    def size():
        """
        Returns the needed size for FeatureVector.bits() bit.

        in_danger, coin_distance, coin_exists, crate_distance, crate_exists, bomb_distance, bomb_exists, move_to_danger, good bomb
        """
        return 2 ** FeatureVector.bits()

    def shortest_useful_path(self):
        if self.in_danger:
            return self.shortest_path_to_safety
        elif self.coin_distance.exists:
            return self.coin_distance
        elif self.crate_distance.exists:
            return self.crate_distance
        else:
            return self.opponent_distance

    def coin_crate(self) -> Tuple[Neighborhood, bool]:
        if self.crate_distance.exists:
            return self.coin_distance, self.coin_distance.exists
        if self.crate_distance.exists:
            return self.crate_distance, self.crate_distance.exists

        return Neighborhood(
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf")
        ), False

    def to_state(self) -> int:
        """
        Layout: |x|xxxx|x|xx|xx|x|xx|x|

        in_danger                       1
        coin_distance | crate_distance  2
        coin_exists   | crate_exists    1
        shortest_path_to_safety         2
        opponent_distance               2
        has_opponents                   1
        move_to_danger                  4
        good_bomb                       1
        """

        coin_crate_distance, coin_crate_exists = self.coin_crate()

        return int(
            + self.in_danger
            + (coin_crate_distance.to_shortest_binary_encoding() << 1)
            + (coin_crate_exists << 3)
            + (self.shortest_path_to_safety.to_shortest_binary_encoding() << 4)
            + (self.opponent_distance.to_shortest_binary_encoding() << 6)
            + (self.has_opponents << 8)
            + (self.move_to_danger.to_binary_encoding() << 9)
            + (self.good_bomb << 13)
        )
