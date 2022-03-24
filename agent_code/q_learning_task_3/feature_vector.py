from __future__ import annotations

from dataclasses import dataclass

from agent_code.common.feature_vector import BaseFeatureVector
from agent_code.common.q_table_feature_vector import QTableFeatureVector


@dataclass
class FeatureVector(QTableFeatureVector, BaseFeatureVector):
    @staticmethod
    def bits():
        """
        Returns the bit size for the feature vector."""
        return 7

    @staticmethod
    def size():
        """
        Returns the needed size for FeatureVector.bits() bit.

        in_danger, coin_distance, coin_exists, crate_distance, crate_exists,
        bomb_distance, bomb_exists, move_to_danger, good bomb
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

    def to_state(self) -> int:
        """
        Layout:   |x|xxxx|x|xx|x|xx|x|
                  | |    | |  | |  |
                  | |    | |  | |  |-in_danger
                  | |    | |  | |-coin_distance|bomb_distance
                  | |    | |  |-coin_exists|bomb_exists
                  | |    | |-crate_distance
                  | |    |-crate_exists
                  | |-move_to_danger
                  |-good bomb
        """

        shortest_path = self.shortest_useful_path()

        return int(
            + (shortest_path.to_shortest_binary_encoding())
            + (self.move_to_danger.to_binary_encoding() << 2)
            + (self.good_bomb << 6)
        )
