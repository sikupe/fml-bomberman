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
        return 11

    @staticmethod
    def size():
        """
        Returns the needed size for FeatureVector.bits() bit.

        in_danger, coin_distance, coin_exists, crate_distance, crate_exists,
        bomb_distance, bomb_exists, move_to_danger, good bomb
        """
        return 2 ** FeatureVector.bits()

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

        if self.in_danger:
            shortest_path = self.shortest_path_to_safety
        elif self.coin_distance.exists:
            shortest_path = self.coin_distance
        else:
            shortest_path = self.crate_distance

        return int(
            + (shortest_path.to_shortest_binary_encoding())
            + (self.move_to_danger.to_binary_encoding() << 2)
            + (self.good_bomb << 6)
            + (self.can_move_in_direction.to_binary_encoding() << 7)
        )
