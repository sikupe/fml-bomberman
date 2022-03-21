from __future__ import annotations

from dataclasses import dataclass

from agent_code.common.neighborhood import Neighborhood, Mirror
from agent_code.common.q_table_feature_vector import QTableFeatureVector


@dataclass
class FeatureVector(QTableFeatureVector):
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
    opponent_distance: Neighborhood
    has_opponents: bool

    def mirror(self, mirror: Mirror) -> FeatureVector:
        return FeatureVector(self.coin_distance.mirror(mirror), self.coin_exists, self.crate_distance.mirror(mirror),
                             self.crate_exists, self.in_danger, self.bomb_distance.mirror(mirror),
                             self.bomb_exists, self.move_to_danger.mirror(mirror), self.bomb_drop_safe, self.good_bomb,
                             self.shortest_path_to_safety.mirror(mirror), self.can_move_in_direction.mirror(mirror),
                             self.opponent_distance.mirror(mirror), self.has_opponents)

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
        elif self.has_opponents:
            shortest_path = self.opponent_distance
        elif self.coin_exists:
            shortest_path = self.coin_distance
        else:
            shortest_path = self.crate_distance

        return int(
            + (shortest_path.to_shortest_binary_encoding())
            + (self.move_to_danger.to_binary_encoding() << 2)
            + (self.good_bomb << 6)
            + (self.can_move_in_direction.to_binary_encoding() << 7)
        )
