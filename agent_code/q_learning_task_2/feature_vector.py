from __future__ import annotations

from dataclasses import dataclass

from agent_code.common.neighborhood import Neighborhood


@dataclass
class FeatureVector:
    coin_distance: Neighborhood
    coin_exists: bool
    crate_distance: Neighborhood
    crate_exists: bool
    in_danger: bool
    bomb_distance: Neighborhood
    bomb_exists: bool
    move_to_danger: Neighborhood

    @staticmethod
    def bits():
        """
        Returns the bit size for the feature vector."""
        return 11

    @staticmethod
    def size():
        """
        Returns the needed size for 11 bit.

        in_danger, coin_distance, coin_exists, crate_distance, crate_exists,
        bomb_distance, bomb_exists, move_to_danger,
        """
        return 2**FeatureVector.bits()

    def to_state(self) -> int:
        """
        Layout: |xxxx|x|xx|x|xx|x|
                |    | |  | |  |
                |    | |  | |  |-in_danger
                |    | |  | |-coin_distance|bomb_distance
                |    | |  |-coin_exists|bomb_exists
                |    | |-crate_distance
                |    |-crate_exists
                |-move_to_danger
        """

        if self.in_danger:
            return int(
                self.in_danger
                + (self.coin_distance.to_shortest_binary_encoding(argmax=True) << 1)
                + (self.coin_exists << 3)
                + (self.crate_distance.to_shortest_binary_encoding() << 4)
                + (self.crate_exists << 6)
                + (self.move_to_danger.to_binary_encoding()<<7)
            )
        else:
            return int(
                self.in_danger
                + (self.bomb_distance.to_shortest_binary_encoding(argmax=True)<<1)
                + (self.bomb_exists << 3)
                + (self.crate_distance.to_shortest_binary_encoding() << 4)
                + (self.crate_exists << 6)
                + (self.move_to_danger.to_binary_encoding()<<7)
            )
