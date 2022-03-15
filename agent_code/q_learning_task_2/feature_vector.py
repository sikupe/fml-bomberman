from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Neighborhood:
    north: int | float | bool = field(default=0)
    south: int | float | bool = field(default=0)
    east: int | float | bool = field(default=0)
    west: int | float | bool = field(default=0)

    def to_feature_vector(self, normalize_max: int) -> np.ndarray:
        if normalize_max >= 0:
            return self.to_vector() / normalize_max
        return self.to_vector().astype('float64')

    def to_vector(self):
        return np.array([self.north, self.south, self.east, self.west])

    def to_one_hot_encoding(self):
        directions = self.to_vector()
        shortest = np.argmin(directions)
        result = np.array([.0, .0, .0, .0])
        result[shortest] = 1.
        return result

    def to_shortest_binary_encoding(self, argmax=False):
        if argmax:
            return np.argmax(self.to_vector())
        return np.argmin(self.to_vector())

    def to_binary_encoding(self) -> int:
        result = 0
        vec = self.to_vector()
        for i, el in enumerate(vec):
            if el:
                result += 2 ** i
        return result

    def minimum(self):
        return np.min(self.to_vector())


@dataclass
class FeatureVector:
    coin_distance: Neighborhood
    crate_distance: Neighborhood
    in_danger: bool
    can_move_in_direction: Neighborhood
    bomb_distance: Neighborhood

    @staticmethod
    def size():
        """Returns the needed size for 11 bit."""
        # in danger (1 bit) + coin distance (2 bit) + crate distance (2 bit) +
        # can move neighborhood (4 bit) + bomb_distance (2 bit)
        return 1 << 1 << 2 << 2 << 4 << 2

    def to_state(self) -> int:
        """
        Layout: |xx|xxxx|xx|xx|x|
                |  |    |  |  |
                |  |    |  |  |-in_danger
                |  |    |  |-coin_distance
                |  |    |-crate_distance
                |  |-can_move_in_direction
                |-bomb_distance
        """

        return int(
            self.in_danger
            + (self.coin_distance.to_shortest_binary_encoding() << 1)
            + (self.crate_distance.to_shortest_binary_encoding() << 3)
            + (self.can_move_in_direction.to_binary_encoding() << 5)
            + (self.bomb_distance.to_shortest_binary_encoding(argmax=True) << 9)
        )
