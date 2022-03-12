from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class Neighborhood:
    north: int | float | bool
    south: int | float | bool
    east: int | float | bool
    west: int | float | bool

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

    def to_shortest_binary_encoding(self):
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
    can_move_in_direction: Neighborhood

    @staticmethod
    def size():
        # coin distance + can move neighborhood
        return 4 * 2 ** 4

    def to_state(self) -> int:
        return self.coin_distance.to_shortest_binary_encoding() + (self.can_move_in_direction.to_binary_encoding() << 2)

    def to_feature_vector(self):
        return np.concatenate([self.coin_distance.to_one_hot_encoding(),
                               self.can_move_in_direction.to_feature_vector(0)
                               ])
