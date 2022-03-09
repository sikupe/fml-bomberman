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
            return np.array([self.north / normalize_max, self.south / normalize_max, self.east / normalize_max,
                             self.west / normalize_max])
        return np.array([float(self.north), float(self.south), float(self.east), float(self.west)])


@dataclass
class FeatureVector:
    opponent_distance: int
    coin_distance: int
    bomb_distance: int
    crate_distance: int
    can_move_in_direction: Neighborhood
    in_blast_radius_currently: bool
    number_of_opponents: int

    def to_feature_vector(self, total_opponent_count):
        return np.concatenate([self.opponent_distance,
                               self.coin_distance,
                               self.bomb_distance,
                               self.crate_distance,
                               self.can_move_in_direction.to_feature_vector(0),
                               float(self.in_blast_radius_currently),
                               self.number_of_opponents / total_opponent_count])
