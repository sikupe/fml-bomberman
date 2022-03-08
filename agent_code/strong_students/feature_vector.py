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
    opponent_distance: Neighborhood
    coin_distance: Neighborhood
    bomb_distance: Neighborhood
    crate_distance: Neighborhood
    can_move_in_direction: Neighborhood
    # can_escape_from_bomb: Neighborhood
    in_blast_radius: Neighborhood
    in_blast_radius_currently: bool
    number_of_opponents: int

    def to_feature_vector(self, game_size, total_opponent_count):
        return np.concatenate([self.opponent_distance.to_feature_vector(game_size),
                               self.coin_distance.to_feature_vector(game_size),
                               self.bomb_distance.to_feature_vector(game_size),
                               self.crate_distance.to_feature_vector(game_size),
                               self.can_move_in_direction.to_feature_vector(0),
                               self.in_blast_radius.to_feature_vector(game_size),
                               float(self.in_blast_radius_currently),
                               self.number_of_opponents / total_opponent_count])
