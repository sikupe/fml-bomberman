from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch


@dataclass
class Neighborhood:
    north: int | float | bool = field(default=0)
    south: int | float | bool = field(default=0)
    east: int | float | bool = field(default=0)
    west: int | float | bool = field(default=0)

    def __str__(self):
        return (
            f"Neighborhood [north: {self.north}, south: {self.south}, "
            f"east: {self.east}, west: {self.west}]"
        )

    def __repr__(self):
        return (
            f"Neighborhood [north: {self.north}, south: {self.south}, "
            f"east: {self.east}, west: {self.west}]"
        )

    def __iter__(self):
        self._iter_values = [self.north, self.south, self.east, self.west]
        self._count = 0
        return self

    def __next__(self):
        if self._count == len(self._iter_values):
            raise StopIteration
        result = self._iter_values[self._count]
        self._count += 1
        return result

    def to_feature_vector(self, normalize_max: int) -> np.ndarray:
        if normalize_max >= 0:
            return self.to_vector() / normalize_max
        return self.to_vector().astype("float64")

    def to_vector(self):
        return np.array([self.north, self.south, self.east, self.west])

    def to_one_hot_encoding(self):
        directions = self.to_vector()
        shortest = np.argmin(directions)
        result = np.array([0.0, 0.0, 0.0, 0.0])
        result[shortest] = 1.0
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

    def to_nn_vector(self):
        return self.to_vector() > 0

    def minimum(self):
        return np.min(self.to_vector())

    def maximum(self):
        return np.max(self.to_vector())


@dataclass
class FeatureVector:
    coin_distance: Neighborhood
    coin_exists: bool
    crate_distance: Neighborhood
    crate_exists: bool
    in_danger: bool
    can_move_in_direction: Neighborhood
    bomb_distance: Neighborhood
    bomb_exists: bool
    move_to_danger: Neighborhood

    @staticmethod
    def size() -> int:
        """
        Returns the needed size for 11 bit.

        coin_distance, coin_exists, can_move_in_direction
        """
        return 4 + 1 + 4

    def to_nn_state(self):
        """
        Layout: |xxxx|x|xxxx|
                |    | |
                |    | |- can move in direction
                |    |- coin exists
                |- coin distance

        """
        vector = np.array([*self.coin_distance.to_one_hot_encoding(), self.coin_exists,
                           *self.can_move_in_direction.to_nn_vector()]) * 2 - 1

        return torch.tensor(vector)
