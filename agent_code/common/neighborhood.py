from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import numpy as np

import events


class Mirror(Enum):
    NO_MIRROR = 0,
    X_AXIS = 1,
    Y_AXIS = 2,
    DIAGONAL_LEFT_DOWN_RIGHT_TOP = 3,
    DIAGONAL_LEFT_TOP_RIGHT_DOWN = 4,
    ROT_CLOCKWISE_1 = 5,
    ROT_CLOCKWISE_2 = 6,
    ROT_CLOCKWISE_3 = 7,

    @staticmethod
    def mirror_action(mirror: Mirror, action: str) -> str:
        if action == 'BOMB' or action == 'WAIT':
            return action

        neighborhood = Neighborhood()
        if action == 'UP':
            neighborhood.north = True
        elif action == 'RIGHT':
            neighborhood.east = True
        elif action == 'DOWN':
            neighborhood.south = True
        elif action == 'LEFT':
            neighborhood.west = True

        mirrored = neighborhood.mirror(mirror)

        if mirrored.north:
            return 'UP'
        elif mirrored.south:
            return 'DOWN'
        elif mirrored.east:
            return 'RIGHT'
        elif mirrored.west:
            return 'LEFT'

        raise ValueError(f"Unexpected value provided: {action}")

    @staticmethod
    def mirror_events(mirror: Mirror, e: str | List[str]) -> str | List[str]:
        if isinstance(e, list):
            return [Mirror.mirror_events(mirror, event) for event in e]

        mirrored_events = [events.MOVED_UP, events.MOVED_DOWN, events.MOVED_RIGHT, events.MOVED_LEFT]
        if e not in mirrored_events:
            return e

        neighborhood = Neighborhood()
        if e == events.MOVED_UP:
            neighborhood.north = True
        elif e == events.MOVED_RIGHT:
            neighborhood.east = True
        elif e == events.MOVED_DOWN:
            neighborhood.south = True
        elif e == events.MOVED_LEFT:
            neighborhood.west = True

        mirrored = neighborhood.mirror(mirror)

        if mirrored.north:
            return events.MOVED_UP
        elif mirrored.south:
            return events.MOVED_DOWN
        elif mirrored.east:
            return events.MOVED_RIGHT
        elif mirrored.west:
            return events.MOVED_LEFT

        raise ValueError(f"Unexpected value provided: {e}")


@dataclass
class Neighborhood:
    north: int | float | bool = field(default=0)
    south: int | float | bool = field(default=0)
    east: int | float | bool = field(default=0)
    west: int | float | bool = field(default=0)
    exists: bool = field(default=False, init=False)

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
        min_val = np.min(directions)
        result = np.array([0.0, 0.0, 0.0, 0.0])
        if not np.all(directions == min_val):
            result[shortest] = 1.0
        return result

    def to_shortest_binary_encoding(self, argmax=False):
        if argmax:
            return np.argmax(self.to_vector())
        return np.argmin(self.to_vector())

    def to_feature_encoding(self, argmax=False) -> int:
        if not self.exists:
            return 4
        if argmax:
            return int(np.argmax(self.to_vector()))
        return int(np.argmin(self.to_vector()))

    def to_binary_encoding(self) -> int:
        """
        Returns: Returns 4 bits
        """
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

    def mirror(self, mirror_state: Mirror):
        if mirror_state == Mirror.X_AXIS:
            return Neighborhood(self.south, self.north, self.east, self.west, self.exists)
        elif mirror_state == Mirror.Y_AXIS:
            return Neighborhood(self.north, self.south, self.west, self.east, self.exists)
        elif mirror_state == Mirror.DIAGONAL_LEFT_DOWN_RIGHT_TOP:
            return Neighborhood(self.east, self.west, self.north, self.south, self.exists)
        elif mirror_state == Mirror.DIAGONAL_LEFT_TOP_RIGHT_DOWN:
            return Neighborhood(self.west, self.east, self.south, self.north, self.exists)
        elif mirror_state == Mirror.ROT_CLOCKWISE_1:
            return Neighborhood(self.west, self.east, self.north, self.south, self.exists)
        elif mirror_state == Mirror.ROT_CLOCKWISE_2:
            return Neighborhood(self.south, self.north, self.west, self.east, self.exists)
        elif mirror_state == Mirror.ROT_CLOCKWISE_3:
            return Neighborhood(self.east, self.west, self.south, self.north, self.exists)
        elif mirror_state == Mirror.NO_MIRROR:
            return self
