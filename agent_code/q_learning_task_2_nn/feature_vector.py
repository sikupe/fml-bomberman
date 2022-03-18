from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from enum import Enum

import events


class Mirror(Enum):
    X_AXIS = 0,
    Y_AXIS = 1,
    DIAGONAL_LEFT_DOWN_RIGHT_TOP = 2,
    DIAGONAL_LEFT_TOP_RIGHT_DOWN = 3,
    ROT_CLOCKWISE_1 = 4,
    ROT_CLOCKWISE_2 = 5,
    ROT_CLOCKWISE_3 = 6,

    @staticmethod
    def mirror_action(mirror: Mirror, action: str):
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

    @staticmethod
    def mirror_events(mirror: Mirror, e: str | List[str]):
        if type(e) == list:
            return list(map(lambda event: Mirror.mirror_events(mirror, event), e))

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

    def mirror(self, mirror_state: Mirror):
        if mirror_state == Mirror.X_AXIS:
            return Neighborhood(self.south, self.north, self.east, self.west)
        elif mirror_state == Mirror.Y_AXIS:
            return Neighborhood(self.north, self.south, self.west, self.east)
        elif mirror_state == Mirror.DIAGONAL_LEFT_DOWN_RIGHT_TOP:
            return Neighborhood(self.east, self.west, self.north, self.south)
        elif mirror_state == Mirror.DIAGONAL_LEFT_TOP_RIGHT_DOWN:
            return Neighborhood(self.west, self.east, self.south, self.north)
        elif mirror_state == Mirror.ROT_CLOCKWISE_1:
            return Neighborhood(self.west, self.east, self.north, self.south)
        elif mirror_state == Mirror.ROT_CLOCKWISE_2:
            return Neighborhood(self.south, self.north, self.west, self.east)
        elif mirror_state == Mirror.ROT_CLOCKWISE_3:
            return Neighborhood(self.east, self.west, self.south, self.north)


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

    def mirror(self, mirror: Mirror) -> FeatureVector:
        return FeatureVector(self.coin_distance.mirror(mirror), self.coin_exists, self.crate_distance.mirror(mirror),
                             self.crate_exists, self.in_danger, self.can_move_in_direction.mirror(mirror),
                             self.bomb_distance.mirror(mirror), self.bomb_exists, self.move_to_danger.mirror(mirror))

    @staticmethod
    def size() -> int:
        """
        Returns the needed size for 11 bit.

        in_danger, coin_distance, coin_exists, crate_distance, crate_exists,
        can_move_in_direction, move_to_danger
        """
        return 1 + 4 + 1 + 4 + 1 + 4 + 4 + 4

    def to_nn_state(self):
        """
        Layout: |x|xxxx|x|xxxx|x|xxxx|xxxx|
                | |    | |    | |    |
                | |    | |    | |    |- move to danger
                | |    | |    | |- can move in direction
                | |    | |    |- crate exists
                | |    | |- crate distance
                | |    |- coin exists
                | |- coin distance
                |- in danger

        """
        vector = np.array([self.in_danger, *self.coin_distance.to_one_hot_encoding(), self.coin_exists,
                           *self.crate_distance.to_one_hot_encoding(), self.crate_exists,
                           *self.can_move_in_direction.to_nn_vector(), *self.move_to_danger.to_nn_vector(),
                           *self.bomb_distance.to_nn_vector()]) * 2 - 1

        return torch.tensor(vector)
