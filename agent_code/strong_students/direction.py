from enum import Enum


class Direction(Enum):
    NORTH = ("north", (0, 1))
    SOUTH = ("south", (0, -1))
    EAST = ("east", (1, 0))
    WEST = ("west", (-1, 0))
