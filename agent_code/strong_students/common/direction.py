from enum import Enum


class Direction(Enum):
    NORTH = ('north', (0, -1))
    SOUTH = ('south', (0, 1))
    EAST = ('east', (1, 0))
    WEST = ('west', (-1, 0))

    @staticmethod
    def from_action(action: str):
        if action == 'UP':
            return Direction.NORTH
        elif action == 'DOWN':
            return Direction.SOUTH
        elif action == 'LEFT':
            return Direction.WEST
        elif action == 'RIGHT':
            return Direction.EAST
