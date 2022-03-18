from unittest.mock import MagicMock
from pytest_mock import MockerFixture
import numpy as np

from agent_code.common.feature_extractor import (
    extract_crates,
    try_to_move_into_safety,
    is_in_danger,
)
from agent_code.common.neighborhood import Neighborhood
from items import Bomb
from settings import BOMB_POWER


def test_try_to_move_into_safety(mocker: MockerFixture):
    """
    Check that the safety neighborhood is an inverse is_in_danger, for each
    direction.
    """
    
    arena = np.array(
        [
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ]
    )
    
    mocker.patch(
        "agent_code.q_learning_task_2.feature_extractor.is_in_danger",
        side_effect=[True, False, True, False],
    )

    origin = (5, 5)
    bombs = [((1, 5), 1)]
    bomb_distance = Neighborhood(4, 3, 2, 4)
    can_move_in_direction = Neighborhood(True, True, True, True)
    # Places that are in danger when there are places that will lead to no
    # danger will be -1
    assert Neighborhood(0, 3, 0, 4) == try_to_move_into_safety(
        arena, origin, bombs, bomb_distance, can_move_in_direction
    )


def test_is_in_danger():
    """
                     B  X
                     |  |
      0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16
    [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1] 0
    [-1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1] 1
    [-1  B -1  0 -1  0 -1  0 -1  0 -1  0 -1  0 -1  0 -1] 2
    [-1  X  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1] 3
    [-1  0 -1  0 -1  B -1  0 -1  0 -1  0 -1  0 -1  0 -1] 4  <-- B
    [-1  0  0  0  0  0  X  0  0  0  0  0  0  0  0  0 -1] 5  <-- X
    [-1  0 -1  0 -1  0 -1  0 -1  0 -1  0 -1  0 -1  0 -1] 6
    [-1  0  0  0  0  0  0  X  0  0  O  0  0  0  0  0 -1] 7
    [-1  0 -1  0 -1  0 -1  0 -1  0 -1  0 -1  0 -1  1 -1] 8
    [-1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1] 9
    [-1  0 -1  0 -1  0 -1  0 -1  0 -1  1 -1  0 -1  0 -1] 10
    [-1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1] 11
    [-1  0 -1  0 -1  0 -1  0 -1  0 -1  0 -1  0 -1  0 -1] 12
    [-1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1] 13
    [-1  0 -1  0 -1  0 -1  0 -1  0 -1  0 -1  0 -1  0 -1] 14
    [-1  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0 -1] 15
    [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1] 16
    """
    arena = np.array(
        [
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ]
    )

    origin = (6, 5)
    bombs = [((5, 4), 1)]
    assert not is_in_danger(arena, origin, bombs)

    origin = (5, 5)
    bombs = [((5, 4), 1)]
    assert is_in_danger(arena, origin, bombs)

    origin = (7, 7)
    bombs = [((10, 7), 1)]
    assert is_in_danger(arena, origin, bombs)
    
    origin = (7, 4)
    bombs = [((5, 4), 1)]
    assert not is_in_danger(arena, origin, bombs)
    
    origin = (7, 3)
    bombs = [((5, 3), 1)]
    assert is_in_danger(arena, origin, bombs)
    
    origin = (7, 3)
    bombs = [((5, 3), 1)]
    arena_cp = arena.copy()
    arena_cp[6,3] = 1
    assert not is_in_danger(arena_cp, origin, bombs)
    
    origin = (7, 3)
    bombs = [((5, 3), 1)]
    arena_cp = arena.copy()
    arena_cp[3,6] = 1
    assert is_in_danger(arena_cp, origin, bombs)

    for i, bol in [(6, False), (7, True), (8, True)]:
        origin = (i, 7)
        mock = MagicMock()
        bombs = [((10, 7), 1)]
        game_bomb = Bomb((10, 7), mock, 2, BOMB_POWER, mock)
        coords = game_bomb.get_blast_coords(arena)
        assert bol == (origin in coords)
        assert bol == is_in_danger(arena, origin, bombs)
        assert (origin in coords) == is_in_danger(arena, origin, bombs)


def test_extract_crates():
    """Test the extract_crates function."""
    arena = np.array(
        [
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ]
    ).T

    wanted_crates = [
        (1, 1),
        (2, 1),
        (15, 15),
        (13, 3),
        (7, 4),
        (5, 6),
    ]
    for (x, y) in wanted_crates:
        arena[x, y] = 1

    received_crates = [np_array.tolist() for np_array in extract_crates(arena)]

    for (x, y) in received_crates:
        assert (x, y) in wanted_crates

    assert len(wanted_crates) == len(received_crates)
