from pytest_mock import MockerFixture

from agent_code.q_learning_task_2.feature_extractor import (
    try_to_move_into_safety,
    is_in_danger,
)
from agent_code.q_learning_task_2.feature_vector import Neighborhood


def test_try_to_move_into_safety(mocker: MockerFixture):
    """
    Check that the safety neighborhood is an inverse is_in_danger, for each
    direction.
    """
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
    assert Neighborhood(-1, 3, -1, 4) == try_to_move_into_safety(
        origin, bombs, bomb_distance, can_move_in_direction
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
    [-1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1] 7
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
    origin = (6, 5)
    bombs = [((5, 4), 1)]
    assert not is_in_danger(origin, bombs)

    origin = (5, 5)
    bombs = [((5, 4), 1)]
    assert is_in_danger(origin, bombs)
