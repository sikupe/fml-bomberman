from typing import List

from agent_code.q_learning_task_2.feature_vector import FeatureVector
from agent_code.common.neighborhood import Neighborhood

import pytest


def int_to_list_of_bit(num: int) -> List[int]:
    """
    Returns binary representation of integer in form of a list of length 11.
    https://stackoverflow.com/questions/30971079/how-to-convert-an-integer-to-a-list-of-bits
    """
    return [
        1 if num & (1 << ((FeatureVector.bits() - 1) - n)) else 0
        for n in range(FeatureVector.bits())
    ]


@pytest.fixture
def feature_vector(
    neighborhood_int: Neighborhood,
    neighborhood_bool: Neighborhood,
) -> FeatureVector:
    return FeatureVector(
        neighborhood_int,
        True,
        neighborhood_int,
        True,
        True,
        neighborhood_int,
        True,
        neighborhood_bool,
    )


def test_feature_vector_size(feature_vector: FeatureVector):
    assert 2 ** feature_vector.bits() == feature_vector.size()


def test_feature_vector_to_state(feature_vector: FeatureVector):
    """
    in_danger              = True -> 1
    argmin(coin_distance)  = 3    -> 11
    argmin(crate_distance) = 3    -> 11
    can_move_in_direction  = [True, False, False, False]
                                  -> 0001
    argmax(bomb_distance)  = 0    -> 00
    _______________________________

    Layout: |0001|1|11|1|00|1|
            |    | |  | |  |
            |    | |  | |  |-in_danger
            |    | |  | |-coin_distance
            |    | |  |-coin_exists
            |    | |-crate_distance
            |    |-crate_exists
            |-move_to_danger
    """

    result = 0
    bit_representation = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1]
    bit_representation.reverse()
    for i, v in enumerate(bit_representation):
        result += (v * 2) ** i
    assert result == feature_vector.to_state()

    result = 0
    bit_representation = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert len(bit_representation) == 11
    for i, v in enumerate(bit_representation):
        result += (v * 2) ** i
    assert int_to_list_of_bit(feature_vector.size() - 1) == bit_representation
    assert result == feature_vector.size() - 1
