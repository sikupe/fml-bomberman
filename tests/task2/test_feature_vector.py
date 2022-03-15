from agent_code.q_learning_task_2.feature_vector import FeatureVector, Neighborhood

import pytest


def int_to_list_of_bit(num: int) -> list[int]:
    """
    Returns binary representation of integer in form of a list of length 11.
    https://stackoverflow.com/questions/30971079/how-to-convert-an-integer-to-a-list-of-bits
    """
    return [1 if num & (1 << (10-n)) else 0 for n in range(11)]


@pytest.fixture
def feature_vector(
    neighborhood_int: Neighborhood,
    neighborhood_bool: Neighborhood,
) -> FeatureVector:
    return FeatureVector(
        neighborhood_int, neighborhood_int, True, neighborhood_bool, neighborhood_int
    )


def test_feature_vector_size(feature_vector: FeatureVector):
    assert 2 ** 11 == feature_vector.size()


def test_feature_vector_to_state(feature_vector: FeatureVector):
    """
    in_danger              = True -> 1
    argmax(coin_distance)  = 3    -> 11
    argmax(crate_distance) = 3    -> 11
    can_move_in_direction  = [True, False, False, False]
                                  -> 0001
    argmax(bomb_distance)  = 3    -> 11
    _______________________________

    |11|0001|11|11|1|
    |  |    |  |  |
    |  |    |  |  |-in_danger
    |  |    |  |-coin_distance
    |  |    |-crate_distance
    |  |-can_move_in_direction
    |-bomb_distance
    """
    result = 0
    bit_representation = [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    bit_representation.reverse()
    for i, v in enumerate(bit_representation):
        result += (v*2)**i
    assert result == feature_vector.to_state()

    result = 0
    bit_representation = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for i, v in enumerate(bit_representation):
        result += (v*2)**i
    assert result == feature_vector.size()-1

    assert int_to_list_of_bit(feature_vector.size()-1) == bit_representation
