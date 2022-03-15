from agent_code.q_learning_task_2.feature_vector import Neighborhood
import numpy as np
import itertools


def test_neighborhood_to_vector(
    neighborhood_int: Neighborhood,
    neighborhood_float: Neighborhood,
    neighborhood_bool: Neighborhood,
):
    assert np.all(np.array([9, 8, 7, 6]) == neighborhood_int.to_vector())
    assert np.all(np.array([9.0, 8.0, 7.0, 6.0]) == neighborhood_float.to_vector())
    assert np.all(
        np.array([True, False, False, False]) == neighborhood_bool.to_vector()
    )


def test_neighborhood_to_one_hot_encoding(
    neighborhood_int: Neighborhood,
    neighborhood_float: Neighborhood,
):
    assert np.all(
        np.array([0.0, 0.0, 0.0, 1.0]) == neighborhood_int.to_one_hot_encoding()
    )
    assert np.all(
        np.array([0.0, 0.0, 0.0, 1.0]) == neighborhood_float.to_one_hot_encoding()
    )


def test_neighborhood_to_one_hot_encoding_with_bool(
    neighborhood_bool: Neighborhood,
):
    """Will fit [.0, 1., .0, .0] because the first false is at index 1."""
    assert np.all(
        np.array([0.0, 1.0, 0.0, 0.0]) == neighborhood_bool.to_one_hot_encoding()
    )


def test_neighborhood_to_binary_encoding():
    combinations = list(itertools.product([True, False], repeat=4))
    combinations.reverse()
    for i, ele in enumerate(combinations):
        # Get it reversed 0001 => [True, False, False, False]
        west, east, south, north = ele
        assert i == Neighborhood(north, south, east, west).to_binary_encoding()


def test_neighborhood_to_shortest_binary_encoding(
    neighborhood_int: Neighborhood,
    neighborhood_float: Neighborhood,
    neighborhood_bool: Neighborhood,
):

    assert 3 == neighborhood_int.to_shortest_binary_encoding()
    assert 3 == neighborhood_float.to_shortest_binary_encoding()
    assert 1 == neighborhood_bool.to_shortest_binary_encoding()

    neighborhood = Neighborhood()
    neighborhood.north = 9
    neighborhood.south = 8
    neighborhood.east = -1
    neighborhood.west = 6

    assert 2 == neighborhood.to_shortest_binary_encoding()


def test_neighborhood_minimum(
    neighborhood_int: Neighborhood,
    neighborhood_float: Neighborhood,
    neighborhood_bool: Neighborhood,
):

    assert 6 == neighborhood_int.minimum()
    assert 6.0 == neighborhood_float.minimum()
    assert 0 == neighborhood_bool.minimum()

    neighborhood = Neighborhood()
    neighborhood.north = 9
    neighborhood.south = 8
    neighborhood.east = -1
    neighborhood.west = 6

    assert -1 == neighborhood.minimum()
