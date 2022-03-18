from agent_code.common.neighborhood import Neighborhood
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


def test_neighborhood_maximum(
    neighborhood_int: Neighborhood,
    neighborhood_float: Neighborhood,
    neighborhood_bool: Neighborhood,
):

    assert 9 == neighborhood_int.maximum()
    assert 9.0 == neighborhood_float.maximum()
    assert 1 == neighborhood_bool.maximum()

    neighborhood = Neighborhood(101, 8, -1, 6)

    assert 101 == neighborhood.maximum()


def test_neighborhood_iter(neighborhood_int):
    res = []
    for i in neighborhood_int:
        assert isinstance(i, int)
        res.append(i)

    assert len(res) == 4


def test_neighborhood_all_same():
    neigh = Neighborhood(float("inf"), float("inf"), float("inf"), float("inf"))
    assert 0 == neigh.to_shortest_binary_encoding()
