from agent_code.q_learning_task_2.feature_vector import Neighborhood

import pytest


@pytest.fixture
def neighborhood_int() -> Neighborhood:
    """Neighborhood int fixture."""
    neighborhood = Neighborhood()
    neighborhood.north = 9
    neighborhood.south = 8
    neighborhood.east = 7
    neighborhood.west = 6
    return neighborhood


@pytest.fixture
def neighborhood_float() -> Neighborhood:
    """Neighborhood float fixture."""
    neighborhood = Neighborhood()
    neighborhood.north = 9.0
    neighborhood.south = 8.0
    neighborhood.east = 7.0
    neighborhood.west = 6.0
    return neighborhood


@pytest.fixture
def neighborhood_bool() -> Neighborhood:
    """Neighborhood bool fixture."""
    neighborhood = Neighborhood(True, False, False, False)
    return neighborhood
