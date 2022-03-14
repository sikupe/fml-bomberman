import pytest


@pytest.fixture
def simple_fixture() -> bool:
    return True


def test_example(simple_fixture: bool):
    assert simple_fixture
