from agent_code.common.neighborhood import Neighborhood, Mirror


def test_mirror_x_axis_north():
    neighborhood = Neighborhood(True, False, False, False)

    mirrored = neighborhood.mirror(Mirror.X_AXIS)

    assert not mirrored.north
    assert mirrored.south
    assert not mirrored.east
    assert not mirrored.west


def test_mirror_x_axis_south():
    neighborhood = Neighborhood(False, True, False, False)

    mirrored = neighborhood.mirror(Mirror.X_AXIS)

    assert mirrored.north
    assert not mirrored.south
    assert not mirrored.east
    assert not mirrored.west


def test_mirror_x_axis_east():
    neighborhood = Neighborhood(False, False, True, False)

    mirrored = neighborhood.mirror(Mirror.X_AXIS)

    assert not mirrored.north
    assert not mirrored.south
    assert mirrored.east
    assert not mirrored.west


def test_mirror_x_axis_west():
    neighborhood = Neighborhood(False, False, False, True)

    mirrored = neighborhood.mirror(Mirror.X_AXIS)

    assert not mirrored.north
    assert not mirrored.south
    assert not mirrored.east
    assert mirrored.west


def test_mirror_y_axis_north():
    neighborhood = Neighborhood(True, False, False, False)

    mirrored = neighborhood.mirror(Mirror.Y_AXIS)

    assert mirrored.north
    assert not mirrored.south
    assert not mirrored.east
    assert not mirrored.west


def test_mirror_y_axis_south():
    neighborhood = Neighborhood(False, True, False, False)

    mirrored = neighborhood.mirror(Mirror.Y_AXIS)

    assert not mirrored.north
    assert mirrored.south
    assert not mirrored.east
    assert not mirrored.west


def test_mirror_y_axis_east():
    neighborhood = Neighborhood(False, False, True, False)

    mirrored = neighborhood.mirror(Mirror.Y_AXIS)

    assert not mirrored.north
    assert not mirrored.south
    assert not mirrored.east
    assert mirrored.west


def test_mirror_y_axis_west():
    neighborhood = Neighborhood(False, False, False, True)

    mirrored = neighborhood.mirror(Mirror.Y_AXIS)

    assert not mirrored.north
    assert not mirrored.south
    assert mirrored.east
    assert not mirrored.west


def test_mirror_diagonal_left_down_right_top_north():
    neighborhood = Neighborhood(True, False, False, False)

    mirrored = neighborhood.mirror(Mirror.DIAGONAL_LEFT_DOWN_RIGHT_TOP)

    assert not mirrored.north
    assert not mirrored.south
    assert mirrored.east
    assert not mirrored.west


def test_mirror_diagonal_left_down_right_top_south():
    neighborhood = Neighborhood(False, True, False, False)

    mirrored = neighborhood.mirror(Mirror.DIAGONAL_LEFT_DOWN_RIGHT_TOP)

    assert not mirrored.north
    assert not mirrored.south
    assert not mirrored.east
    assert mirrored.west


def test_mirror_diagonal_left_down_right_top_east():
    neighborhood = Neighborhood(False, False, True, False)

    mirrored = neighborhood.mirror(Mirror.DIAGONAL_LEFT_DOWN_RIGHT_TOP)

    assert mirrored.north
    assert not mirrored.south
    assert not mirrored.east
    assert not mirrored.west


def test_mirror_diagonal_left_down_right_top_west():
    neighborhood = Neighborhood(False, False, False, True)

    mirrored = neighborhood.mirror(Mirror.DIAGONAL_LEFT_DOWN_RIGHT_TOP)

    assert not mirrored.north
    assert mirrored.south
    assert not mirrored.east
    assert not mirrored.west


def test_mirror_diagonal_left_top_right_down_north():
    neighborhood = Neighborhood(True, False, False, False)

    mirrored = neighborhood.mirror(Mirror.DIAGONAL_LEFT_TOP_RIGHT_DOWN)

    assert not mirrored.north
    assert not mirrored.south
    assert not mirrored.east
    assert mirrored.west


def test_mirror_diagonal_left_top_right_down_south():
    neighborhood = Neighborhood(False, True, False, False)

    mirrored = neighborhood.mirror(Mirror.DIAGONAL_LEFT_TOP_RIGHT_DOWN)

    assert not mirrored.north
    assert not mirrored.south
    assert mirrored.east
    assert not mirrored.west


def test_mirror_diagonal_left_top_right_down_east():
    neighborhood = Neighborhood(False, False, True, False)

    mirrored = neighborhood.mirror(Mirror.DIAGONAL_LEFT_TOP_RIGHT_DOWN)

    assert not mirrored.north
    assert mirrored.south
    assert not mirrored.east
    assert not mirrored.west


def test_mirror_diagonal_left_top_right_down_west():
    neighborhood = Neighborhood(False, False, False, True)

    mirrored = neighborhood.mirror(Mirror.DIAGONAL_LEFT_TOP_RIGHT_DOWN)

    assert mirrored.north
    assert not mirrored.south
    assert not mirrored.east
    assert not mirrored.west


def test_mirror_rotate_clockwise_1_north():
    neighborhood = Neighborhood(True, False, False, False)

    mirrored = neighborhood.mirror(Mirror.ROT_CLOCKWISE_1)

    assert not mirrored.north
    assert not mirrored.south
    assert mirrored.east
    assert not mirrored.west


def test_mirror_rotate_clockwise_1_south():
    neighborhood = Neighborhood(False, True, False, False)

    mirrored = neighborhood.mirror(Mirror.ROT_CLOCKWISE_1)

    assert not mirrored.north
    assert not mirrored.south
    assert not mirrored.east
    assert mirrored.west


def test_mirror_rotate_clockwise_1_east():
    neighborhood = Neighborhood(False, False, True, False)

    mirrored = neighborhood.mirror(Mirror.ROT_CLOCKWISE_1)

    assert not mirrored.north
    assert mirrored.south
    assert not mirrored.east
    assert not mirrored.west


def test_mirror_rotate_clockwise_1_west():
    neighborhood = Neighborhood(False, False, False, True)

    mirrored = neighborhood.mirror(Mirror.ROT_CLOCKWISE_1)

    assert mirrored.north
    assert not mirrored.south
    assert not mirrored.east
    assert not mirrored.west


def test_mirror_rotate_clockwise_2_north():
    neighborhood = Neighborhood(True, False, False, False)

    mirrored = neighborhood.mirror(Mirror.ROT_CLOCKWISE_2)

    assert not mirrored.north
    assert mirrored.south
    assert not mirrored.east
    assert not mirrored.west


def test_mirror_rotate_clockwise_2_south():
    neighborhood = Neighborhood(False, True, False, False)

    mirrored = neighborhood.mirror(Mirror.ROT_CLOCKWISE_2)

    assert mirrored.north
    assert not mirrored.south
    assert not mirrored.east
    assert not mirrored.west


def test_mirror_rotate_clockwise_2_east():
    neighborhood = Neighborhood(False, False, True, False)

    mirrored = neighborhood.mirror(Mirror.ROT_CLOCKWISE_2)

    assert not mirrored.north
    assert not mirrored.south
    assert not mirrored.east
    assert mirrored.west


def test_mirror_rotate_clockwise_2_west():
    neighborhood = Neighborhood(False, False, False, True)

    mirrored = neighborhood.mirror(Mirror.ROT_CLOCKWISE_2)

    assert not mirrored.north
    assert not mirrored.south
    assert mirrored.east
    assert not mirrored.west


def test_mirror_rotate_clockwise_3_north():
    neighborhood = Neighborhood(True, False, False, False)

    mirrored = neighborhood.mirror(Mirror.ROT_CLOCKWISE_3)

    assert not mirrored.north
    assert not mirrored.south
    assert not mirrored.east
    assert mirrored.west


def test_mirror_rotate_clockwise_3_south():
    neighborhood = Neighborhood(False, True, False, False)

    mirrored = neighborhood.mirror(Mirror.ROT_CLOCKWISE_3)

    assert not mirrored.north
    assert not mirrored.south
    assert mirrored.east
    assert not mirrored.west


def test_mirror_rotate_clockwise_3_east():
    neighborhood = Neighborhood(False, False, True, False)

    mirrored = neighborhood.mirror(Mirror.ROT_CLOCKWISE_3)

    assert mirrored.north
    assert not mirrored.south
    assert not mirrored.east
    assert not mirrored.west


def test_mirror_rotate_clockwise_3_west():
    neighborhood = Neighborhood(False, False, False, True)

    mirrored = neighborhood.mirror(Mirror.ROT_CLOCKWISE_3)

    assert not mirrored.north
    assert mirrored.south
    assert not mirrored.east
    assert not mirrored.west


def test_no_mirror_north():
    neighborhood = Neighborhood(True, False, False, False)

    mirrored = neighborhood.mirror(Mirror.NO_MIRROR)

    assert mirrored.north
    assert not mirrored.south
    assert not mirrored.east
    assert not mirrored.west


def test_no_mirror_south():
    neighborhood = Neighborhood(False, True, False, False)

    mirrored = neighborhood.mirror(Mirror.NO_MIRROR)

    assert not mirrored.north
    assert mirrored.south
    assert not mirrored.east
    assert not mirrored.west


def test_no_mirror_east():
    neighborhood = Neighborhood(False, False, True, False)

    mirrored = neighborhood.mirror(Mirror.NO_MIRROR)

    assert not mirrored.north
    assert not mirrored.south
    assert mirrored.east
    assert not mirrored.west


def test_no_mirror_west():
    neighborhood = Neighborhood(False, False, False, True)

    mirrored = neighborhood.mirror(Mirror.NO_MIRROR)

    assert not mirrored.north
    assert not mirrored.south
    assert not mirrored.east
    assert mirrored.west
