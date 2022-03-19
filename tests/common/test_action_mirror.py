from agent_code.common.neighborhood import Mirror


def test_up():
    action = 'UP'

    mirrored = Mirror.mirror_action(Mirror.NO_MIRROR, action)

    assert action == mirrored


def test_down():
    action = 'DOWN'

    mirrored = Mirror.mirror_action(Mirror.NO_MIRROR, action)

    assert action == mirrored


def test_left():
    action = 'LEFT'

    mirrored = Mirror.mirror_action(Mirror.NO_MIRROR, action)

    assert action == mirrored


def test_right():
    action = 'RIGHT'

    mirrored = Mirror.mirror_action(Mirror.NO_MIRROR, action)

    assert action == mirrored


def test_mirror_clockwise():
    UP = 'UP'
    DOWN = 'DOWN'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'

    assert RIGHT == Mirror.mirror_action(Mirror.ROT_CLOCKWISE_1, UP)
    assert DOWN == Mirror.mirror_action(Mirror.ROT_CLOCKWISE_1, RIGHT)
    assert LEFT == Mirror.mirror_action(Mirror.ROT_CLOCKWISE_1, DOWN)
    assert UP == Mirror.mirror_action(Mirror.ROT_CLOCKWISE_1, LEFT)
