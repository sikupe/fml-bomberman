import events
from agent_code.common.neighborhood import Mirror


def test_moved_up():
    event = events.MOVED_UP

    mirrored = Mirror.mirror_events(Mirror.NO_MIRROR, event)

    assert event == mirrored


def test_moved_down():
    event = events.MOVED_DOWN

    mirrored = Mirror.mirror_events(Mirror.NO_MIRROR, event)

    assert event == mirrored


def test_moved_left():
    event = events.MOVED_LEFT

    mirrored = Mirror.mirror_events(Mirror.NO_MIRROR, event)

    assert event == mirrored


def test_moved_right():
    event = events.MOVED_RIGHT

    mirrored = Mirror.mirror_events(Mirror.NO_MIRROR, event)

    assert event == mirrored


def test_mirror_list():
    es = [events.MOVED_UP, events.MOVED_RIGHT, events.KILLED_SELF]

    mirrored = Mirror.mirror_events(Mirror.ROT_CLOCKWISE_1, es)

    assert mirrored[0] == events.MOVED_RIGHT and mirrored[1] == events.MOVED_DOWN and mirrored[2] == events.KILLED_SELF
