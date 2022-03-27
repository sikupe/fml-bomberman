import events as e
import json
import os

from agent_code.strong_students.common.events import MOVE_IN_DANGER, MOVE_OUT_OF_DANGER, GOOD_BOMB, BAD_BOMB, IN_DANGER, \
    APPROACH_SECURITY, MOVED_AWAY_FROM_SECURITY, APPROACH_COIN, MOVED_AWAY_FROM_COIN, APPROACH_CRATE, \
    MOVED_AWAY_FROM_CRATE, APPROACH_OPPONENT, MOVED_AWAY_FROM_OPPONENT, WIGGLE

REWARDS = os.environ.get("REWARDS", "UNSET")

if REWARDS != "UNSET":
    rewards = json.loads(REWARDS)
else:
    rewards = {
        IN_DANGER: -20,
        MOVE_IN_DANGER: -35,
        MOVE_OUT_OF_DANGER: 35,
        GOOD_BOMB: 45,
        BAD_BOMB: -40,
        APPROACH_SECURITY: 35,
        MOVED_AWAY_FROM_SECURITY: -35,
        APPROACH_COIN: 9,
        MOVED_AWAY_FROM_COIN: -9,
        APPROACH_CRATE: 9,
        MOVED_AWAY_FROM_CRATE: -9,
        APPROACH_OPPONENT: 7,
        MOVED_AWAY_FROM_OPPONENT: -7,
        WIGGLE: -2,
        e.COIN_COLLECTED: 25,
        e.MOVED_UP: -3,
        e.MOVED_DOWN: -3,
        e.MOVED_LEFT: -3,
        e.MOVED_RIGHT: -3,
        e.INVALID_ACTION: -60,
        e.WAITED: -15,
        e.SURVIVED_ROUND: 5,
        e.BOMB_DROPPED: 10,
        e.BOMB_EXPLODED: 0,
        e.COIN_FOUND: 30,
        e.CRATE_DESTROYED: 40,
        e.GOT_KILLED: -200,
        e.KILLED_SELF: -200,
        e.KILLED_OPPONENT: 50,
        e.OPPONENT_ELIMINATED: 0,
    }
