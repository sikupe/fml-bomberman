import events as e
import os
import json
from agent_code.common.events import MOVED_AWAY_FROM_COIN, APPROACH_COIN, WIGGLE

REWARDS = os.environ.get("REWARDS", "UNSET")

if REWARDS != "UNSET":
    rewards = json.loads(REWARDS)
else:
    rewards = {
        e.COIN_COLLECTED: 25,
        e.MOVED_UP: -2,
        e.MOVED_DOWN: -2,
        e.MOVED_LEFT: -2,
        e.MOVED_RIGHT: -2,
        e.INVALID_ACTION: -7,
        e.WAITED: -10,
        APPROACH_COIN: 5,
        MOVED_AWAY_FROM_COIN: -5,
        WIGGLE: 0,
        e.SURVIVED_ROUND: 5,
        # e.BOMB_DROPPED: 2,
        # e.BOMB_EXPLODED: 2,
        # e.OPPONENT_ELIMINATED: 0,
        # e.COIN_FOUND: 10,
        # e.CRATE_DESTROYED: 3,
        # e.GOT_KILLED: -100,
        # e.KILLED_OPPONENT: 20,
        # e.KILLED_SELF: -100,
    }
