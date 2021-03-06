import events as e
from agent_code.common.events import APPROACH_COIN, MOVED_AWAY_FROM_COIN, APPROACH_CRATE, MOVED_AWAY_FROM_CRATE, \
    IN_DANGER, MOVE_IN_DANGER, GOOD_BOMB, BAD_BOMB, APPROACH_SECURITY, MOVED_AWAY_FROM_SECURITY, WIGGLE

rewards = {
    e.COIN_COLLECTED: 25,
    e.MOVED_UP: -3,
    e.MOVED_DOWN: -3,
    e.MOVED_LEFT: -3,
    e.MOVED_RIGHT: -3,
    e.INVALID_ACTION: -10,
    e.WAITED: -10,
    APPROACH_COIN: 8,
    MOVED_AWAY_FROM_COIN: -8,
    APPROACH_CRATE: 10,
    MOVED_AWAY_FROM_CRATE: -10,
    IN_DANGER: -20,
    MOVE_IN_DANGER: -35,
    GOOD_BOMB: 15,
    BAD_BOMB: -50,
    APPROACH_SECURITY: 15,
    MOVED_AWAY_FROM_SECURITY: -15,
    WIGGLE: 0,
    e.SURVIVED_ROUND: 5,
    e.BOMB_DROPPED: -2,
    e.BOMB_EXPLODED: 0,
    e.COIN_FOUND: 30,
    e.CRATE_DESTROYED: 35,
    e.GOT_KILLED: -200,
    e.KILLED_SELF: -200,
    # e.KILLED_OPPONENT: 20,
    # e.OPPONENT_ELIMINATED: 0,
}
