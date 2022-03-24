import events as e
from agent_code.common.events import APPROACH_COIN, MOVED_AWAY_FROM_COIN, APPROACH_CRATE, MOVED_AWAY_FROM_CRATE, \
    IN_DANGER

rewards = {
    e.COIN_COLLECTED: 25,
    e.MOVED_UP: -2,
    e.MOVED_DOWN: -2,
    e.MOVED_LEFT: -2,
    e.MOVED_RIGHT: -2,
    e.INVALID_ACTION: -10,
    e.WAITED: -15,
    APPROACH_COIN: 9,
    MOVED_AWAY_FROM_COIN: -9,
    APPROACH_CRATE: 7,
    MOVED_AWAY_FROM_CRATE: -7,
    IN_DANGER: -20,
    e.SURVIVED_ROUND: 5,
    e.BOMB_DROPPED: 30,
    e.BOMB_EXPLODED: -10,
    e.COIN_FOUND: 30,
    e.CRATE_DESTROYED: 35,
    e.GOT_KILLED: -200,
    e.KILLED_SELF: -200,
    # e.KILLED_OPPONENT: 20,
    # e.OPPONENT_ELIMINATED: 0,
}
