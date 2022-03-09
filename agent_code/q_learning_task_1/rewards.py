import events as e

APPROACH_COIN = 'APPROACH_COIN'
MOVED_AWAY_FROM_COIN = 'MOVED_AWAY_FROM_COIN'

rewards = {
    e.COIN_COLLECTED: 20,
    e.MOVED_UP: -1,
    e.MOVED_DOWN: -1,
    e.MOVED_LEFT: -1,
    e.MOVED_RIGHT: -1,
    e.INVALID_ACTION: -10,
    e.WAITED: -3,
    APPROACH_COIN: 2,
    MOVED_AWAY_FROM_COIN: -2,
    e.BOMB_DROPPED: 2,
    e.BOMB_EXPLODED: 2,
    e.SURVIVED_ROUND: 10,
    e.OPPONENT_ELIMINATED: 0,
    e.COIN_FOUND: 5,
    e.CRATE_DESTROYED: 3,
    e.GOT_KILLED: -100,
    e.KILLED_OPPONENT: 20,
    e.KILLED_SELF: -100,
}
