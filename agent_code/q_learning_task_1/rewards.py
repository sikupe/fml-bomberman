import events as e

APPROACH_COIN = 'APPROACH_COIN'
MOVED_AWAY_FROM_COIN = 'MOVED_AWAY_FROM_COIN'
DID_NOT_PICK_UP_COIN = 'DID_NOT_PICK_UP_COIN'

rewards = {
    e.COIN_COLLECTED: 10,
    e.MOVED_UP: -1,
    e.MOVED_DOWN: -1,
    e.MOVED_LEFT: -1,
    e.MOVED_RIGHT: -1,
    e.INVALID_ACTION: -10,
    e.WAITED: -3,
    APPROACH_COIN: 3,
    MOVED_AWAY_FROM_COIN: -2,
    DID_NOT_PICK_UP_COIN: -10,
    e.BOMB_DROPPED: 2,
    e.BOMB_EXPLODED: 2,
    e.SURVIVED_ROUND: 10,
    e.OPPONENT_ELIMINATED: 0,
    e.COIN_FOUND: 10,
    e.CRATE_DESTROYED: 3,
    e.GOT_KILLED: -100,
    e.KILLED_OPPONENT: 20,
    e.KILLED_SELF: -100,
}
