import events as e

APPROACH_COIN = 'APPROACH_COIN'
MOVED_AWAY_FROM_COIN = 'MOVED_AWAY_FROM_COIN'

APPROACH_CRATE = 'APPROACH_CRATE'
MOVED_AWAY_FROM_CRATE = 'MOVED_AWAY_FROM_CRATE'

APPROACH_BOMB = 'APPROACH_BOMB'
MOVED_AWAY_FROM_BOMB = 'MOVED_AWAY_FROM_BOMB'

IN_DANGER = 'IN_DANGER'
MOVE_IN_DANGER = 'MOVE_IN_DANGER'

GOOD_BOMB = 'GOOD_BOMB'
BAD_BOMB = 'BAD_BOMB'

MOVED_TO_SUICIDE = 'MOVED_TO_SUICIDE'

APPROACH_SECURITY = 'APPROACH_SECURITY'
MOVED_AWAY_FROM_SECURITY = 'MOVED_AWAY_FROM_SECURITY'

rewards = {
    e.COIN_COLLECTED: 25,
    e.MOVED_UP: -3,
    e.MOVED_DOWN: -3,
    e.MOVED_LEFT: -3,
    e.MOVED_RIGHT: -3,
    e.INVALID_ACTION: -20,
    e.WAITED: -5,
    APPROACH_COIN: 15,
    MOVED_AWAY_FROM_COIN: -15,
    APPROACH_CRATE: 10,
    MOVED_AWAY_FROM_CRATE: -10,
    APPROACH_BOMB: -15,
    MOVED_AWAY_FROM_BOMB: 15,
    IN_DANGER: -20,
    MOVE_IN_DANGER: -35,
    GOOD_BOMB: 20,
    BAD_BOMB: -50,
    APPROACH_SECURITY: 15,
    MOVED_AWAY_FROM_SECURITY: -15,
    MOVED_TO_SUICIDE: -30,
    e.SURVIVED_ROUND: 5,
    e.BOMB_DROPPED: -100,
    e.BOMB_EXPLODED: 0,
    e.COIN_FOUND: 30,
    e.CRATE_DESTROYED: 35,
    e.GOT_KILLED: -200,
    e.KILLED_SELF: -200,
    # e.KILLED_OPPONENT: 20,
    # e.OPPONENT_ELIMINATED: 0,
}