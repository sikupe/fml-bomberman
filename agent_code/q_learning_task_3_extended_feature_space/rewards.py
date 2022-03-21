import events as e

APPROACH_USEFUL = 'APPROACH_USEFUL'
MOVED_AWAY_FROM_USEFUL = 'MOVED_AWAY_FROM_USEFUL'

IN_DANGER = 'IN_DANGER'
MOVE_IN_DANGER = 'MOVE_IN_DANGER'

GOOD_BOMB = 'GOOD_BOMB'
BAD_BOMB = 'BAD_BOMB'

MOVED_TO_SUICIDE = 'MOVED_TO_SUICIDE'

APPROACH_SECURITY = 'APPROACH_SECURITY'
MOVED_AWAY_FROM_SECURITY = 'MOVED_AWAY_FROM_SECURITY'

APPROACH_COIN = 'APPROACH_COIN'
MOVED_AWAY_FROM_COIN = 'MOVED_AWAY_FROM_COIN'

APPROACH_CRATE = 'APPROACH_CRATE'
MOVED_AWAY_FROM_CRATE = 'MOVED_AWAY_FROM_CRATE'

APPROACH_OPPONENT = 'APPROACH_OPPONENT'
MOVED_AWAY_FROM_OPPONENT = 'MOVED_AWAY_FROM_OPPONENT'

rewards = {
    e.COIN_COLLECTED: 25,
    e.MOVED_UP: -3,
    e.MOVED_DOWN: -3,
    e.MOVED_LEFT: -3,
    e.MOVED_RIGHT: -3,
    e.INVALID_ACTION: -40,
    e.WAITED: -20,
    # APPROACH_USEFUL: 8,
    # MOVED_AWAY_FROM_USEFUL: -8,
    APPROACH_COIN: 14,
    MOVED_AWAY_FROM_COIN: -14,
    APPROACH_CRATE: 6,
    MOVED_AWAY_FROM_CRATE: -6,
    APPROACH_OPPONENT: 10,
    MOVED_AWAY_FROM_OPPONENT: -10,
    IN_DANGER: -20,
    MOVE_IN_DANGER: -35,
    GOOD_BOMB: 15,
    BAD_BOMB: -50,
    APPROACH_SECURITY: 15,
    MOVED_AWAY_FROM_SECURITY: -15,
    MOVED_TO_SUICIDE: -30,
    e.SURVIVED_ROUND: 5,
    e.BOMB_DROPPED: -2,
    e.BOMB_EXPLODED: 0,
    e.COIN_FOUND: 30,
    e.CRATE_DESTROYED: 40,
    e.GOT_KILLED: -200,
    e.KILLED_SELF: -200,
    e.KILLED_OPPONENT: 50,
    e.OPPONENT_ELIMINATED: 0,
}
