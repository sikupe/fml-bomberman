import events as e
from agent_code.common.events import APPROACH_COIN, MOVED_AWAY_FROM_COIN, WIGGLE

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
}
