from typing import List

from agent_code.common.feature_vector import BaseFeatureVector
from agent_code.common.train import detect_wiggle

IN_DANGER = 'IN_DANGER'
MOVE_IN_DANGER = 'MOVE_IN_DANGER'
MOVE_OUT_OF_DANGER = 'MOVE_OUT_OF_DANGER'

GOOD_BOMB = 'GOOD_BOMB'
BAD_BOMB = 'BAD_BOMB'

WIGGLE = 'WIGGLE'

APPROACH_SECURITY = 'APPROACH_SECURITY'
MOVED_AWAY_FROM_SECURITY = 'MOVED_AWAY_FROM_SECURITY'

APPROACH_COIN = 'APPROACH_COIN'
MOVED_AWAY_FROM_COIN = 'MOVED_AWAY_FROM_COIN'

APPROACH_CRATE = 'APPROACH_CRATE'
MOVED_AWAY_FROM_CRATE = 'MOVED_AWAY_FROM_CRATE'

APPROACH_OPPONENT = 'APPROACH_OPPONENT'
MOVED_AWAY_FROM_OPPONENT = 'MOVED_AWAY_FROM_OPPONENT'


def extract_events_from_state(self, old_features: BaseFeatureVector, new_features: BaseFeatureVector,
                              action: str) -> List[str]:
    custom_events = []

    # Coins
    if old_features.coin_distance.exists and new_features.coin_distance.exists:
        if old_features.coin_distance.minimum() <= new_features.coin_distance.minimum():
            custom_events.append(MOVED_AWAY_FROM_COIN)
        elif old_features.coin_distance.minimum() > new_features.coin_distance.minimum():
            custom_events.append(APPROACH_COIN)

    # Crates
    if old_features.crate_distance.exists and new_features.crate_distance.exists:
        if old_features.crate_distance.minimum() <= new_features.crate_distance.minimum():
            custom_events.append(MOVED_AWAY_FROM_CRATE)
        elif old_features.crate_distance.minimum() > new_features.crate_distance.minimum():
            custom_events.append(APPROACH_CRATE)

    # Opponents
    if old_features.opponent_distance.exists and new_features.opponent_distance.exists:
        if old_features.opponent_distance.minimum() <= new_features.opponent_distance.minimum():
            custom_events.append(MOVED_AWAY_FROM_OPPONENT)
        elif old_features.opponent_distance.minimum() > new_features.opponent_distance.minimum():
            custom_events.append(APPROACH_OPPONENT)

    # In danger
    in_danger_current = old_features.in_danger
    in_danger_next = new_features.in_danger
    if in_danger_next:
        custom_events.append(IN_DANGER)

    if not in_danger_current and in_danger_next:
        custom_events.append(MOVE_IN_DANGER)

    if in_danger_current and not in_danger_next:
        custom_events.append(MOVE_OUT_OF_DANGER)

    # Security
    if in_danger_current:
        if old_features.shortest_path_to_safety.minimum() <= new_features.shortest_path_to_safety.minimum():
            custom_events.append(MOVED_AWAY_FROM_SECURITY)
        if old_features.shortest_path_to_safety.minimum() > new_features.shortest_path_to_safety.minimum():
            custom_events.append(APPROACH_SECURITY)

    # Bombs
    if action == "BOMB":
        if old_features.good_bomb:
            custom_events.append(GOOD_BOMB)
        else:
            custom_events.append(BAD_BOMB)

    return custom_events + [WIGGLE for _ in range(detect_wiggle(self.transitions))]
