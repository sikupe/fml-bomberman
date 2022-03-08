from typing import Dict

from game_state import GameState


def convert_to_state_object(state: Dict):
    return GameState(state)


def extract_features(state: Dict):
    state = convert_to_state_object(state)
