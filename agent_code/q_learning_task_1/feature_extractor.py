from __future__ import annotations

from agent_code.common.feature_extractor import calculate_neighborhood_distance, can_move
from agent_code.q_learning_task_1.feature_vector import FeatureVector
from agent_code.common.game_state import GameState


def extract_features(state: GameState) -> FeatureVector:
    coin_distance = calculate_neighborhood_distance(state.field, state.self.position, state.coins, state.bombs)

    can_move_in_direction = can_move(state.field, state.self.position)

    return FeatureVector(coin_distance, can_move_in_direction)
