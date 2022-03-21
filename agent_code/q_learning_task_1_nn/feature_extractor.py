from __future__ import annotations

from agent_code.common.feature_extractor import calculate_neighborhood_distance, can_move
from agent_code.common.game_state import GameState
from agent_code.q_learning_task_1_nn.feature_vector import FeatureVector


def extract_features(state: GameState) -> FeatureVector:
    coin_exists = len(state.coins) > 0
    coin_distance = calculate_neighborhood_distance(state.field, state.self.position, state.coins,
                                                    [b[0] for b in state.bombs])

    bombs = [(x, y) for ((x, y), _) in state.bombs]

    can_move_in_direction = can_move(state.field, state.self.position, bombs)

    return FeatureVector(coin_distance, coin_exists, can_move_in_direction)
