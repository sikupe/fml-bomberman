from __future__ import annotations

import numpy as np

from agent_code.common.feature_extractor import calculate_neighborhood_distance, can_move, nearest_path_to_safety, \
    move_to_danger, is_in_danger, find_nearest_crate_approx, next_to_bomb_target
from agent_code.common.game_state import GameState
from agent_code.q_learning_task_1_nn_evolution.feature_vector import FeatureVector


def extract_features(state: GameState) -> FeatureVector:
    coin_exists = len(state.coins) > 0
    coin_distance = calculate_neighborhood_distance(state.field, state.self.position, state.coins,
                                                    [b[0] for b in state.bombs])

    bombs = [(x, y) for ((x, y), _) in state.bombs]

    can_move_in_direction = can_move(state.field, state.self.position, bombs)

    safety = nearest_path_to_safety(state.field, state.explosion_map, state.self.position, state.bombs,
                                    state.others)

    danger_neighborhood = move_to_danger(state.field, state.self.position, state.bombs, state.explosion_map)

    bomb_distance = calculate_neighborhood_distance(state.field, state.self.position, bombs, [])

    in_danger = is_in_danger(state.field, state.self.position, state.bombs)

    crate_distance = find_nearest_crate_approx(state.field, state.self.position, bombs)

    useful_bomb = next_to_bomb_target(state.field, state.self.position, state.others)
    current_bomb_drop_escapes = nearest_path_to_safety(state.field, state.explosion_map, state.self.position,
                                                       state.bombs + [(state.self.position, 4)], state.others)
    bomb_drop_safe = current_bomb_drop_escapes.minimum() < float('inf')
    good_bomb = useful_bomb and bomb_drop_safe and state.self.is_bomb_possible

    crate_exists = np.any(state.field == 1)

    return FeatureVector(coin_distance, coin_exists, can_move_in_direction, safety, danger_neighborhood, in_danger,
                         bomb_distance, crate_distance, good_bomb, crate_exists)
