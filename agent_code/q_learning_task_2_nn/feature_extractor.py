from __future__ import annotations

from agent_code.common.feature_extractor import calculate_neighborhood_distance, extract_crates, can_move, \
    calculate_neighborhood_distance_for_bombs, try_to_move_into_safety, is_in_danger, move_to_danger
from agent_code.common.game_state import GameState
from agent_code.q_learning_task_2_nn.feature_vector import FeatureVector, Neighborhood


def extract_features(state: GameState) -> FeatureVector:
    coin_exists = len(state.coins) > 0
    coin_distance = calculate_neighborhood_distance(state.field, state.self.position, state.coins, state.bombs)

    crates = extract_crates(state.field)

    crate_exists = len(crates) > 0
    crate_distance = calculate_neighborhood_distance(state.field, state.self.position, crates, state.bombs)

    bombs = [(x, y) for ((x, y), _) in state.bombs]

    can_move_in_direction = can_move(state.field, state.self.position, bombs)
    bomb_exists = len(bombs) > 0
    bomb_distance = calculate_neighborhood_distance_for_bombs(state.field, state.self.position, bombs, state.bombs)
    bomb_distance = try_to_move_into_safety(state.field, state.self.position, state.bombs, bomb_distance,
                                            can_move_in_direction)

    in_danger = is_in_danger(state.field, state.self.position, state.bombs)

    if not in_danger:
        mv_to_danger = move_to_danger(state.field, state.self.position, state.bombs, state.explosion_map)
    else:
        mv_to_danger = Neighborhood(False, False, False, False)

    return FeatureVector(coin_distance, coin_exists, crate_distance, crate_exists, in_danger, can_move_in_direction,
                         bomb_distance, bomb_exists, mv_to_danger)
