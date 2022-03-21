from __future__ import annotations

from agent_code.common.feature_extractor import calculate_neighborhood_distance, extract_crates, can_move, \
    calculate_neighborhood_distance_for_bombs, try_to_move_into_safety, is_in_danger, move_to_danger, \
    next_to_bomb_target, nearest_path_to_safety
from agent_code.common.game_state import GameState
from agent_code.q_learning_task_3.feature_vector import FeatureVector, Neighborhood


def extract_features(state: GameState) -> FeatureVector:
    # Coins
    coin_exists = len(state.coins) > 0

    coin_distance = Neighborhood(0, 0, 0, 0)
    if coin_exists:
        coin_distance = calculate_neighborhood_distance(
            state.field, state.self.position, state.coins, state.bombs
        )

    # Crates
    crates = extract_crates(state.field)
    crate_exists = len(crates) > 0

    crate_distance = Neighborhood(0, 0, 0, 0)
    if crate_exists:
        crate_distance = calculate_neighborhood_distance(
            state.field, state.self.position, crates, state.bombs, with_crates=False
        )

    # Bombs
    bombs = [(x, y) for ((x, y), _) in state.bombs]
    bomb_exists = len(bombs) > 0

    can_move_in_direction = can_move(state.field, state.self.position, bombs)

    bomb_distance = Neighborhood(0, 0, 0, 0)
    if bomb_exists:
        bomb_distance = calculate_neighborhood_distance_for_bombs(
            state.field, state.self.position, bombs, state.bombs
        )
        bomb_distance = try_to_move_into_safety(
            state.field, state.self.position, state.bombs, bomb_distance, can_move_in_direction
        )

    # Danger
    in_danger = is_in_danger(state.field, state.self.position, state.bombs)

    if not in_danger:
        mv_to_danger = move_to_danger(
            state.field, state.self.position, state.bombs, state.explosion_map
        )
    else:
        mv_to_danger = Neighborhood(False, False, False, False)

    useful_bomb = next_to_bomb_target(state.field, state.self.position, state.others)
    current_bomb_drop_escapes = nearest_path_to_safety(state.field, state.explosion_map, state.self.position,
                                                       state.bombs + [(state.self.position, 4)])
    bomb_drop_safe = current_bomb_drop_escapes.minimum() < float('inf')
    good_bomb = useful_bomb and bomb_drop_safe and state.self.is_bomb_possible

    if in_danger:
        safety = nearest_path_to_safety(state.field, state.explosion_map, state.self.position, state.bombs)
    else:
        safety = Neighborhood()

    has_opponents = len(state.others) > 0

    opponent_dest = [o.position for o in state.others]
    opponent_distance = calculate_neighborhood_distance(state.field, state.self.position, opponent_dest, state.bombs)

    return FeatureVector(
        coin_distance,
        coin_exists,
        crate_distance,
        crate_exists,
        in_danger,
        bomb_distance,
        bomb_exists,
        mv_to_danger,
        bomb_drop_safe,
        good_bomb,
        safety,
        can_move_in_direction,
        opponent_distance,
        has_opponents
    )
