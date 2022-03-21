from __future__ import annotations

from agent_code.common.feature_extractor import calculate_neighborhood_distance, extract_crates, can_move, \
    is_in_danger, move_to_danger, \
    next_to_bomb_target, nearest_path_to_safety, find_nearest_crate_approx
from agent_code.common.game_state import GameState
from agent_code.q_learning_task_2_nn.feature_vector import FeatureVector, Neighborhood


def extract_features(state: GameState) -> FeatureVector:
    bombs = [(x, y) for ((x, y), _) in state.bombs]
    can_move_in_direction = can_move(state.field, state.self.position, bombs)

    backup_shortest_path = Neighborhood(1 if can_move_in_direction.north else float('inf'),
                                        1 if can_move_in_direction.south else float('inf'),
                                        1 if can_move_in_direction.east else float('inf'),
                                        1 if can_move_in_direction.west else float('inf'))

    coin_exists = len(state.coins) > 0
    coin_distance = calculate_neighborhood_distance(state.field, state.self.position, state.coins,
                                                    [b[0] for b in state.bombs])

    crates = extract_crates(state.field)
    crate_exists = len(crates) > 0

    crate_distance = backup_shortest_path
    if crate_exists:
        crate_distance = find_nearest_crate_approx(state.field, state.self.position, bombs)

    bombs = [(x, y) for ((x, y), _) in state.bombs]

    useful_bomb = next_to_bomb_target(state.field, state.self.position, state.others)
    current_bomb_drop_escapes = nearest_path_to_safety(state.field, state.explosion_map, state.self.position,
                                                       state.bombs + [(state.self.position, 4)], state.others)
    bomb_drop_safe = current_bomb_drop_escapes.minimum() < float('inf')
    good_bomb = useful_bomb and bomb_drop_safe and state.self.is_bomb_possible

    can_move_in_direction = can_move(state.field, state.self.position, bombs)
    bomb_exists = len(bombs) > 0

    in_danger = is_in_danger(state.field, state.self.position, state.bombs)

    safety = backup_shortest_path
    if bomb_exists and in_danger:
        safety = nearest_path_to_safety(state.field, state.explosion_map, state.self.position, state.bombs,
                                        state.others)

    if not in_danger:
        mv_to_danger = move_to_danger(state.field, state.self.position, state.bombs, state.explosion_map)
    else:
        mv_to_danger = Neighborhood(False, False, False, False)

    bomb_target = next_to_bomb_target(state.field, state.self.position, state.others)

    return FeatureVector(coin_distance, coin_exists, crate_distance, crate_exists, in_danger, can_move_in_direction,
                         safety, bomb_exists, mv_to_danger, bomb_target, good_bomb)
