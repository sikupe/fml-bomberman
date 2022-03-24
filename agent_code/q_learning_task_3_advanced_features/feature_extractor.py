from __future__ import annotations

from agent_code.common.feature_extractor import calculate_neighborhood_distance, extract_crates, can_move, \
    is_in_danger, move_to_danger, next_to_bomb_target, nearest_path_to_safety, find_nearest_crate_approx
from agent_code.common.game_state import GameState
from agent_code.q_learning_task_3_advanced_features.feature_vector import BaseFeatureVector, Neighborhood


def extract_features(state: GameState) -> BaseFeatureVector:
    bombs = [(x, y) for ((x, y), _) in state.bombs]
    can_move_in_direction = can_move(state.field, state.self.position, bombs)

    backup_shortest_path = Neighborhood(1 if can_move_in_direction.north else float('inf'),
                                        1 if can_move_in_direction.south else float('inf'),
                                        1 if can_move_in_direction.east else float('inf'),
                                        1 if can_move_in_direction.west else float('inf'))
    backup_shortest_path.exists = False

    # Coins
    coin_exists = len(state.coins) > 0

    coin_distance = backup_shortest_path
    if coin_exists:
        coin_distance = calculate_neighborhood_distance(
            state.field, state.self.position, state.coins, state.bombs
        )
        coin_distance.exists = True

    # Crates
    crates = extract_crates(state.field)
    crate_exists = len(crates) > 0

    crate_distance = backup_shortest_path
    if crate_exists:
        crate_distance = find_nearest_crate_approx(state.field, state.self.position, bombs)
        crate_distance.exists = True

    # Bombs
    bombs = [(x, y) for ((x, y), _) in state.bombs]
    bomb_exists = len(bombs) > 0

    # Danger
    in_danger = is_in_danger(state.field, state.self.position, state.bombs)

    if not in_danger:
        mv_to_danger = move_to_danger(
            state.field, state.self.position, state.bombs, state.explosion_map
        )
        # We do not set mv_to_danger.exists because it is encoded with 16
        # states [0, 0, 0, 0] already contains the information that there is no
        # move to danger
    else:
        mv_to_danger = Neighborhood(False, False, False, False)

    useful_bomb = next_to_bomb_target(state.field, state.self.position, state.others)
    current_bomb_drop_escapes = nearest_path_to_safety(state.field, state.explosion_map, state.self.position,
                                                       state.bombs + [(state.self.position, 4)], state.others)
    bomb_drop_safe = current_bomb_drop_escapes.minimum() < float('inf')
    good_bomb = useful_bomb and bomb_drop_safe and state.self.is_bomb_possible

    safety = backup_shortest_path
    if in_danger:
        safety = nearest_path_to_safety(state.field, state.explosion_map, state.self.position, state.bombs,
                                        state.others)
        safety.exists = True  # We are in danger

    has_opponents = len(state.others) > 0

    opponent_dest = [o.position for o in state.others]
    opponent_distance = calculate_neighborhood_distance(state.field, state.self.position, opponent_dest,
                                                        [b[0] for b in state.bombs])

    return BaseFeatureVector(
        coin_distance,
        crate_distance,
        in_danger,
        mv_to_danger,
        bomb_drop_safe,
        good_bomb,
        safety,
        can_move_in_direction,
        opponent_distance,
        has_opponents
    )
