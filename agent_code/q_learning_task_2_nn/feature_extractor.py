from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from agent_code.q_learning_task_2_nn.direction import Direction
from agent_code.q_learning_task_2_nn.feature_vector import FeatureVector, Neighborhood
from agent_code.q_learning_task_2_nn.game_state import GameState
from agent_code.q_learning_task_2_nn.player import Player
from agent_code.q_learning_task_2_nn.types import Position, Bomb


def convert_to_state_object(state: Dict) -> GameState:
    rnd: int = state['round']
    step: int = state['step']
    field: np.ndarray = state['field']
    bombs: List[Bomb] = state['bombs']
    explosion_map: np.ndarray = state['explosion_map']
    coins: List[Position] = state['coins']
    self: Player = Player(state['self'])
    others: List[Player] = list(map(Player, state['others']))
    user_input: str | None = state['user_input']
    return GameState(rnd, step, field.T, bombs, explosion_map, coins, self, others, user_input)


def calculate_neighborhood_distance(
    field: np.ndarray,
    origin: Position,
    destinations: List[Position],
    bombs: List[Bomb],
    with_crates: bool = True,
    with_bombs: bool = True,
) -> Neighborhood:

    field: np.ndarray = field.copy()

    # Make creates to obstacles for pathfinding
    field[field > 0] = -1
    # Make free fields to pathfinding free fields
    field[field == 0] = 1

    if with_bombs:
        for bomb_coord, time in bombs:
            field[bomb_coord] = 1

    neighborhood = Neighborhood()
    grid = Grid(matrix=field)
    finder = AStarFinder()

    for d in Direction:
        name, coords = d.value
        shortest_path = float('inf')

        for dest in destinations:
            x = origin[0] + coords[0]
            y = origin[1] + coords[1]
            if field[x][y] > 0:
                start = grid.node(x, y)
                end = grid.node(dest[0], dest[1])

                path, runs = finder.find_path(start, end, grid)
                grid.cleanup()

                if len(path) < shortest_path:
                    shortest_path = len(path)

            else:
                break

        setattr(neighborhood, name, shortest_path)

    return neighborhood


def calculate_neighborhood_distance_for_bombs(
    field: np.ndarray,
    origin: Position,
    destinations: List[Position],
    bombs: List[Bomb],
) -> Neighborhood:

    field: np.ndarray = field.copy()

    # Make creates to obstacles for pathfinding
    field[field > 0] = -1
    # Make free fields to pathfinding free fields
    field[field == 0] = 1

    for bomb_coord, time in bombs:
        field[bomb_coord] = 1

    neighborhood = Neighborhood()
    grid = Grid(matrix=field)
    finder = AStarFinder()

    for d in Direction:
        name, coords = d.value
        longest_path = 0

        for bomb in destinations:
            if (origin[0], origin[1]) == (bomb[0], bomb[1]):
                return Neighborhood(1, 1, 1, 1)
            x = origin[0] + coords[0]
            y = origin[1] + coords[1]
            if field[x][y] > 0:
                start = grid.node(x, y)
                end = grid.node(bomb[0], bomb[1])

                path, runs = finder.find_path(start, end, grid)
                grid.cleanup()

                # if len(path) > longest_path:
                #     longest_path = len(path)
                longest_path += len(path)

            else:
                continue

        setattr(neighborhood, name, longest_path)

    return neighborhood


def can_move(field: np.ndarray, position: Position) -> Neighborhood:
    neighborhood = Neighborhood()
    for d in Direction:
        name, coords = d.value
        x = position[0] + coords[0]
        y = position[1] + coords[1]
        move_possible = field[x, y] == 0
        setattr(neighborhood, name, move_possible)
    return neighborhood


def try_to_move_into_safety(origin: Position, bombs: List[Bomb], bomb_distance: Neighborhood, can_move_in_direction: Neighborhood) -> Neighborhood:
    """
    Go into each direction and check if it safe.

    We check for each direction if we would still be in danger.
    To see if something is really safe we also check if we can move towards it.
    Still existing valid paths are keept, invalid paths (not movable, and not
    safe) are set to -1.
    """

    safety = []
    for d in Direction:
        _, coords = d.value
        new_origin = (origin[0] + coords[0], origin[1] + coords[1])
        safety.append(not is_in_danger(new_origin, bombs))

    for i, move in enumerate(can_move_in_direction):
        safety[i] = move and safety[i]

    if True in safety:
        for (i, d) in enumerate(Direction):
            if not safety[i]:
                setattr(bomb_distance, d.value[0], 0)

    return bomb_distance


def is_in_danger(origin: Position, bombs: List[Bomb]) -> bool:
    in_danger = False

    for bomb_coords, exp_time in bombs:
        for i in range(2):
            if origin[i] == bomb_coords[i]:
                other_i = (i + 1) % 2
                dist = abs(origin[other_i] - bomb_coords[other_i])
                if dist < 4:
                    in_danger = True
                    break

    return in_danger


def move_to_danger(field: np.ndarray, origin: Position, bombs: List[Bomb], explosion_map: np.ndarray):
    neighborhood = Neighborhood()
    for d in Direction:
        name, coords = d.value
        x = origin[0] + coords[0]
        y = origin[1] + coords[1]

        in_danger = False

        if field[x, y] == 0:
            for bomb_coords, _ in bombs:
                if x == bomb_coords[0] or y == bomb_coords[1]:
                    dist = max(abs(x - bomb_coords[0]), abs(y - bomb_coords[1]))

                    if dist <= 4:
                        in_danger = True
                        break

            if not in_danger:
                if explosion_map[x, y] > 0:
                    in_danger = True

        setattr(neighborhood, name, in_danger)

    return neighborhood


def extract_crates(field: np.ndarray) -> List[Position]:
    crates = np.where(field == 1)
    return list(np.array(crates).T)


def extract_features(state: GameState) -> FeatureVector:

    coin_exists = len(state.coins) > 0
    coin_distance = calculate_neighborhood_distance(state.field, state.self.position, state.coins, state.bombs)

    crates = extract_crates(state.field)

    crate_exists = len(crates) > 0
    crate_distance = calculate_neighborhood_distance(state.field, state.self.position, crates, state.bombs)

    can_move_in_direction = can_move(state.field, state.self.position)

    bombs = [(x, y) for ((x, y), _) in state.bombs]
    bomb_exists = len(bombs) > 0
    bomb_distance = calculate_neighborhood_distance_for_bombs(state.field, state.self.position, bombs, state.bombs)
    bomb_distance = try_to_move_into_safety(state.self.position, state.bombs, bomb_distance, can_move_in_direction)

    in_danger = is_in_danger(state.self.position, state.bombs)

    if not in_danger:
        mv_to_danger = move_to_danger(state.field, state.self.position, state.bombs, state.explosion_map)
    else:
        mv_to_danger = Neighborhood(False, False, False, False)

    return FeatureVector(coin_distance, coin_exists, crate_distance, crate_exists, in_danger, can_move_in_direction, bomb_distance, bomb_exists, mv_to_danger)
