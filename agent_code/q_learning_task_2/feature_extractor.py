from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from agent_code.q_learning_task_2.direction import Direction
from agent_code.q_learning_task_2.feature_vector import FeatureVector, Neighborhood
from agent_code.q_learning_task_2.game_state import GameState
from agent_code.q_learning_task_2.player import Player
from agent_code.q_learning_task_2.types import Position, Bomb


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


def calculate_neighborhood_distance(field: np.ndarray, origin: Position, destinations: List[Position],
                                    bombs: List[Bomb], with_crates: bool = True,
                                    with_bombs: bool = True) -> Neighborhood:
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


def can_move(field: np.ndarray, position: Position) -> Neighborhood:
    neighborhood = Neighborhood()
    for d in Direction:
        name, coords = d.value
        x = position[0] + coords[0]
        y = position[1] + coords[1]
        move_possible = field[x, y] == 0
        setattr(neighborhood, name, move_possible)
    return neighborhood


def is_in_danger(origin: Position, bombs: List[Bomb]):
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


def extract_crates(field: np.ndarray) -> List[Position]:
    crates = np.where(field == 1)
    return list(np.array(crates).T)


def extract_features(state: GameState) -> FeatureVector:
    coin_distance = calculate_neighborhood_distance(state.field, state.self.position, state.coins, state.bombs)

    crates = extract_crates(state.field)

    crate_distance = calculate_neighborhood_distance(state.field, state.self.position, crates, state.bombs)

    can_move_in_direction = can_move(state.field, state.self.position)

    in_danger = is_in_danger(state.self.position, state.bombs)

    return FeatureVector(coin_distance, crate_distance, in_danger, can_move_in_direction)
