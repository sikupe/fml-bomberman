from __future__ import annotations

from typing import Dict, List

import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from agent_code.strong_students.direction import Direction
from agent_code.strong_students.player import Player
from agent_code.strong_students.types import Position, Bomb
from agent_code.strong_students.feature_vector import FeatureVector, Neighborhood
from agent_code.strong_students.game_state import GameState


def convert_to_state_object(state: Dict) -> GameState:
    rnd: int = state["round"]
    step: int = state["step"]
    field: np.ndarray = state["field"]
    bombs: List[Bomb] = state["bombs"]
    explosion_map: np.ndarray = state["explosion_map"]
    coins: List[Position] = state["coins"]
    self: Player = Player(state["self"])
    others: List[Player] = list(map(Player, state["others"]))
    user_input: str | None = state["user_input"]
    return GameState(
        rnd, step, field, bombs, explosion_map, coins, self, others, user_input
    )


def calculate_neighborhood_distance(
    field: np.ndarray, origin: Position, destinations: List[Position], bombs: List[Bomb]
) -> Neighborhood:
    field_without_obstacles: np.ndarray = field.copy()
    field_with_obstacles = field.copy()
    field_with_obstacles[field < 0] = 1
    field_without_obstacles[field < 0] = 0

    for bomb_coord, time in bombs:
        field_with_obstacles[bomb_coord] = 1

    neighborhood = Neighborhood()
    grid = Grid(matrix=field_with_obstacles)
    finder = AStarFinder()
    for d in Direction:
        name, coords = d.value
        shortest_path = float("inf")

        for dest in destinations:
            x = origin[0] + coords[0]
            y = origin[1] + coords[1]
            start = grid.node(x, y)
            end = grid.node(dest[0], dest[1])
            path, runs = finder.find_path(start, end, grid)

            if len(path) < shortest_path:
                shortest_path = len(path)

        setattr(neighborhood, name, shortest_path)

    return neighborhood


def extract_crates(field: np.ndarray):
    crates = np.where(field == -1)
    return list(crates)


def can_move(field: np.ndarray, position: Position) -> Neighborhood:
    neighborhood = Neighborhood()
    for d in Direction:
        name, coords = d.value
        x = position[0] + coords[0]
        y = position[1] + coords[1]
        move_possible = field[x, y] == 0
        setattr(neighborhood, name, move_possible)
    return neighborhood


def is_in_blast_radius(
    field: np.ndarray, position: Position, bombs: List[Bomb]
) -> tuple[Neighborhood, bool]:
    neighborhood = Neighborhood()
    for d in Direction:
        name, coord = d.value
        can_burn = is_in_line_with_bomb(bombs, coord, field, position)
        setattr(neighborhood, name, can_burn)

    currently = is_in_line_with_bomb(bombs, (0, 0), field, position)
    return neighborhood, currently


def is_in_line_with_bomb(bombs, coord, field, position):
    can_burn = False
    for bomb_coord, _ in bombs:
        x_start = position[0] + coord[0]
        y_start = position[1] + coord[1]

        x_end = bomb_coord[0]
        y_end = bomb_coord[1]
        if x_start == x_end or y_start == y_end:
            line = field[x_start:x_end, y_start:y_end] != 0
            obstacles = np.sum(line) > 0
            if not obstacles:
                can_burn = True
                break
    return can_burn


def extract_features(state_dict: Dict) -> FeatureVector:
    state = convert_to_state_object(state_dict)

    opponent_distance = calculate_neighborhood_distance(
        state.field,
        state.self.position,
        list(map(lambda x: x.position, state.others)),
        state.bombs,
    )
    coin_distance = calculate_neighborhood_distance(
        state.field, state.self.position, state.coins, state.bombs
    )
    bomb_distance = calculate_neighborhood_distance(
        state.field, state.self.position, list(map(lambda x: x[0], state.bombs)), []
    )

    crates = extract_crates(state.field)

    crates_distance = calculate_neighborhood_distance(
        state.field, state.self.position, crates, state.bombs
    )

    can_move_in_direction = can_move(state.field, state.self.position)

    in_blast_radius, in_blast_radius_currently = is_in_blast_radius(
        state.field, state.self.position, state.bombs
    )

    n_opponents = len(state.others)

    return FeatureVector(
        opponent_distance,
        coin_distance,
        bomb_distance,
        crates_distance,
        can_move_in_direction,
        in_blast_radius,
        in_blast_radius_currently,
        n_opponents,
    )
