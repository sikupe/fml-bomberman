from __future__ import annotations

from agent_code.strong_students.feature_vector import FeatureVector, Neighborhood
from agent_code.strong_students.direction import Direction
from agent_code.strong_students.game_state import GameState
from agent_code.strong_students.neighborhood import Neighborhood
from agent_code.strong_students.player import Player
from agent_code.strong_students.types import Position, Bomb

from typing import Dict, List, Set

import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

import items
import settings



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
        field: np.ndarray,
        origin: Position,
        destinations: List[Position],
        obstacles: List[Position],
        with_crates: bool = True,
        with_bombs: bool = True,
) -> Neighborhood:
    field: np.ndarray = field.copy()

    # Make creates to obstacles for pathfinding
    if with_crates:
        field[field > 0] = -1
    # Make free fields to pathfinding free fields
    field[field == 0] = 1

    if with_bombs:
        for bomb_coord in obstacles:
            field[bomb_coord] = -1

    neighborhood = Neighborhood()
    grid = Grid(matrix=field.T)
    finder = AStarFinder()

    for d in Direction:
        name, coords = d.value
        shortest_path = float("inf")

        for dest in destinations:
            x = origin[0] + coords[0]
            y = origin[1] + coords[1]
            if field[x][y] > 0:
                start = grid.node(x, y)
                end = grid.node(dest[0], dest[1])

                path, runs = finder.find_path(start, end, grid)
                grid.cleanup()

                if len(path) != 0 and len(path) < shortest_path:
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
    grid = Grid(matrix=field.T)
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


def can_move(field: np.ndarray, position: Position, bombs: List[Position]) -> Neighborhood:
    neighborhood = Neighborhood()
    field = field.copy()
    for bomb in bombs:
        field[bomb[0], bomb[1]] = 1
    for d in Direction:
        name, coords = d.value
        x = position[0] + coords[0]
        y = position[1] + coords[1]
        move_possible = field[x, y] == 0
        setattr(neighborhood, name, move_possible)
    return neighborhood


def try_to_move_into_safety(
        field,
        origin: Position,
        bombs: List[Bomb],
        bomb_distance: Neighborhood,
        can_move_in_direction: Neighborhood,
) -> Neighborhood:
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
        safety.append(not is_in_danger(field, new_origin, bombs))

    for i, move in enumerate(can_move_in_direction):
        safety[i] = move and safety[i]

    if True in safety:
        for (i, d) in enumerate(Direction):
            if not safety[i]:
                setattr(bomb_distance, d.value[0], 0)

    return bomb_distance


def is_in_danger(field: np.ndarray, origin: Position, bombs: List[Bomb]) -> bool:
    in_danger = False

    for bomb_coords, exp_time in bombs:
        for i in range(2):
            if origin[i] == bomb_coords[i]:
                other_i = (i + 1) % 2
                dist = abs(origin[other_i] - bomb_coords[other_i])
                if dist < 4:
                    if not crate_or_wall_in_between(field, i, other_i, origin, bomb_coords):
                        in_danger = True
                        break

    return in_danger


def crate_or_wall_in_between(field: np.ndarray, i, other_i: int, origin: Position, bomb: Position):
    steps = 1 if origin[other_i] < bomb[other_i] else -1

    for position in range(origin[other_i], bomb[other_i], steps):
        if i == 0:
            x = origin[i]
            y = position
        else:
            x = position
            y = origin[i]
        if field[x, y] != 0:
            return True
    return False


def next_to_bomb_target(field: np.ndarray, position: Position, players: List[Player]):
    b = items.Bomb(position, "", 4, settings.BOMB_POWER, None)
    blast_coords = b.get_blast_coords(field)

    for coord in blast_coords:
        if field[coord] == 1:
            return True

        for player in players:
            if player.position == coord:
                return True

    return False

    # for d in Direction:
    #     _, coords = d.value
    #     x = position[0] + coords[0]
    #     y = position[1] + coords[1]
    #
    #     if field[x, y] == 1:
    #         return True
    #
    #     for player in players:
    #         if player.position == (x, y):
    #             return True
    #
    # return False


def move_to_danger(
        field: np.ndarray, origin: Position, bombs: List[Bomb], explosion_map: np.ndarray
) -> Neighborhood:
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


def find_nearest_crate_approx(field: np.ndarray, origin: Position, bombs: List[Position]):
    crates = extract_crates(field)
    if len(crates) > 0:
        crates_np = np.array(crates)
        position_np = np.array(origin)
        distances = crates_np - position_np
        distances = np.abs(distances)
        distances = np.sum(distances, axis=1)

        smallest_indices = np.argsort(distances)

        crate_coords = crates_np[smallest_indices]

        for crate_coord in crate_coords:
            neighborhood = calculate_neighborhood_distance(field, origin, [crate_coord], bombs, with_crates=False)
            if neighborhood.minimum() < float('inf'):
                return neighborhood

    return Neighborhood()


def nearest_path_to_safety(field: np.ndarray, explosion_map: np.ndarray, position: Position, bombs: List[Bomb],
                           players: List[Player]):
    field = field.copy()
    possible_safety: Set = set()

    # Finding all spots next to a position in the explosion radius
    for bomb in bombs:
        if is_in_danger(field, position, [bomb]):
            bomb_coords, t = bomb
            b = items.Bomb(bomb_coords, None, t, settings.BOMB_POWER, None)
            blast_coords = b.get_blast_coords(field)

            dangers = list(filter(lambda coord: (coord[0] == bomb_coords[0] and coord[0] == position[0]) or (
                    coord[1] == bomb_coords[1] and coord[1] == position[1]), blast_coords))

            for danger in dangers:
                for d in Direction:
                    name, coords = d.value
                    possible_safe_position = (danger[0] + coords[0], danger[1] + coords[1])
                    if possible_safe_position not in possible_safety:
                        possible_safety.add(possible_safe_position)

    # Removing the spots which are actually in the explosion
    safety = possible_safety.copy()
    # For bombs that are not detonated now
    for bomb in bombs:
        bomb_coords, t = bomb
        b = items.Bomb(bomb_coords, None, t, settings.BOMB_POWER, None)
        blast_coords = b.get_blast_coords(field)

        for blast_coord in blast_coords:
            if blast_coord in safety.copy():
                safety.remove(blast_coord)
    # For actual current explosions
    for safety_pos in safety.copy():
        explosion_time = explosion_map[safety_pos]
        min_steps = abs(position[0] - safety_pos[0]) + abs(position[1] - safety_pos[1])
        if explosion_time >= min_steps:
            safety.remove(safety_pos)
    # Remove positions which are not accessible
    for safety_pos in safety.copy():
        if field[safety_pos] != 0:
            safety.remove(safety_pos)
    # Remove safety positions which are bombs:
    for bomb_coords, _ in bombs:
        if bomb_coords in safety.copy():
            safety.remove(bomb_coords)

    obstacles = [b[0] for b in bombs] + [p.position for p in players]

    shortest_safety_neighborhood = calculate_neighborhood_distance(field, position, list(safety), obstacles)
    return shortest_safety_neighborhood


def extract_crates(field: np.ndarray) -> List[Position]:
    crates = np.where(field == 1)
    return list(np.array(crates).T)


def next_to_bomb(origin: Position, bombs: List[Position]):
    bombs = np.array(bombs)

    distances = bombs - origin

    in_line_with_bomb_indices = np.where(distances == 0)[0]

    distances_in_line = distances[in_line_with_bomb_indices]

    distances_reachable_indices = np.all(distances_in_line < 4, axis=1)

    distances_reachable = distances_in_line[distances_reachable_indices]

    min_dist = np.argsort(np.sum(distances_reachable, axis=1))

    if len(min_dist) > 0:
        min_bomb = min_dist[0]

        if origin[0] == min_bomb[0]:
            if origin[1] < min_bomb[1]:
                pass


def extract_features(state: GameState) -> FeatureVector:
    bombs = [(x, y) for ((x, y), _) in state.bombs]
    can_move_in_direction = can_move(state.field, state.self.position, bombs)

    backup_shortest_path = Neighborhood(1 if can_move_in_direction.north else float('inf'),
                                        1 if can_move_in_direction.south else float('inf'),
                                        1 if can_move_in_direction.east else float('inf'),
                                        1 if can_move_in_direction.west else float('inf'))

    # Coins
    coin_exists = len(state.coins) > 0

    coin_distance = backup_shortest_path
    if coin_exists:
        coin_distance = calculate_neighborhood_distance(
            state.field, state.self.position, state.coins, state.bombs
        )

    # Crates
    crates = extract_crates(state.field)
    crate_exists = len(crates) > 0

    crate_distance = backup_shortest_path
    if crate_exists:
        crate_distance = find_nearest_crate_approx(state.field, state.self.position, bombs)

    # Bombs
    bombs = [(x, y) for ((x, y), _) in state.bombs]
    bomb_exists = len(bombs) > 0

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
                                                       state.bombs + [(state.self.position, 4)], state.others)
    bomb_drop_safe = current_bomb_drop_escapes.minimum() < float('inf')
    good_bomb = useful_bomb and bomb_drop_safe and state.self.is_bomb_possible

    safety = backup_shortest_path
    if in_danger:
        safety = nearest_path_to_safety(state.field, state.explosion_map, state.self.position, state.bombs,
                                        state.others)

    has_opponents = len(state.others) > 0

    opponent_dest = [o.position for o in state.others]
    opponent_distance = calculate_neighborhood_distance(state.field, state.self.position, opponent_dest,
                                                        [b[0] for b in state.bombs])

    return FeatureVector(
        coin_distance,
        coin_exists,
        crate_distance,
        crate_exists,
        in_danger,
        bomb_exists,
        mv_to_danger,
        bomb_drop_safe,
        good_bomb,
        safety,
        can_move_in_direction,
        opponent_distance,
        has_opponents
    )
