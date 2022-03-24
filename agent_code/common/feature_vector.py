from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, Type, Generic

from agent_code.common.neighborhood import Neighborhood, Mirror

T = TypeVar("T", bound='BaseFeatureVector')


@dataclass
class BaseFeatureVector(Generic[T]):
    f_type: Type[T]
    coin_distance: Neighborhood
    crate_distance: Neighborhood
    in_danger: bool
    move_to_danger: Neighborhood
    bomb_drop_safe: bool
    good_bomb: bool
    shortest_path_to_safety: Neighborhood
    can_move_in_direction: Neighborhood
    opponent_distance: Neighborhood
    has_opponents: bool

    def mirror(self, mirror: Mirror) -> T:
        return self.f_type(self.f_type, self.coin_distance.mirror(mirror), self.crate_distance.mirror(mirror),
                           self.in_danger, self.move_to_danger.mirror(mirror), self.bomb_drop_safe, self.good_bomb,
                           self.shortest_path_to_safety.mirror(mirror), self.can_move_in_direction.mirror(mirror),
                           self.opponent_distance.mirror(mirror), self.has_opponents)
