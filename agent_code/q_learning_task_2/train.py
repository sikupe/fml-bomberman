from __future__ import annotations

from os.path import join, dirname, isfile
from typing import List, Optional

import numpy as np

from agent_code.q_learning_task_2 import rewards
from agent_code.q_learning_task_2.feature_extractor import extract_features, convert_to_state_object
from agent_code.q_learning_task_2.feature_vector import FeatureVector
import os

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

Q_TABLE_FILE = os.environ.get("Q_TABLE_FILE", join(dirname(__file__), 'q_learning_task_2.npy'))
STATS_FILE = os.environ.get("STATS_FILE", join(dirname(__file__), 'stats_q_learning_task_2.txt'))

# Hyperparameter
gamma = 1
alpha = 0.05


#def is_action_allowed(self, action: ACTIONS, ):


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    if isfile(Q_TABLE_FILE):
        self.q_table = np.load(Q_TABLE_FILE)
    else:
        self.q_table = np.zeros((FeatureVector.size(), len(ACTIONS)))


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    if old_game_state:
        old_state = convert_to_state_object(old_game_state)
        current_feature_state = extract_features(old_state)
        new_state = convert_to_state_object(new_game_state)
        next_feature_state = extract_features(new_state)

        custom_events = extract_events_from_state(self, current_feature_state, next_feature_state, self_action)

        total_events = custom_events + events

        update_q_table(self, current_feature_state, next_feature_state, self_action, total_events)


def update_q_table(self, current_feature_state: FeatureVector, next_feature_state: Optional[FeatureVector],
                   self_action: str, total_events: List[str]):
    reward = reward_from_events(self, total_events)

    current_action_index = ACTIONS.index(self_action)

    q_current = self.q_table[current_feature_state.to_state(), current_action_index]

    if next_feature_state:
        next_action_index = np.argmax(self.q_table[next_feature_state.to_state()])
        q_next = self.q_table[next_feature_state.to_state(), next_action_index]
    else:
        q_next = 0

    q_updated = q_current + alpha * (reward + gamma * q_next - q_current)

    self.q_table[current_feature_state.to_state(), current_action_index] = q_updated


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    old_state = convert_to_state_object(last_game_state)
    current_feature_state = extract_features(old_state)

    update_q_table(self, current_feature_state, None, last_action, events)

    with open(STATS_FILE, 'a+') as f:
        f.write(f'{len(old_state.coins)}, ')
    np.save(Q_TABLE_FILE, self.q_table)


def extract_events_from_state(self, old_features: FeatureVector, new_features: FeatureVector, action: ACTIONS) -> List:
    custom_events = []
    if old_features.coin_distance.minimum() < new_features.coin_distance.minimum():
        custom_events.append(rewards.MOVED_AWAY_FROM_COIN)
    elif old_features.coin_distance.minimum() > new_features.coin_distance.minimum():
        custom_events.append(rewards.APPROACH_COIN)

    if old_features.coin_distance.minimum() < new_features.coin_distance.minimum():
        custom_events.append(rewards.MOVED_AWAY_FROM_COIN)
    elif old_features.coin_distance.minimum() > new_features.coin_distance.minimum():
        custom_events.append(rewards.APPROACH_COIN)

    if old_features.crate_distance.minimum() < new_features.crate_distance.minimum():
        custom_events.append(rewards.MOVED_AWAY_FROM_CRATE)
    elif old_features.crate_distance.minimum() > new_features.crate_distance.minimum():
        custom_events.append(rewards.APPROACH_CRATE)

    if old_features.bomb_exists and old_features.bomb_distance.maximum() < new_features.bomb_distance.maximum():
        custom_events.append(rewards.MOVED_AWAY_FROM_BOMB)
    elif old_features.bomb_exists and old_features.bomb_distance.maximum() == new_features.bomb_distance.maximum() and new_features.in_danger:
        custom_events.append(rewards.APPROACH_BOMB)
    elif old_features.bomb_exists and old_features.bomb_distance.maximum() > new_features.bomb_distance.maximum():
        custom_events.append(rewards.APPROACH_BOMB)

    if new_features.in_danger:
        custom_events.append(rewards.IN_DANGER)
        
    if not old_features.in_danger and new_features.in_danger and not action == "BOMB":
        custom_events.append(rewards.MOVE_IN_DANGER)

    return custom_events

def is_invalid_action(action: ACTIONS, game_state):
    field = game_state.field
    origin = game_state.self.position
    is_bomb_possible = game_state.self.is_bomb_possible
    if action == "UP" and field[origin[0], origin[1]-1] != 0:
        return True;
    if action == "DOWN" and field[origin[0], origin[1]+1] != 0:
        return True;
    if action == "LEFT" and field[origin[0]-1, origin[1]] != 0:
        return True;
    if action == "RIGHT" and field[origin[0]+1, origin[1]] != 0:
        return True;
    if action == "BOMB" and not is_bomb_possible:
        return True;
    return False;


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    reward_sum = 0
    for event in events:
        if event in rewards.rewards:
            reward_sum += rewards.rewards[event]
        else:
            raise Exception("Event is not in reward list")
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
