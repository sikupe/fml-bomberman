from __future__ import annotations

import os
from os.path import join, dirname, isfile
from typing import List

import numpy as np

from agent_code.common.feature_extractor import convert_to_state_object
from agent_code.common.neighborhood import Mirror
from agent_code.common.train import update_q_table
from agent_code.q_learning_task_3_extended_feature_space import rewards
from agent_code.q_learning_task_3_extended_feature_space.feature_extractor import extract_features
from agent_code.q_learning_task_3_extended_feature_space.feature_vector import FeatureVector

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

Q_TABLE_FILE = os.environ.get("Q_TABLE_FILE", join(dirname(__file__), 'q_learning_task_3_extended_feature_space.npy'))
STATS_FILE = os.environ.get("STATS_FILE", join(dirname(__file__), 'stats_q_learning_task_3_extended_feature_space.txt'))
with open(STATS_FILE, 'a+') as f:
    f.write(f'SCORE, SCORE2, SCORE3, SCORE4, ENDSTATE, LAST STEP\n')

# Hyperparameter
gamma = 0.9
alpha = 0.05


# def is_action_allowed(self, action: ACTIONS, ):


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

        for mirror in Mirror:
            rot_current_state = current_feature_state.mirror(mirror)
            rot_next_state = next_feature_state.mirror(mirror)
            rot_action = Mirror.mirror_action(mirror, self_action)
            rot_events = Mirror.mirror_events(mirror, total_events)

            update_q_table(self, rot_current_state, rot_next_state, rot_action, rot_events, reward_from_events, ACTIONS,
                           alpha, gamma)


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

    for mirror in Mirror:
        rot_current_state = current_feature_state.mirror(mirror)
        rot_action = Mirror.mirror_action(mirror, last_action)
        rot_events = Mirror.mirror_events(mirror, events)

        update_q_table(self, rot_current_state, None, rot_action, rot_events, reward_from_events, ACTIONS,
                       alpha, gamma)

    # Write Stats
    if "KILLED_SELF" in events: 
        endstate = "Suicide"
    elif "GOT_KILLED" in events:
        endstate = "Killed "
    else:
        endstate = "Survive"
        
    score_others = ""
    for opponent in old_state.others:
        score_others += f"{opponent.score}, "
    
    with open(STATS_FILE, 'a+') as f:
        f.write(f'{old_state.self.score}, {score_others}, {endstate}, {old_state.step}\n')
    np.save(Q_TABLE_FILE, self.q_table)


def extract_events_from_state(self, old_features: FeatureVector, new_features: FeatureVector, action: ACTIONS) -> List:
    custom_events = []

    old_features_coin_crate_distance, old_features_coin_crate_exists = old_features.coin_crate()
    new_features_coin_crate_distance, _ = new_features.coin_crate()

    if old_features_coin_crate_exists and old_features_coin_crate_distance.minimum() <= new_features_coin_crate_distance.minimum():
        custom_events.append(rewards.MOVED_AWAY_FROM_COIN)  # or crate
    elif old_features.coin_exists and old_features.coin_distance.minimum() > new_features.coin_distance.minimum():
        custom_events.append(rewards.APPROACH_COIN)  # or crate

    if old_features.in_danger and old_features.shortest_path_to_safety.minimum() <= new_features.shortest_path_to_safety.minimum():
        custom_events.append(rewards.MOVED_AWAY_FROM_SECURITY)
    elif old_features.in_danger and old_features.shortest_path_to_safety.minimum() > new_features.shortest_path_to_safety.minimum():
        custom_events.append(rewards.APPROACH_SECURITY)

    if old_features.has_opponents and old_features.opponent_distance.minimum() <= new_features.opponent_distance.minimum():
        custom_events.append(rewards.MOVED_AWAY_FROM_OPPONENT)
    elif old_features.has_opponents and old_features.opponent_distance.minimum() > new_features.opponent_distance.minimum():
        custom_events.append(rewards.APPROACH_OPPONENT)

    # if old_features.shortest_useful_path().minimum() <= new_features.shortest_useful_path().minimum():
    #     custom_events.append(rewards.MOVED_AWAY_FROM_USEFUL)
    # elif old_features.shortest_useful_path().minimum() > new_features.shortest_useful_path().minimum():
    #     custom_events.append(rewards.APPROACH_USEFUL)

    if new_features.in_danger:
        # TODO is that useful?
        if (action == 'BOMB' and old_features.in_danger) or action != 'BOMB':
            custom_events.append(rewards.IN_DANGER)

    if not old_features.in_danger and new_features.in_danger and not action == "BOMB":
        custom_events.append(rewards.MOVE_IN_DANGER)

    if action == 'BOMB':
        if old_features.good_bomb:
            custom_events.append(rewards.GOOD_BOMB)
        else:
            custom_events.append(rewards.BAD_BOMB)

    return custom_events


def is_invalid_action(action: ACTIONS, game_state):
    field = game_state.field
    origin = game_state.self.position
    is_bomb_possible = game_state.self.is_bomb_possible
    if action == "UP" and field[origin[0], origin[1] - 1] != 0:
        return True
    if action == "DOWN" and field[origin[0], origin[1] + 1] != 0:
        return True
    if action == "LEFT" and field[origin[0] - 1, origin[1]] != 0:
        return True
    if action == "RIGHT" and field[origin[0] + 1, origin[1]] != 0:
        return True
    if action == "BOMB" and not is_bomb_possible:
        return True
    return False


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
            raise Exception(f"Event is not in reward list: {event}")
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
