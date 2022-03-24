from __future__ import annotations

from collections import deque, namedtuple
from os.path import join, dirname, isfile
from typing import List

import numpy as np

from agent_code.common.events import extract_events_from_state
from agent_code.common.feature_extractor import convert_to_state_object
from agent_code.common.feature_extractor import extract_features
from agent_code.common.neighborhood import Mirror
from agent_code.common.train import update_weights, detect_wiggle
from agent_code.q_learning_task_3_function_learning import rewards
from agent_code.q_learning_task_3_function_learning.feature_vector import FeatureVector
from agent_code.q_learning_task_3_function_learning.rewards import WIGGLE

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

WEIGHT_FILE = join(dirname(__file__), 'q_learning_task_3_function_learning.npy')
STATS_FILE = join(dirname(__file__), 'stats_q_learning_task_1.txt')

TRANSITION_HISTORY_SIZE = 10

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyperparameter
gamma = 0.95
alpha = 0.05


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    if isfile(WEIGHT_FILE):
        self.weights = np.load(WEIGHT_FILE)
    else:
        self.weights = np.zeros((len(ACTIONS), FeatureVector.size()))


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
    new_state = convert_to_state_object(new_game_state)
    self.transitions.append(new_state)

    if old_game_state:
        old_state = convert_to_state_object(old_game_state)
        current_feature_state = extract_features(old_state, FeatureVector)
        next_feature_state = extract_features(new_state, FeatureVector)

        custom_events = extract_events_from_state(self, current_feature_state, next_feature_state, self_action)

        total_events = custom_events + events

        for mirror in Mirror:
            rot_current_state = current_feature_state.mirror(mirror)
            rot_next_state = next_feature_state.mirror(mirror)
            rot_action = Mirror.mirror_action(mirror, self_action)
            rot_events = Mirror.mirror_events(mirror, total_events)

            update_weights(self, rot_current_state, rot_next_state, rot_action, rot_events, reward_from_events, ACTIONS,
                           alpha, gamma)
        # update_q_table(self, current_feature_state, next_feature_state, self_action, total_events, reward_from_events,
        #                ACTIONS, alpha, gamma)


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
    current_feature_state = extract_features(old_state, FeatureVector)

    for mirror in Mirror:
        rot_current_state = current_feature_state.mirror(mirror)
        rot_action = Mirror.mirror_action(mirror, last_action)
        rot_events = Mirror.mirror_events(mirror, events)

        update_weights(self, rot_current_state, None, rot_action, rot_events, reward_from_events, ACTIONS,
                       alpha, gamma)

    with open(STATS_FILE, 'a+') as f:
        f.write(f'{len(old_state.coins)}, ')
    np.save(WEIGHT_FILE, self.weights)


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
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum