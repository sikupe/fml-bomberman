from __future__ import annotations

from collections import deque, namedtuple
from os.path import join, dirname, isfile
from typing import List

import numpy as np

from agent_code.common.events import extract_events_from_state
from agent_code.common.feature_extractor import convert_to_state_object
from agent_code.common.feature_extractor import extract_features
from agent_code.common.neighborhood import Mirror
from agent_code.common.train import update_q_table, setup_training_global, teardown_training, parse_train_env, \
    reward_from_events
from agent_code.q_learning_task_2 import rewards
from agent_code.q_learning_task_2.feature_vector import FeatureVector

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

MODEL_FILE, STATS_FILE, REWARDS_FILE, _, NO_TRAIN = parse_train_env(__name__)

TRANSITION_HISTORY_SIZE = 10

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyperparameter
gamma = 0.8
alpha = 0.05


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    setup_training_global(self, TRANSITION_HISTORY_SIZE)

    if isfile(MODEL_FILE):
        self.q_table = np.load(MODEL_FILE)
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
    new_state = convert_to_state_object(new_game_state)
    self.transitions.append(new_state)

    if old_game_state:
        old_state = convert_to_state_object(old_game_state)
        current_feature_state = extract_features(old_state, FeatureVector)
        next_feature_state = extract_features(new_state, FeatureVector)

        custom_events = extract_events_from_state(self, current_feature_state, next_feature_state, self_action)

        total_events = custom_events + events

        reward = reward_from_events(self, total_events, rewards.rewards)

        for mirror in Mirror:
            rot_current_state = current_feature_state.mirror(mirror)
            rot_next_state = next_feature_state.mirror(mirror)
            rot_action = Mirror.mirror_action(mirror, self_action)

            update_q_table(self, rot_current_state, rot_next_state, rot_action, reward, ACTIONS,
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
    current_feature_state = extract_features(old_state, FeatureVector)

    reward = reward_from_events(self, events, rewards.rewards)

    for mirror in Mirror:
        rot_current_state = current_feature_state.mirror(mirror)
        rot_action = Mirror.mirror_action(mirror, last_action)

        update_q_table(self, rot_current_state, None, rot_action, reward, ACTIONS,
                       alpha, gamma)

    teardown_training(self, join(dirname(__file__), 'rewards.json'))
    with open(STATS_FILE, 'a+') as f:
        f.write(f'{len(old_state.coins)}, ')
    np.save(MODEL_FILE, self.q_table)
