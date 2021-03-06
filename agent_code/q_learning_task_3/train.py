from __future__ import annotations

from collections import deque, namedtuple
from os.path import isfile
from typing import List

import numpy as np

from agent_code.common.events import extract_events_from_state
from agent_code.common.feature_extractor import convert_to_state_object, extract_features
from agent_code.common.neighborhood import Mirror
from agent_code.common.train import update_q_table, teardown_training, setup_training_global, parse_train_env, \
    reward_from_events
from agent_code.q_learning_task_3 import rewards
from agent_code.q_learning_task_3.feature_vector import FeatureVector

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

MODEL_FILE, STATS_FILE, REWARDS_FILE, MODEL_FILE_COUNTER, NO_TRAIN = parse_train_env(__name__)

TRANSITION_HISTORY_SIZE = 10
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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

    setup_training_global(self, TRANSITION_HISTORY_SIZE)

    if NO_TRAIN:
        with open(STATS_FILE, 'a+') as f:
            f.write(f'SCORE, ENDSTATE, LAST STEP\n')
        return

    if isfile(MODEL_FILE):
        self.q_table = np.load(MODEL_FILE)
        # self.q_table_counter = np.load(MODEL_FILE_COUNTER)
    else:
        self.q_table = np.zeros((FeatureVector.size(), len(ACTIONS)))
        # self.q_table_counter = np.zeros((FeatureVector.size(), len(ACTIONS)))


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

    if NO_TRAIN:
        return

    if old_game_state:
        old_state = convert_to_state_object(old_game_state)
        current_feature_state: FeatureVector = extract_features(old_state, FeatureVector)
        next_feature_state: FeatureVector = extract_features(new_state, FeatureVector)

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

    if NO_TRAIN:
        # Write Stats
        if "KILLED_SELF" in events:
            # endstate = "Suicide"
            endstate = 0.75
        elif "GOT_KILLED" in events:
            endstate = 0.85
            # endstate = "Killed "
        else:
            # endstate = "Survive"
            endstate = 1.25

        with open(STATS_FILE, 'a+') as f:
            f.write(f'{old_state.self.score}, {endstate}, {old_state.step}\n')
        return

    teardown_training(self, REWARDS_FILE)
    np.save(MODEL_FILE, self.q_table)
    # np.save(MODEL_FILE_COUNTER, self.q_table_counter)
    # np.savetxt("q_table_counter.csv", self.q_table_counter, delimiter=";")
