from __future__ import annotations

import os
from os.path import join, dirname, isfile
from typing import List

import torch
import torch.nn as nn
from torch import optim

from agent_code.common.feature_extractor import convert_to_state_object
from agent_code.common.neighborhood import Mirror
from agent_code.common.train import update_nn
from agent_code.q_learning_task_1_nn_evolution import rewards
from agent_code.q_learning_task_1_nn_evolution.feature_extractor import extract_features
from agent_code.q_learning_task_1_nn_evolution.feature_vector import FeatureVector
from agent_code.q_learning_task_1_nn_evolution.q_nn import QNN

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

Q_NN_FILE = os.environ.get("Q_NN_FILE", join(dirname(__file__), 'qnn.pt'))
STATS_FILE = os.environ.get("STATS_FILE", join(dirname(__file__), 'q_learning_task_2_nn.txt'))

# Hyperparameter
gamma = 1
alpha = 0.05

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    if isfile(Q_NN_FILE):
        self.model = QNN(FeatureVector.size(), len(ACTIONS))
        self.model.load_state_dict(torch.load(Q_NN_FILE))
        self.model.eval()
    else:
        self.model = QNN(FeatureVector.size(), len(ACTIONS))

    self.optimizer = optim.Adam(self.model.parameters())
    self.criterion = nn.MSELoss()


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

        custom_events = extract_events_from_state(self, current_feature_state, next_feature_state)

        total_events = custom_events + events

        for mirror in Mirror:
            rot_current_state = current_feature_state.mirror(mirror)
            rot_next_state = next_feature_state.mirror(mirror)
            rot_action = Mirror.mirror_action(mirror, self_action)
            rot_events = Mirror.mirror_events(mirror, total_events)

            update_nn(self, rot_current_state, rot_next_state, rot_action, rot_events, reward_from_events, ACTIONS,
                      gamma)
        # update_nn(self, current_feature_state, next_feature_state, self_action, total_events, reward_from_events,
        #           ACTIONS, gamma)


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

    # for mirror in Mirror:
    #     rot_current_state = current_feature_state.mirror(mirror)
    #     rot_action = Mirror.mirror_action(mirror, last_action)
    #     rot_events = Mirror.mirror_events(mirror, events)
    #
    #     update_nn(self, rot_current_state, None, rot_action, rot_events, reward_from_events, ACTIONS, gamma)
    # update_nn(self, current_feature_state, None, last_action, events, reward_from_events, ACTIONS, gamma)

    with open(STATS_FILE, 'a+') as f:
        f.write(f'{len(old_state.coins)}, ')
    torch.save(self.model.state_dict(), Q_NN_FILE)


def extract_events_from_state(self, old_features: FeatureVector, new_features: FeatureVector) -> List:
    custom_events = []
    if old_features.coin_distance.minimum() <= new_features.coin_distance.minimum():
        custom_events.append(rewards.MOVED_AWAY_FROM_COIN)
    elif old_features.coin_distance.minimum() > new_features.coin_distance.minimum():
        custom_events.append(rewards.APPROACH_COIN)

    return custom_events


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
