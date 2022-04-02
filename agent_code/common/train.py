from collections import deque, namedtuple
import contextlib
from typing import List, Optional, Dict, Tuple
import os
import re
from os.path import join, dirname

import numpy as np

from agent_code.common.function_learning_feature_vector import FunctionLearningFeatureVector
from agent_code.common.game_state import GameState
from agent_code.common.nn_feature_vector import NNFeatureVector
from agent_code.common.q_table_feature_vector import QTableFeatureVector


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def parse_notrain() -> bool:
    """Convert string of env var to bool."""
    bool_dict = {"true": True, "false": False}
    NO_TRAIN = bool_dict[os.environ.get("NO_TRAIN", "False").lower()]
    return NO_TRAIN


def parse_train_env(module_name: str) -> Tuple[str, str, str, str, bool]:
    """Parse env var and return values."""
    MODEL_FILE = os.environ.get("MODEL_FILE", join(dirname(module_name), 'model.npy'))
    STATS_FILE = os.environ.get("STATS_FILE", join(dirname(module_name), 'stats.txt'))
    REWARDS_FILE = re.sub(r"\..*$", ".list", STATS_FILE)
    MODEL_FILE_COUNTER = os.environ.get("MODEL_FILE_COUNTER", join(dirname(module_name), 'model_counter.npy'))
    return MODEL_FILE, STATS_FILE, REWARDS_FILE, MODEL_FILE_COUNTER, parse_notrain()


def setup_training_global(self, transition_history_size: int):
    self.transitions = deque(maxlen=transition_history_size)
    self.rewards = []


def teardown_training(self, rewards_file: str):
    with open(rewards_file, 'a+') as f:
        f.write(",".join([str(r) for r in self.rewards]))
        f.write("\n")
        self.rewards.clear()


def detect_wiggle(states: List[GameState]) -> int:
    positions = list(map(lambda s: s.self.position, states))
    without_duplicates = [positions[i] for i in range(1, len(positions)) if positions[i] != positions[i - 1]]
    wiggles = [True for i in range(3, len(without_duplicates)) if
               without_duplicates[i] == without_duplicates[i - 2] and without_duplicates[i - 1] == without_duplicates[
                   i - 3]]

    return len(wiggles)


def reward_from_events(self, events: List[str], rewards: Dict[str, float]) -> float:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    reward_sum = 0
    for event in events:
        if event in rewards:
            reward_sum += rewards[event]
        else:
            self.logger.error("Event is not in reward list: %s", event)
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    self.rewards.append(reward_sum)
    return reward_sum


def update_weights(self, current_feature_state: FunctionLearningFeatureVector,
                   next_feature_state: Optional[FunctionLearningFeatureVector],
                   self_action: str, total_events: List[str], rewards: Dict[str, float], possible_actions: List[str],
                   alpha: float,
                   gamma: float):
    reward = reward_from_events(self, total_events, rewards)

    current_action_index = possible_actions.index(self_action)

    q_current = np.max(self.weights[current_action_index, :] @ current_feature_state.to_state())

    assert not np.any(self.weights == np.nan) and not np.any(current_feature_state.to_state() == np.nan)

    weight = self.weights[current_action_index, :]

    if next_feature_state:
        q_next = np.max(self.weights @ next_feature_state.to_state())
    else:
        q_next = 0

    weight_updates = weight + alpha * (reward + gamma * q_next - q_current) * current_feature_state.to_state()

    self.weights[current_action_index, :] = weight_updates


def update_q_table(self, current_feature_state: QTableFeatureVector, next_feature_state: Optional[QTableFeatureVector],
                   self_action: str, reward: float, possible_actions: List[str],
                   alpha: float, gamma: float):
    current_action_index = possible_actions.index(self_action)

    q_current = self.q_table[current_feature_state.to_state(), current_action_index]

    if next_feature_state:
        next_action_index = np.argmax(self.q_table[next_feature_state.to_state()])
        q_next = self.q_table[next_feature_state.to_state(), next_action_index]
    else:
        q_next = 0

    q_updated = q_current + alpha * (reward + gamma * q_next - q_current)

    self.q_table[current_feature_state.to_state(), current_action_index] = q_updated
    with contextlib.suppress(AttributeError):
        self.q_table_counter[current_feature_state.to_state(), current_action_index] += 1


def update_nn(self, current_feature_state: NNFeatureVector, next_feature_state: Optional[NNFeatureVector],
              self_action: str, total_events: List[str], rewards: Dict[str, float], possible_actions: List[str],
              gamma: float):
    reward = reward_from_events(self, total_events, rewards)

    current_action_index = possible_actions.index(self_action)

    prediction = self.model(current_feature_state.to_nn_state())
    target = prediction.clone()

    if next_feature_state:
        q_next = np.max(prediction.detach().numpy())
    else:
        q_next = 0

    q_updated = reward + gamma * q_next

    target[current_action_index] = q_updated

    loss = self.criterion(prediction, target)
    loss.backward()
    self.optimizer.step()

    prediction_new = self.model(current_feature_state.to_nn_state())

    self.logger.debug(f'Old prediction: {prediction}, New prediction: {prediction_new}, Rewards: {reward}')
