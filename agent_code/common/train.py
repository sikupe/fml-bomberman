from typing import List, Optional

import numpy as np

from agent_code.common.function_learning_feature_vector import FunctionLearningFeatureVector
from agent_code.common.game_state import GameState
from agent_code.common.nn_feature_vector import NNFeatureVector
from agent_code.common.q_table_feature_vector import QTableFeatureVector


def detect_wiggle(states: List[GameState]) -> int:
    positions = list(map(lambda s: s.self.position, states))
    without_duplicates = [positions[i] for i in range(1, len(positions)) if positions[i] != positions[i - 1]]
    wiggles = [True for i in range(3, len(without_duplicates)) if
               without_duplicates[i] == without_duplicates[i - 2] and without_duplicates[i - 1] == without_duplicates[
                   i - 3]]

    return len(wiggles)


def update_weights(self, current_feature_state: FunctionLearningFeatureVector,
                   next_feature_state: Optional[FunctionLearningFeatureVector],
                   self_action: str, total_events: List[str], reward_from_events, possible_actions, alpha, gamma):
    reward = reward_from_events(self, total_events)

    current_action_index = possible_actions.index(self_action)

    q_current = np.max(self.weights[current_action_index, :] @ current_feature_state.to_state())

    weight = self.weights[current_action_index, :]

    if next_feature_state:
        q_next = np.max(self.weights @ next_feature_state.to_state())
    else:
        q_next = 0

    weight_updates = weight + alpha * (reward + gamma * q_next - q_current) * current_feature_state.to_state()

    self.weights[current_action_index, :] = weight_updates


def update_q_table(self, current_feature_state: QTableFeatureVector, next_feature_state: Optional[QTableFeatureVector],
                   self_action: str, total_events: List[str], reward_from_events, possible_actions, alpha, gamma):
    reward = reward_from_events(self, total_events)

    current_action_index = possible_actions.index(self_action)

    q_current = self.q_table[current_feature_state.to_state(), current_action_index]

    if next_feature_state:
        next_action_index = np.argmax(self.q_table[next_feature_state.to_state()])
        q_next = self.q_table[next_feature_state.to_state(), next_action_index]
    else:
        q_next = 0

    q_updated = q_current + alpha * (reward + gamma * q_next - q_current)

    self.q_table[current_feature_state.to_state(), current_action_index] = q_updated


def update_nn(self, current_feature_state: NNFeatureVector, next_feature_state: Optional[NNFeatureVector],
              self_action: str, total_events: List[str], reward_from_events, possible_actions, gamma):
    reward = reward_from_events(self, total_events)

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
