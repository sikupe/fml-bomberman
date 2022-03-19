from typing import List, Optional

import numpy as np

from agent_code.common.nn_feature_vector import NNFeatureVector
from agent_code.common.q_table_feature_vector import QTableFeatureVector


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
