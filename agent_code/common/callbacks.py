from typing import List

import numpy as np

from agent_code.common.function_learning_feature_vector import FunctionLearningFeatureVector
from agent_code.common.nn_feature_vector import NNFeatureVector
from agent_code.common.q_table_feature_vector import QTableFeatureVector
from agent_code.common.train import parse_notrain

NOTRAIN = parse_notrain()

def act(self, probabilities: np.ndarray, actions: List[str], epsilon: float):
    self.logger.debug(f'Current train probabilities: {probabilities}')
    selected_action = actions[np.argmax(probabilities)]

    if not NOTRAIN:
        if self.train and (np.random.rand() < epsilon or np.all(probabilities == 0)):
            selected_action = np.random.choice(actions)

    self.logger.info(f"Selected action: {selected_action}")
    return selected_action


def q_table_act(self, feature_vector: QTableFeatureVector, actions: List[str], epsilon: float):
    probabilities = self.q_table[feature_vector.to_state()].copy()

    return act(self, probabilities, actions, epsilon)


def function_learning_act(self, feature_vector: FunctionLearningFeatureVector, actions: List[str], epsilon: float):
    probabilities = self.weights @ feature_vector.to_state()

    return act(self, probabilities, actions, epsilon)


def nn_act(self, feature_vector: NNFeatureVector, actions: List[str], epsilon: float):
    probabilities = self.model(feature_vector.to_nn_state()).detach().numpy()

    return act(self, probabilities, actions, epsilon)
