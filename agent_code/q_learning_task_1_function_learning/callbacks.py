from os.path import isfile

import numpy as np

from agent_code.common.feature_extractor import convert_to_state_object
from agent_code.q_learning_task_1_function_learning.feature_extractor import extract_features
from agent_code.q_learning_task_1_function_learning.train import ACTIONS, WEIGHT_FILE


epsilon = 0.1


def setup(self):
    if isfile(WEIGHT_FILE):
        self.weights = np.load(WEIGHT_FILE)
    self.index = 0


def act(self, game_state: dict):
    game_state = convert_to_state_object(game_state)
    feature_vector = extract_features(game_state)

    probabilities = self.weights @ feature_vector.to_state()

    selected_action = ACTIONS[np.argmax(probabilities)]

    if self.train and (np.random.rand() < epsilon or np.all(probabilities == 0)):
        selected_action = np.random.choice(ACTIONS)

    self.logger.info(f"Selected action: {selected_action}")
    return selected_action
#     return ACTIONS[action_index]
