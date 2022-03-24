from os.path import isfile

import numpy as np

from agent_code.common.callbacks import function_learning_act
from agent_code.common.feature_extractor import convert_to_state_object
from agent_code.common.feature_extractor import extract_features
from agent_code.q_learning_task_3_function_learning.feature_vector import FeatureVector
from agent_code.q_learning_task_3_function_learning.train import ACTIONS, WEIGHT_FILE


epsilon = 0.1


def setup(self):
    if isfile(WEIGHT_FILE):
        self.weights = np.load(WEIGHT_FILE)
    self.index = 0


def act(self, game_state: dict):
    game_state = convert_to_state_object(game_state)
    feature_vector = extract_features(game_state, FeatureVector)

    return function_learning_act(self, feature_vector, ACTIONS, epsilon)
