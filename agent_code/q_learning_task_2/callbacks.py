from os.path import isfile

import numpy as np

from agent_code.common.act import q_table_act
from agent_code.common.feature_extractor import convert_to_state_object
from agent_code.common.feature_extractor import extract_features
from agent_code.q_learning_task_2.feature_vector import FeatureVector
from agent_code.q_learning_task_2.train import ACTIONS, MODEL_FILE

epsilon = 0.1


def setup(self):
    if isfile(MODEL_FILE):
        self.q_table = np.load(MODEL_FILE)


def act(self, game_state: dict):
    game_state = convert_to_state_object(game_state)
    feature_vector = extract_features(game_state, FeatureVector)

    return q_table_act(self, feature_vector, ACTIONS, epsilon)
