from os.path import isfile

import torch

from agent_code.common.act import nn_act
from agent_code.common.feature_extractor import convert_to_state_object, extract_features
from agent_code.q_learning_task_1_nn_evolution.feature_vector import FeatureVector
from agent_code.q_learning_task_1_nn_evolution.q_nn import QNN
from agent_code.q_learning_task_1_nn_evolution.train import ACTIONS, MODEL_FILE

epsilon = 0.25


def setup(self):
    if isfile(MODEL_FILE):
        self.model = QNN(FeatureVector.size(), len(ACTIONS))
        self.model.load_state_dict(torch.load(MODEL_FILE))
        self.model.eval()


def act(self, game_state: dict):
    self.logger.info('Pick action at random')

    game_state = convert_to_state_object(game_state)
    feature_vector = extract_features(game_state, FeatureVector)

    return nn_act(self, feature_vector, ACTIONS, epsilon)
