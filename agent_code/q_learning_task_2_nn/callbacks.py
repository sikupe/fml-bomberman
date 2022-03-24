from os.path import isfile

import numpy as np
import torch

from agent_code.common.callbacks import nn_act
from agent_code.common.feature_extractor import convert_to_state_object
from agent_code.common.feature_extractor import extract_features
from agent_code.q_learning_task_1_nn.feature_vector import FeatureVector
from agent_code.q_learning_task_2_nn.q_nn import QNN
from agent_code.q_learning_task_2_nn.train import ACTIONS, Q_NN_FILE

epsilon = 0.1


def setup(self):
    if isfile(Q_NN_FILE):
        self.model = QNN(FeatureVector.size(), len(ACTIONS))
        self.model.load_state_dict(torch.load(Q_NN_FILE))
        self.model.eval()


def act(self, game_state: dict):
    game_state = convert_to_state_object(game_state)
    feature_vector = extract_features(game_state, FeatureVector)

    nn_act(self, feature_vector, ACTIONS, epsilon)
