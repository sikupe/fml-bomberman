from os.path import isfile

import numpy as np
import torch

from agent_code.common.feature_extractor import convert_to_state_object
from agent_code.q_learning_task_2_nn.feature_extractor import extract_features
from agent_code.q_learning_task_2_nn.feature_vector import FeatureVector
from agent_code.q_learning_task_2_nn.q_nn import QNN
from agent_code.q_learning_task_2_nn.train import ACTIONS, Q_NN_FILE

epsilon = 0.1


def setup(self):
    if isfile(Q_NN_FILE):
        self.model = QNN(FeatureVector.size(), len(ACTIONS))
        self.model.load_state_dict(torch.load(Q_NN_FILE))
        self.model.eval()


def act(self, game_state: dict):
    self.logger.info('Pick action at random')

    game_state = convert_to_state_object(game_state)
    feature_vector = extract_features(game_state)

    probabilities = self.model(feature_vector.to_nn_state()).detach().numpy()
    self.logger.debug(f'Current train probabilities: {probabilities}')

    # Smallest to highest
    action_indices = np.argsort(probabilities)
    # Reversing, as we are interested in the highest probability
    action_indices = action_indices[::-1]

    if self.train:
        if np.random.rand() < epsilon or np.all(probabilities == 0):
            selected_action = np.random.choice(ACTIONS)
        else:
            selected_action = ACTIONS[action_indices[0]]
    else:
        selected_action = ACTIONS[action_indices[0]]

    self.logger.info(f"Selected action: {selected_action}")
    return selected_action
