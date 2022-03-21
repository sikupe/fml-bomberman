from os.path import isfile

import numpy as np
import torch

from agent_code.q_learning_task_1_nn_evolution.feature_vector import FeatureVector
from agent_code.q_learning_task_1_nn_evolution.q_nn import QNN
from agent_code.q_learning_task_1_nn_evolution.train import ACTIONS, Q_NN_FILE
from agent_code.q_learning_task_1_nn_evolution.feature_extractor import extract_features
from agent_code.common.feature_extractor import convert_to_state_object


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
    if self.train:
        self.logger.debug(f'Current train probabilities: {probabilities}')

        probabilities -= np.min(probabilities)
        prob_sum = np.sum(probabilities)

        if prob_sum != 0:
            probabilities = probabilities / prob_sum
        else:
            probabilities = None

        selected_action = np.random.choice(ACTIONS, p=probabilities)
    else:
        selected_action = ACTIONS[np.argmax(probabilities)]
    self.logger.info(f"Selected action: {selected_action}")
    return selected_action
