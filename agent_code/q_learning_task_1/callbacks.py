from os.path import isfile

import numpy as np

from agent_code.common.feature_extractor import convert_to_state_object
from agent_code.q_learning_task_1.feature_extractor import extract_features
from agent_code.q_learning_task_1.train import ACTIONS, Q_TABLE_FILE


def setup(self):
    if isfile(Q_TABLE_FILE):
        self.q_table = np.load(Q_TABLE_FILE)
    self.index = 0


def act(self, game_state: dict):
    game_state = convert_to_state_object(game_state)
    feature_vector = extract_features(game_state)
    # if self.train:

    q_table = self.q_table.copy()

    probabilities = q_table[feature_vector.to_state()].copy()
    self.logger.info(f'{self.index}: Current train probabilities: {probabilities}')
    if self.train:
        probabilities -= np.min(probabilities)
        prob_sum = np.sum(probabilities)

        if prob_sum != 0:
            probabilities = probabilities / prob_sum
        else:
            probabilities = None

        selected_action = np.random.choice(ACTIONS, p=probabilities)
    else:
        selected_action = ACTIONS[np.argmax(probabilities)]
    self.logger.info(f"{self.index}: Selected action: {selected_action}")
    self.index += 1
    return selected_action
# else:
#     # TODO Select action from q_table
#     feature_state = feature_vector.to_state()
#     action_index = np.argmax(q_table[feature_state])
#     return ACTIONS[action_index]
