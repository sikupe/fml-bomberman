from os.path import isfile

import numpy as np

from agent_code.q_learning_task_2.train import ACTIONS, Q_TABLE_FILE
from agent_code.q_learning_task_2.feature_extractor import extract_features, convert_to_state_object


def setup(self):
    if isfile(Q_TABLE_FILE):
        self.q_table = np.load(Q_TABLE_FILE)


def act(self, game_state: dict):
    self.logger.info('Pick action at random')

    game_state = convert_to_state_object(game_state)
    feature_vector = extract_features(game_state)
    # if self.train:

    q_table = self.q_table.copy()

    probabilities = q_table[feature_vector.to_state()].copy()
    self.logger.debug(f'Current train probabilities: {probabilities}')

    probabilities -= np.min(probabilities)
    prob_sum = np.sum(probabilities)

    if prob_sum != 0:
        probabilities = probabilities / prob_sum
    else:
        probabilities = None

    selected_action = np.random.choice(ACTIONS, p=probabilities)
    self.logger.info(f"Selected action: {selected_action}")
    return selected_action
# else:
#     # TODO Select action from q_table
#     feature_state = feature_vector.to_state()
#     action_index = np.argmax(q_table[feature_state])
#     return ACTIONS[action_index]
