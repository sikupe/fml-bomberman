from os.path import isfile

import numpy as np

from agent_code.q_learning_task_1.train import ACTIONS, Q_TABLE_FILE
from feature_extractor import extract_features, convert_to_state_object


def setup(self):
    if isfile(Q_TABLE_FILE):
        self.q_table = np.load(Q_TABLE_FILE)



def act(self, game_state: dict):
    self.logger.info('Pick action at random')

    game_state = convert_to_state_object(game_state)
    feature_vector = extract_features(game_state)
    if self.train:
        return np.random.choice(ACTIONS, p=self.q_table[feature_vector.to_state()])
    else:
        # TODO Select action from self.q_table
        feature_state = feature_vector.to_state()
        action_index = np.argmax(self.q_table[feature_state])
        return ACTIONS[action_index]
