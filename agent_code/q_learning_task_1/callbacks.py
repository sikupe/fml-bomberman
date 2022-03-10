from os.path import isfile

import numpy as np

from agent_code.q_learning_task_1.train import ACTIONS, Q_TABLE_FILE
from agent_code.q_learning_task_1.feature_extractor import extract_features, convert_to_state_object


def setup(self):
    if isfile(Q_TABLE_FILE):
        self.q_table = np.load(Q_TABLE_FILE)


def act(self, game_state: dict):
    self.logger.info('Pick action at random')

    game_state = convert_to_state_object(game_state)
    feature_vector = extract_features(game_state)
    if self.train and np.random.rand() > 0.8 and False:
        selected_action = np.random.choice(ACTIONS)
        self.logger.info(f"Selected action: {selected_action}")
        return selected_action
    else:
        # TODO Select action from self.q_table
        feature_state = feature_vector.to_state()
        props = self.q_table[feature_state]
        #print(f"'UP',{props[0]:0.2f} 'RIGHT',{props[1]:0.2f} 'DOWN',{props[2]:0.2f} 'LEFT',{props[3]:0.2f} 'WAIT' {props[4]:0.2f}")
        action_indicies = np.argwhere(self.q_table[feature_state] == np.max(self.q_table[feature_state])).T[0]
        return ACTIONS[np.random.choice(action_indicies)]
