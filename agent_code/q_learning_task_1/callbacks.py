import numpy as np


def setup(self):
    # np.random.seed()
    # TODO Load model from hard drive to self.qs
    pass


def act(self, game_state: dict):
    self.logger.info('Pick action at random')

    if self.train:
        return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN'])

    # TODO Select action from self.qs
