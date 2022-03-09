import time

import numpy as np

from agent_code.strong_students.feature_extractor import extract_features


def setup(self):
    # np.random.seed()
    pass

times = []


def measure_feature_extraction(game_state: dict):
    start = time.time()
    extract_features(game_state)
    end = time.time()

    exec_time = end - start
    times.append(exec_time)

    mean = np.mean(times)

    # print(f'extract_features took: {exec_time} seconds')
    # print(f'mean extract_features took: {mean} seconds')


def act(agent, game_state: dict):
    agent.logger.info('Pick action at random')

    measure_feature_extraction(game_state)

    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
