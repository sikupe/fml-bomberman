import json
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import join, isfile
from typing import List, Dict


def get_rewards() -> Dict[str, List[List[float]]]:
    agents = listdir('agent_code')

    rewards = dict()

    for agent in agents:
        path = join('agent_code', agent, 'rewards.json')
        if isfile(path):
            with open(path) as f:
                try:
                    agent_rewards = json.load(f)
                    rewards[agent] = agent_rewards
                except:
                    pass

    return rewards


def create_plot(agent: str, rewards: List[List[float]]):
    means = [np.mean(round_rewards) for round_rewards in rewards]
    x = np.linspace(0, len(means), len(means))

    fig, ax = plt.subplots()

    ax.plot(x, means)
    ax.set_title(agent)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Total rewards per round')

    plt.show()


if __name__ == '__main__':
    rewards = get_rewards()

    for agent in rewards:
        create_plot(agent, rewards[agent])
