import argparse

import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import join, isfile
from typing import List, Dict, TextIO


def get_rewards_from_file(f: TextIO) -> List[float]:
    agent_rewards = []
    while line := f.readline():
        rewards = [float(reward) for reward in line.split(",")]
        mean = float(np.mean(rewards))
        agent_rewards.append(mean)
    return agent_rewards


def get_rewards() -> Dict[str, List[float]]:
    agents = listdir('agent_code')

    rewards = dict()

    for agent in agents:
        path = join('agent_code', agent, 'stats.list')
        if not isfile(path):
            path = join('agent_code', agent, 'rewards.json')

        if not isfile(path):
            continue

        with open(path) as f:
            rewards[agent] = get_rewards_from_file(f)

    return rewards


def create_plot(agent: str, rewards: List[float]):
    x = np.linspace(0, len(rewards), len(rewards))

    fig, ax = plt.subplots()

    ax.plot(x, rewards)
    ax.set_title(agent)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Total rewards per round')

    plt.show()


def get_rewards_from_files(paths: List[TextIO]) -> Dict[str, List[float]]:
    rewards = dict()

    for path in paths:
        rewards[path.name] = get_rewards_from_file(path)

    return rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, type=argparse.FileType('r'), nargs='*')
    args = parser.parse_args()

    if args.input and len(args.input) > 0:
        rewards = get_rewards_from_files(args.input)
    else:
        rewards = get_rewards()

    for agent in rewards:
        create_plot(agent, rewards[agent])
