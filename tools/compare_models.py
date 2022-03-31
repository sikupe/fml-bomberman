#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import logging
import numpy as np
from numpy.core.function_base import linspace

parser = argparse.ArgumentParser(description="Merge multiple trained models")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser.add_argument('-n', '--names-list', nargs='+', default=[])


def parse_line(line: str, name: str) -> int:
    prev, last = line.split(":")
    assert (name in prev) or ("strong_students" in prev)
    return int(last.strip())


def parse_file(file_name: str) -> list[int]:
    res = []
    with open(file_name, "r") as f:
        counter = 0
        while True:
            try:
                agents = ["q_learning_task_3_advanced_features", "rule_based_agent", "rule_based_agent", "coin_collector_agent"]
                game = np.zeros((4))
                for i, agent in enumerate(agents):
                    line = f.readline()
                    if line and len(line) != 0:
                        game[i] = parse_line(line, agent)
                    else:
                        raise StopIteration()
                    counter += 1
                res.append((np.argsort(game)[::-1].argsort()+1)[0])
            except StopIteration:
                print("EOF")
                return res


if __name__ == "__main__":
    args = parser.parse_args()
    max_len = 0
    length = 0
    for file_name in args.names_list:
        data = np.array(parse_file(file_name))
        _, count = np.unique(data, return_counts=True)
        length = len(data)
        print(file_name)
        print(f"{count[0]}/{length} = {count[0]/length}")
        plt.plot(data, label=file_name)
        if length > max_len:
            max_len = length

    xticks = np.arange(0, max_len, step=1)
    yticks = [1, 2, 3, 4]

    plt.xticks(xticks, xticks)
    plt.yticks(yticks, yticks)

    ax = plt.gca()
    ax.invert_yaxis()
    plt.figlegend()
    plt.show()
