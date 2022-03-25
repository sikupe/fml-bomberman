import json
import subprocess
import csv
import numpy as np
import time
from typing import Dict, List, Tuple, Callable

from agent_code.q_learning_task_1.rewards import rewards

Mutation = Dict[str, float]
MutationFitness = Tuple[Mutation, float]


def mutation(original_state: Mutation, mutation_rate: float, mutation_count: int) -> List[Mutation]:
    mutations = []
    for i in range(mutation_count):
        mutation = original_state.copy()
        for reward in mutation:
            mutation[reward] += np.round(np.random.uniform(-mutation_rate / 2, mutation_rate / 2), decimals=1)

        mutations.append(mutation)

    return mutations


def intermediate_recombination(parent_1: Mutation, parent_2: Mutation) -> Mutation:
    # https://de.wikipedia.org/wiki/Rekombination_(evolution%C3%A4rer_Algorithmus)#Intermedi%C3%A4re_Rekombination
    result = dict()

    for event in parent_1:
        d = 0.25
        beta = np.random.uniform(-d, 1 + d)
        result[event] = parent_1[event] * beta + parent_2[event] * (1 - beta)

    return result


def fitness(mutation: Mutation, agent: str, opponents: List[str], train_scenario: str, train_rounds: int,
            test_scenario: str, test_rounds: int) -> float:
    current_time_millis = time.time_ns()

    model_file = f'/tmp/{current_time_millis}.npy'
    stats_file = f'/tmp/{current_time_millis}.txt'

    rewards_json = json.dumps(mutation)

    train_env = {
        'MODEL_FILE': model_file,
        'REWARDS': rewards_json
    }

    test_env = {
        'MODEL_FILE': model_file,
        'STATS_FILE': stats_file,
        'NO_TRAIN': 'True'
    }

    train_command = f'python3 main.py play --train 1 --scenario {train_scenario} --n-rounds {train_rounds} --no-gui --agents {agent} {" ".join(opponents)}'
    test_command = f'python3 main.py play --train 1 --scenario {test_scenario} --n-rounds {test_rounds} --no-gui --agents {agent} {" ".join(opponents)}'

    # Train
    subprocess.call(train_command, shell=True, env=train_env)

    # Test
    subprocess.call(test_command, shell=True, env=test_env)

    rows = [row for row in csv.reader(stats_file)][1:]

    score = [int(remaining_coins) + int(steps) for _, remaining_coins, steps in rows]

    return float(np.mean(score))


def selection(parents: List[Mutation], children: List[Mutation], mu: int, fitness_func: Callable[[Mutation], float]) -> \
        List[Mutation]:
    old_population = parents + children

    old_population_fitness: List[MutationFitness] = []

    for mutation in old_population:
        fitn = fitness_func(mutation)
        old_population_fitness.append((mutation, fitn))

    old_population_fitness = sorted(old_population_fitness, key=lambda x: x[1], reverse=True)

    return [mutation for mutation, fitn in old_population_fitness[:mu]]


def get_initial_state() -> Mutation:
    initial = dict()
    for event in rewards:
        initial[event] = 0.0
    return initial


def evolution(mu: int, lambda_param: int, iterations: int, agent: str, opponents: List[str], train_scenario: str,
              train_rounds: int,
              test_scenario: str, test_rounds: int):
    initial = get_initial_state()

    mutation_rates = np.linspace(100, 0, iterations)

    parents = mutation(initial, mutation_rates[0], mu)

    for i in range(iterations):
        children = []
        # Recombination
        for _ in range(lambda_param):
            parent_1 = parents[np.random.randint(0, mu)]
            parent_2 = parents[np.random.randint(0, mu)]

            child = intermediate_recombination(parent_1, parent_2)

            children.append(child)

        # Mutation
        children = [mutation(child, mutation_rates[i], 1)[0] for child in children]

        # Selection
        parents = selection(parents, children, mu,
                            lambda mut: fitness(mut, agent, opponents, train_scenario, train_rounds, test_scenario,
                                                test_rounds))


if __name__ == '__main__':
    evolution(30, 30 * 7, 10, 'q_learning_task_1', [], 'coin-hell', 10, 'coin-heaven', 10)
