import csv
import json
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Callable
from genealogy import Genealogy

import numpy as np

PROJECT_DIR = (
    subprocess.run("git rev-parse --show-toplevel", shell=True, stdout=subprocess.PIPE)
        .stdout.decode()
        .replace("\n", "")
)
sys.path.append(PROJECT_DIR)

from agent_code.common.events import APPROACH_COIN, MOVED_AWAY_FROM_COIN, WIGGLE
import events as e

Mutation = Dict[str, float]
MutationFitness = Tuple[Mutation, float, str]

MOVED = 'MOVED'


def mutation(
        original_state: Mutation, mutation_rate: float, mutation_count: int
) -> List[Mutation]:
    mutations = []
    for _ in range(mutation_count):
        mutation = original_state.copy()
        for reward in mutation:
            mutation[reward] += np.round(
                np.random.uniform(-mutation_rate / 2, mutation_rate / 2), decimals=1
            )

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


def fitness_mock(
        _mutation: Mutation,
        _agent: str,
        _opponents: List[str],
        _train_scenario: str,
        _train_rounds: int,
        _test_scenario: str,
        _test_rounds: int,
) -> Tuple[float, str]:
    return np.random.rand(), str(uuid.uuid4())


def fitness(
        mutation: Mutation,
        agent: str,
        opponents: List[str],
        train_scenario: str,
        train_rounds: int,
        test_scenario: str,
        test_rounds: int,
) -> Tuple[float, str]:
    name = str(uuid.uuid4())

    model_file = f"/tmp/{name}.npy"
    stats_file = f"/tmp/{name}.txt"

    rewards_json = json.dumps(mutation)

    train_env = {"MODEL_FILE": model_file, "REWARDS": rewards_json}

    test_env = {"MODEL_FILE": model_file, "STATS_FILE": stats_file, "NO_TRAIN": "True"}

    train_command = f'/bin/bash -c "source venv/bin/activate && python3 main.py play --train 1 --scenario {train_scenario} --n-rounds {train_rounds} --no-gui --agents {agent} {" ".join(opponents)}"'
    test_command = f'/bin/bash -c "source venv/bin/activate && python3 main.py play --train 1 --scenario {test_scenario} --n-rounds {test_rounds} --no-gui --agents {agent} {" ".join(opponents)}"'

    # Train
    subprocess.call(train_command, shell=True, env=train_env)

    # Test
    subprocess.call(test_command, shell=True, env=test_env)

    with open(stats_file) as csvfile:
        rows = [row for row in csv.reader(csvfile, delimiter=",")][1:]

        score = [
            int(remaining_coins) + int(steps) for _, remaining_coins, steps in rows
        ]

        return float(np.mean(score)), name


def selection(
        parents: List[Mutation],
        children: List[Mutation],
        mu: int,
        fitness_func: Callable[[Mutation], Tuple[float, str]],
) -> Tuple[List[int], List[Mutation], List[str]]:
    old_population = parents + children

    old_population_fitness: List[MutationFitness] = []

    executor = ThreadPoolExecutor(max_workers=10)

    old_phenotypes = [to_phenotype(mut) for mut in old_population]

    futures = [
        executor.submit(lambda: fitness_func(mut)) for mut in old_phenotypes
    ]

    for i, future in enumerate(futures):
        try:
            fitn, name = future.result()
        except Exception as exc:
            print("%r generated an exception: %s" % (old_population[i], exc))
        else:
            old_population_fitness.append((old_population[i], fitn, name))

    winner_indicies: List[int] = np.argsort(
        np.array([fitn for _, fitn, _ in old_population_fitness])
    )[::-1].tolist()[:mu]

    old_population_fitness = sorted(
        old_population_fitness, key=lambda x: x[1], reverse=False
    )

    return (
        winner_indicies,
        [mutation for mutation, _, _ in old_population_fitness[:mu]],
        [name for _, _, name in old_population_fitness[:mu]]
    )


def to_phenotype(genotype: Mutation) -> Mutation:
    phenotype = genotype.copy()

    phenotype[e.MOVED_UP] = phenotype[MOVED]
    phenotype[e.MOVED_DOWN] = phenotype[MOVED]
    phenotype[e.MOVED_LEFT] = phenotype[MOVED]
    phenotype[e.MOVED_RIGHT] = phenotype[MOVED]

    del phenotype[MOVED]

    return phenotype


def get_initial_state() -> Mutation:
    return {
        e.COIN_COLLECTED: 0,
        MOVED: 0,
        e.WAITED: 0,
        APPROACH_COIN: 0,
        MOVED_AWAY_FROM_COIN: 0,
        WIGGLE: 0,
        e.SURVIVED_ROUND: 0,
    }


def evolution(
        mu: int,
        lambda_param: int,
        iterations: int,
        agent: str,
        opponents: List[str],
        train_scenario: str,
        train_rounds: int,
        test_scenario: str,
        test_rounds: int,
):
    # Generate a genealogy
    genealogy = Genealogy(mu)

    initial = get_initial_state()

    mutation_rates = np.concatenate([100 * np.exp(- np.linspace(0, int(iterations / 3), int(iterations / 3))),
                                     np.linspace(100 * np.exp(- int(iterations / 3)), 0,
                                                 iterations - int(iterations / 3))])

    parents = mutation(initial, mutation_rates[0], mu)

    model_names = ""

    for i in range(1, iterations):
        children = []
        # Recombination
        for _ in range(lambda_param):
            parent_1_index = np.random.randint(0, mu)
            parent_2_index = np.random.randint(0, mu)
            parent_1 = parents[parent_1_index]
            parent_2 = parents[parent_2_index]

            child = intermediate_recombination(parent_1, parent_2)

            children.append(child)
            genealogy.add_child(parent_1_index, parent_2_index)

        # Mutation
        children = [mutation(child, mutation_rates[i], 1)[0] for child in children]

        # Selection
        winner_indicies, parents, names = selection(
            parents,
            children,
            mu,
            lambda mut: fitness(
                mut,
                agent,
                opponents,
                train_scenario,
                train_rounds,
                test_scenario,
                test_rounds,
            ),
        )

        model_names = names
        genealogy.process_winners(winner_indicies, i)

    genealogy.generate_dot(mu, lambda_param, iterations)

    for name, parent in zip(model_names, parents):
        print(f"Model: {name}")
        print(json.dumps(parent, indent=4))


if __name__ == "__main__":
    evolution(
        4, 4 * 3, 10, "q_learning_task_1", [], "coin-hell", 10, "coin-heaven", 10
    )
