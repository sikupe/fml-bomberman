import contextlib
import csv
import json
import os
import signal
import subprocess
import sys
import uuid
from os.path import isfile

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Callable

import numpy as np

PROJECT_DIR = (
    subprocess.run("git rev-parse --show-toplevel", shell=True, stdout=subprocess.PIPE)
        .stdout.decode()
        .replace("\n", "")
)
sys.path.append(PROJECT_DIR)

from agent_code.common.events import MOVE_IN_DANGER, MOVE_OUT_OF_DANGER, GOOD_BOMB, BAD_BOMB, IN_DANGER, \
    APPROACH_SECURITY, MOVED_AWAY_FROM_SECURITY, APPROACH_COIN, MOVED_AWAY_FROM_COIN, APPROACH_CRATE, \
    MOVED_AWAY_FROM_CRATE, APPROACH_OPPONENT, MOVED_AWAY_FROM_OPPONENT, WIGGLE
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


def extract_score_task_1(stats_file: str):
    with open(stats_file) as csvfile:
        rows = [row for row in csv.reader(csvfile, delimiter=",")][1:]

        score = [
            int(remaining_coins) + int(steps) for _, remaining_coins, steps in rows
        ]

        return float(np.mean(score))


def extract_score_task_3(stats_file: str):
    with open(stats_file) as csvfile:
        rows = [row for row in csv.reader(csvfile, delimiter=",")][1:]

        score = [
            float(points) * float(endstate) for points, endstate, _ in rows
        ]

        return float(np.mean(score))


def fitness(
        mutation: Mutation,
        agent: str,
        opponents: List[str],
        train_scenario: str,
        train_rounds: int,
        test_scenario: str,
        test_rounds: int,
        extract_score: Callable[[str], float]
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
    subprocess.call(train_command, shell=True, env=train_env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Test
    subprocess.call(test_command, shell=True, env=test_env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    return extract_score(stats_file), name


def dump_mutations(iteration: int, mutation_fitnesses: List[MutationFitness]):
    result_file = 'reward_evolution-result.json'
    result = []
    if isfile(result_file):
        with open(result_file) as f:
            result = json.load(f)

    result.append({
        iteration: iteration,
        'mutations': [{'mutation': mutation, 'fitness': fitness} for mutation, fitness, _ in mutation_fitnesses]
    })

    with open(result_file, 'w+') as f:
        json.dump(result, f)


def selection(
        iteration: int,
        parents: List[Mutation],
        children: List[Mutation],
        mu: int,
        reverse: bool,
        fitness_func: Callable[[Mutation], Tuple[float, str]],
) -> Tuple[List[int], List[Mutation], List[str]]:
    old_population = parents + children

    old_population_fitness: List[MutationFitness] = []

    executor = ThreadPoolExecutor(max_workers=10)

    old_phenotypes = [to_phenotype(mut) for mut in old_population]

    futures = [
        executor.submit(lambda: fitness_func(mut)) for mut in old_phenotypes
    ]

    for i, future in enumerate(tqdm(futures, position=1, desc="Selection")):
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
        old_population_fitness, key=lambda x: x[1], reverse=reverse
    )

    dump_mutations(iteration, old_population_fitness)

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


def get_initial_state_task_1() -> Mutation:
    return {
        e.COIN_COLLECTED: 25,
        MOVED: -2,
        e.INVALID_ACTION: -7,
        e.WAITED: -10,
        APPROACH_COIN: 5,
        MOVED_AWAY_FROM_COIN: -5,
        WIGGLE: 0,
        e.SURVIVED_ROUND: 5,
    }


def get_initial_state_for_task_3() -> Mutation:
    return {
        MOVED: 0,
        APPROACH_COIN: 0,
        MOVED_AWAY_FROM_COIN: 0,
        WIGGLE: 0,
        IN_DANGER: 0,
        MOVE_IN_DANGER: 0,
        MOVE_OUT_OF_DANGER: 0,
        GOOD_BOMB: 0,
        BAD_BOMB: 0,
        APPROACH_SECURITY: 0,
        MOVED_AWAY_FROM_SECURITY: 0,
        APPROACH_CRATE: 0,
        MOVED_AWAY_FROM_CRATE: 0,
        APPROACH_OPPONENT: 0,
        MOVED_AWAY_FROM_OPPONENT: 0,
        e.COIN_COLLECTED: 0,
        e.INVALID_ACTION: 0,
        e.WAITED: 0,
        e.SURVIVED_ROUND: 0,
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,
        e.COIN_FOUND: 0,
        e.CRATE_DESTROYED: 0,
        e.GOT_KILLED: 0,
        e.KILLED_SELF: 0,
        e.KILLED_OPPONENT: 0,
        e.OPPONENT_ELIMINATED: 0,
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
        extract_score: Callable[[str], float],
        get_initial_state: Callable[[], Mutation]
):
    # Generate a genealogy
    # genealogy = Genealogy(mu)

    initial = get_initial_state()

    mutation_rates = np.concatenate([0.5 * np.exp(- np.linspace(0, int(iterations / 3), int(iterations / 3))),
                                     np.linspace(100 * np.exp(- int(iterations / 3)), 0,
                                                 iterations - int(iterations / 3))])

    parents = mutation(initial, mutation_rates[0], mu)

    model_names = ""

    for i in tqdm(range(1, iterations), position=0, desc="Iteration"):
        children = []
        # Recombination
        for _ in range(lambda_param):
            parent_1_index = np.random.randint(0, mu)
            parent_2_index = np.random.randint(0, mu)
            parent_1 = parents[parent_1_index]
            parent_2 = parents[parent_2_index]

            child = intermediate_recombination(parent_1, parent_2)

            children.append(child)
            # with contextlib.suppress(Exception):
            #     genealogy.add_child(parent_1_index, parent_2_index)

        # Mutation
        children = [mutation(child, mutation_rates[i], 1)[0] for child in children]

        # Selection
        winner_indices, parents, names = selection(
            i,
            parents,
            children,
            mu,
            True,
            lambda mut: fitness(
                mut,
                agent,
                opponents,
                train_scenario,
                train_rounds,
                test_scenario,
                test_rounds,
                extract_score,
            ),
        )

        model_names = names
        # with contextlib.suppress(Exception):
        #     genealogy.process_winners(winner_indices, i)

        print(f"ITERATION {i}")
        for name, parent in zip(model_names, parents):
            print(f"Model: {name}")
            print(json.dumps(parent, indent=4))

    # with contextlib.suppress(Exception):
    #     genealogy.generate_dot(mu, lambda_param, iterations)
    print("FINAL EVOLUTION RESULT")


if __name__ == "__main__":
    os.setpgrp()
    try:
        evolution(
            30, 30 * 7, 10, "q_learning_task_1", [], "coin-hell", 10, "coin-heaven", 10, extract_score_task_1,
            get_initial_state_task_1
        )
    except Exception as err:
        print(err)
    finally:
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
