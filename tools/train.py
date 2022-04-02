import os
import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

configurations = {
    'q_learning_task_1': {
        'rounds': 250,
        'scenario': 'coin-hell',
        'opponents': []
    },
    'q_learning_task_1_function_learning': {
        'rounds': 250,
        'scenario': 'coin-hell',
        'opponents': []
    },
    'q_learning_task_1_nn': {
        'rounds': 250,
        'scenario': 'coin-hell',
        'opponents': []
    },
    'q_learning_task_1_nn_evolution': {
        'rounds': 250,
        'scenario': 'coin-hell',
        'opponents': []
    },
    'q_learning_task_2': {
        'rounds': 250,
        'scenario': 'few-crates',
        'opponents': []
    },
    'q_learning_task_2_function_learning': {
        'rounds': 250,
        'scenario': 'few-crates',
        'opponents': []
    },
    'q_learning_task_2_nn': {
        'rounds': 250,
        'scenario': 'few-crates',
        'opponents': []
    },
    'q_learning_task_3': {
        'rounds': 250,
        'scenario': 'train-crates',
        'opponents': [
            'peaceful_agent',
            'peaceful_agent',
            'peaceful_agent'
        ]
    },
    # 'q_learning_task_3_advanced_features': {
    #     'rounds': 1000,
    #     'scenario': 'classic',
    #     'opponents': [
    #         'rule_based_agent',
    #         'rule_based_agent',
    #         'coin_collector_agent'
    #     ]
    # },
    'q_learning_task_3_advanced_features': {
        'rounds': 250,
        'scenario': 'classic',
        'opponents': [
            'rule_based_agent',
            'rule_based_agent',
            'coin_collector_agent'
        ]
    },
    'q_learning_task_3_extended_feature_space': {
        'rounds': 250,
        'scenario': 'classic',
        'opponents': [
            'rule_based_agent',
            'rule_based_agent',
            'coin_collector_agent'
        ]
    },
    'q_learning_task_3_function_learning': {
        'rounds': 250,
        'scenario': 'few-crates',
        'opponents': [
            'peaceful_agent',
            'peaceful_agent',
            'peaceful_agent'
        ]
    },
    'q_learning_task_4_function_learning': {
        'rounds': 250,
        'scenario': 'few-crates',
        'opponents': [
            'rule_based_agent',
            'rule_based_agent',
            'rule_based_agent',
        ]
    }
}

if __name__ == '__main__':
    os.setpgrp()
    try:
        executor = ThreadPoolExecutor(max_workers=20)

        futures = []

        for agent in configurations:
            scenario = configurations[agent]['scenario']
            rounds = configurations[agent]['rounds']
            opponents = configurations[agent]['opponents']

            model_file = f'/tmp/{uuid4()}.npy'

            env = {
                'MODEL_FILE': model_file
            }

            command = f'/bin/bash -c "source venv/bin/activate && python3 main.py play --no-gui --scenario {scenario} --n-rounds {rounds} --agents {agent} {" ".join(opponents)} --train 1"'

            future = executor.submit(lambda: subprocess.call(command, shell=True, env=env))

            futures.append(future)

        for i, future in enumerate(futures):
            try:
                fitn, name = future.result()
            except Exception as exc:
                print("%r generated an exception: %s" % (list(configurations.keys())[i], exc))
    finally:
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
