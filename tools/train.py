import subprocess
from threading import Thread

configurations = {
    'q_learning_task_1': {
        'rounds': 10,
        'scenario': 'coin-hell',
        'opponents': []
    },
    'q_learning_task_1_function_learning': {
        'rounds': 3,
        'scenario': 'coin-hell',
        'opponents': []
    },
    'q_learning_task_1_nn': {
        'rounds': 50,
        'scenario': 'coin-hell',
        'opponents': []
    },
    'q_learning_task_1_nn_evolution': {
        'rounds': 50,
        'scenario': 'coin-hell',
        'opponents': []
    },
    'q_learning_task_2': {
        'rounds': 20,
        'scenario': 'few-crates',
        'opponents': []
    },
    'q_learning_task_2_function_learning': {
        'rounds': 300,
        'scenario': 'few-crates',
        'opponents': []
    },
    'q_learning_task_2_nn': {
        'rounds': 1000,
        'scenario': 'few-crates',
        'opponents': []
    },
    'q_learning_task_3': {
        'rounds': 50,
        'scenario': 'train-crates',
        'opponents': [
            'peaceful_agent',
            'peaceful_agent',
            'peaceful_agent'
        ]
    },
    'q_learning_task_3_advanced_features': {
        'rounds': 2000,
        'scenario': 'classic',
        'opponents': [
            'rule_based_agent',
            'rule_based_agent',
            'coin_collector_agent'
        ]
    },
    'q_learning_task_3_extended_feature_space': {
        'rounds': 1000,
        'scenario': 'classic',
        'opponents': [
            'rule_based_agent',
            'rule_based_agent',
            'coin_collector_agent'
        ]
    },
    'q_learning_task_3_function_learning': {
        'rounds': 1000,
        'scenario': 'few-crates',
        'opponents': [
            'peaceful_agent',
            'peaceful_agent',
            'peaceful_agent'
        ]
    },
    'q_learning_task_4_function_learning': {
        'rounds': 1000,
        'scenario': 'few-crates',
        'opponents': [
            'rule_based_agent',
            'rule_based_agent',
            'rule_based_agent',
        ]
    }
}


for agent in configurations:
    scenario = configurations[agent]['scenario']
    rounds = configurations[agent]['rounds']
    opponents = configurations[agent]['opponents']
    command = f'python3 main.py play --no-gui --scenario {scenario} --n-rounds {rounds} --agents {agent} {" ".join(opponents)} --train 1'

    Thread(target=lambda: subprocess.call(command, shell=True)).start()
