import os

for i in range(10):
    print(f'Train round {i}')
    os.system("venv/bin/python main.py play --scenario coin-heaven --agents q_learning_task_1  --no-gui --train 1")