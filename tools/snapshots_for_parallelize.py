#!/usr/bin/env python3

import subprocess
import os
import shutil
from os.path import join, abspath, exists
from time import sleep
from tqdm import tqdm

count = 18
training_rounds = 1000

res = subprocess.run(
    "git rev-parse --show-toplevel", shell=True, stdout=subprocess.PIPE
)
assert res.returncode == 0


repo_dir = abspath(res.stdout.decode().strip())
tools_dir = join(repo_dir, "tools")
train_script = join(tools_dir, "fml.sh")
merge_script = join(tools_dir, "merge.py")

tmp_dir = "/tmp/train_q_learning_task_3_advanced_features"
snapshots_dir = join(tools_dir, "snapshots")

while True:
    res = subprocess.run(
        "tmux ls | rg train | awk '{print$2}'", shell=True, stdout=subprocess.PIPE
    )
    assert res.returncode == 0
    res = res.stdout.decode()

    # No running train session, we are good to go
    if res == "":
        break

    if int(res) != 1:
        print("train might still be running, window count is not 1")
        sleep(2)
        continue

    break


for d in (tmp_dir, snapshots_dir):
    if exists(d):
        while True:
            print(f"{d} exists, want to clear y/n?")
            user_input = input("")
            if user_input.lower() == "y":
                shutil.rmtree(d)
                assert not exists(d)
                break
            if user_input.lower() == "n":
                assert exists(d)
                break

os.chdir(tools_dir)

for i in tqdm(range(1, 11)):

    res = subprocess.run(
        f'sh {train_script} "q_learning_task_3_advanced_features rule_based_agent rule_based_agent coin_collector_agent" classic {training_rounds} {count-1}',
        shell=True,
        stdout=subprocess.PIPE,
    )

    sleep(10)

    while True:
        res = subprocess.run(
            "tmux ls | rg train | awk '{print$2}'", shell=True, stdout=subprocess.PIPE
        )
        assert res.returncode == 0
        if int(res.stdout.decode()) == 1:
            train_dir = f"{snapshots_dir}/train-{i*training_rounds}"
            merged_model_file = f"{snapshots_dir}/train-{i*training_rounds}/{i*training_rounds}_merged_model.npy"

            shutil.copytree(tmp_dir, train_dir)
            res = subprocess.run(
                f"python3 {merge_script} {train_dir} {merged_model_file}",
                shell=True,
                stdout=subprocess.PIPE,
            )
            assert res.returncode == 0
            print(res.stdout.decode())
            break
        sleep(30)
