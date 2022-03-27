#!/usr/bin/env sh
MODEL_FILE=/home/mulc/repos/bomberman/fml-bomberman/agent_code/q_learning_task_3_advanced_features/blobs/5000x8merged_model.npy
COMMON_DIR=/home/mulc/repos/bomberman/save-bomb/agent_code/common/
AGENT_DIR=/home/mulc/repos/bomberman/save-bomb/agent_code/q_learning_task_3_advanced_features/

AGENT_NAME=q_learning_task_3_advanced_features

pushd /tmp
git clone https://github.com/ukoethe/bomberman_rl final_bomberman
pushd final_bomberman
pushd agent_code

cp -r $AGENT_DIR strong_students
pushd strong_students

cp $MODEL_FILE ./model.npy

sed -i "s/\.$AGENT_NAME/\.strong_students/g" *.py

cp -r $COMMON_DIR common

sed -i 's/common/strong_students\.common/g' **/*.py

popd
popd

podman build -t test_strong_students .

podman run -i test_strong_students bash -c 'python3 main.py play --agents strong_students'
