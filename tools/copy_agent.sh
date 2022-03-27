#!/usr/bin/env bash
podman ps
if [ $? -eq 0 ];then
    CONTAINER_CMD=podman
else
    CONTAINER_CMD=docker
fi

MODEL_FILE=/home/mulc/repos/bomberman/fml-bomberman/agent_code/q_learning_task_3_advanced_features/blobs/5000x8merged_model.npy

REPO_DIR=/home/mulc/repos/bomberman/save-bomb/
AGENT_NAME=q_learning_task_3_advanced_features

COMMON_DIR=$REPO_DIR/agent_code/common/
AGENT_DIR=$REPO_DIR/agent_code/$AGENT_NAME

pushd /tmp
# If the repo exists reclone it
if [ -d ./final_bomberman ];then
    rm -rf ./final_bomberman
fi
git clone https://github.com/ukoethe/bomberman_rl final_bomberman
pushd final_bomberman

# Apply patch to see result with --no-gui
git apply <<EOF
diff --git a/environment.py b/environment.py
index 0ec3726..6d094a7 100644
--- a/environment.py
+++ b/environment.py
@@ -507,6 +507,7 @@ class BombeRLeWorld(GenericWorld):
         for a in self.agents:
             # Send exit message to shut down agent
             self.logger.debug(f'Sending exit message to agent <{a.name}>')
+            print(f"{a.name}: {a.total_score}")
             # todo multiprocessing shutdown
 
 
EOF

pushd agent_code

cp -r $AGENT_DIR strong_students
pushd strong_students

# Remove temp and bin files of dev repos
rm *.npy
rm *.txt
rm -rf blobs
rm rewards.json
rm logs/*.log
rm *.csv

cp $MODEL_FILE ./model.npy

sed -i "s/\.$AGENT_NAME/\.strong_students/g" *.py

cp -r $COMMON_DIR common

# sed in strong_students
sed -i 's/common/strong_students\.common/g' *.py

# sed in strong_students/common
pushd common
sed -i 's/common/strong_students\.common/g' *.py
popd

popd
popd

$CONTAINER_CMD build -t test_strong_students .

$CONTAINER_CMD run -i test_strong_students bash -c 'python3 main.py play --agents strong_students --no-gui'
