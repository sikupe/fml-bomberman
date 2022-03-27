#!/usr/bin/env bash
REPO_DIR=$(git rev-parse --show-toplevel)

if [ "$1" == "help" ] || [ "$1" == "-h" ] || [ "$1" == "--help" ] || [ -z "$1" ] ;then
    echo sh fml.sh AGENTS SCENARIO ROUNDS [count] [random_dir?]
    exit
fi

if [ ! -z "$1" ];then
    AGENTS=$1
else
    AGENTS="q_learning_task_3_advanced_features"
fi

if [ ! -z "$2" ];then
    SCENARIO=$2
else
    SCENARIO="coin-hell"
fi

if [ ! -z "$3" ];then
    ROUNDS=$3
else
    ROUNDS=100
fi

if [ ! -z "$4" ];then
    COUNT=$4
else
    COUNT=20
fi

if [ ! -z "$5" ];then
    RANDOM=false
else
    RANDOM=true
fi

if [ ${RANDOM} = true ];then
    TMP_DIR=$(mktemp -d -u)
    echo random
else
    TMP_DIR="/tmp/train"
    echo not random
fi

TMP_DIR_ADDITION=$(echo ${AGENTS} | sed 's/\s.*$//')
mkdir -p ${TMP_DIR}_${TMP_DIR_ADDITION}/agents_blobs

tmux ls | grep train
if [ $? -ne 0 ];then
    echo Create new tmux session
    tmux new -d -s train
fi

pushd ${REPO_DIR}
if [ ! -f "main.py" ];then
    echo Seems like we are not in the right directory. Exiting.
    exit
fi

if [ ${TMP_DIR_ADDITION} = *"_nn" ];then
    EXTENSION=.pt
else
    EXTENSION=.npy
fi


for i in $( seq 0  ${COUNT} );do
    tmux neww -t train
    tmux send -t train "pushd ${REPO_DIR}" ENTER
    tmux send -t train "source venv/bin/activate" ENTER
    tmux send -t train "export MODEL_FILE=${TMP_DIR}_${TMP_DIR_ADDITION}/agents_blobs/${TMP_DIR_ADDITION}_${i}${EXTENSION}" ENTER
    tmux send -t train "export STATS_FILE=${TMP_DIR}_${TMP_DIR_ADDITION}/agents_blobs/stats_${TMP_DIR_ADDITION}_${i}.txt" ENTER
    tmux send -t train "python main.py play --scenario ${SCENARIO} --agents ${AGENTS} --n-rounds ${ROUNDS} --train 1 --no-gui" ENTER
    tmux send -t train "exit" ENTER
    echo $i
done

echo DIR: ${TMP_DIR}_${TMP_DIR_ADDITION}
