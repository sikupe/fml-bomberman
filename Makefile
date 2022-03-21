all:
	# run with: $$ python3 main.py play [options]


.PHONY: test
test:
	PYTHONPATH=. pytest

clean:
	rm ./agent_code/q_learning_task_1.npy ./agent_code/stats_q_learning_task_1.txt

format:
	black ./agent_code/strong_students/

lint:
	pylint ./agent_code/strong_students/

uninstall:
	rm -rf ./venv

venv:
	python -m venv ./venv

install: ./venv
	sh -c "source ./venv/bin/activate; pip install -r ./requirements.txt"

install-pytorch-rocr: ./venv
	sh -c "source ./venv/bin/activate; pip uninstall pytorch torch; pip install torch torchvision==0.11.3 -f https://download.pytorch.org/whl/rocm4.2/torch_stable.html"

train2:
	python main.py play --scenario few-crates --agents q_learning_task_2 --n-rounds 20 --train 1 --no-gui

train1nn:
	python main.py play --scenario coin-hell --agents q_learning_task_2_nn --n-rounds 20 --train 1 --no-gui

play2:
	python main.py play --scenario crate-hell --agents q_learning_task_2 --n-rounds 5
