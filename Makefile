all:
	# run with: $$ python3 main.py play [options]

clean:
	rip ./agent_code/q_learning_task_1.npy ./agent_code/stats_q_learning_task_1.txt

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
