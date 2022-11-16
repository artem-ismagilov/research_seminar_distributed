# Distributed learning sandbox

### Setup

1. Create venv: `python3 -m venv .venv`
2. Activate venv: `source .venv/bin/activate`
3. Install requirements: `pip3 install -r requirements.txt`
4. Continue to experiments

### Experiments

1. `python3 00_run_simple_worker.py` –– just run two processes and print from them
2. `python3 01_run_point_to_point.py` –– send tensor from one worker to another
3. `python3 02_run_all_reduce.py` –– get sum of tensors from all workers and sync the result
4. `python3 03_train.py` –– simple implementation of distributed training with sync on every batch
5. `python3 04_train.py` –– example of `torch.nn.parallel.DistributedDataParallel` usage. It's the default choice for training on single machine? but multiple GPUs or different machines
