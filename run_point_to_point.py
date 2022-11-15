from process_group import Worker, DistributedRunner
import torch
import torch.distributed as D
import time


class PointToPointWorker(Worker):
    def __init__(self):
        pass

    def __call__(self, id, n_processes):
        if id == 0:
            data = torch.ones(5)
            D.send(tensor=data, dst=1)
        else:
            data = torch.tensor(0)
            D.recv(tensor=data, src=0)

        print(f'Process {id}: data {data}!')


if __name__ == '__main__':
    def spawn_worker(id):
        return PointToPointWorker()

    runner = DistributedRunner(
        n_processes=2,
        spawn_worker=spawn_worker,
        master_addr='127.0.0.1',
        master_port='29500',
    )

    runner()
