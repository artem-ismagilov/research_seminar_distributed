from process_group import Worker, DistributedRunner
import torch
import torch.distributed as D
import time


class AllReduceWorker(Worker):
    def __init__(self):
        pass

    def __call__(self, id, n_processes):
        if id == 0:
            time.sleep(5)

        data = torch.ones(5)
        print(f'Process {id} got data. Reducing...')
        D.all_reduce(data, op=D.ReduceOp.SUM, group=D.new_group([0, 1]))

        print(f'Process {id} data: {data}!')


if __name__ == '__main__':
    def spawn_worker(id):
        return AllReduceWorker()

    runner = DistributedRunner(
        n_processes=2,
        spawn_worker=spawn_worker,
        master_addr='127.0.0.1',
        master_port='29500',
    )

    runner()
