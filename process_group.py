import os
import torch
import torch.distributed as D
import torch.multiprocessing as mp


class Worker:
    def __call__(self, id, n_processes):
        raise NotImplementedError()


def _start_process(id, n_processes, worker, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # os.environ['GLOO_SOCKET_IFNAME'] = 'en0'
    D.init_process_group('gloo', rank=id, world_size=n_processes)

    worker(id, n_processes)


class DistributedRunner:
    def __init__(self, n_processes, spawn_worker, master_addr, master_port):
        self.n_processes = n_processes
        self.spawn_worker = spawn_worker
        self.master_addr = master_addr
        self.master_port = master_port

    def __call__(self):
        mp.set_start_method("spawn")

        processes = []
        for id in range(self.n_processes):
            processes.append(
                mp.Process(
                    target=_start_process,
                    args=(id, self.n_processes, self.spawn_worker(id), self.master_addr, self.master_port)
                )
            )
            processes[-1].start()

        for p in processes:
            p.join()
