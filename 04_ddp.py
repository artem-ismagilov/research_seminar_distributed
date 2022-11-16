from torch.nn.parallel import DistributedDataParallel as DDP
from process_group import Worker, DistributedRunner
import torch
import torch.distributed as D
from util import train_epoch, test, ConvNet, prepare_data
import numpy as np


class TrainWorker(Worker):
    def __init__(self, model, group_size):
        self.train_dataset, self.test_dataset = prepare_data()

        self.model = model
        self.group_size = group_size

    def __call__(self, id, n_processes):
        print(f'Process {id}')

        self.model = DDP(self.model)
        torch.set_num_threads(1)

        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        for i in range(10):
            print(f'Process {id}, epoch {i}')
            train_epoch(self.model, train_loader, optimizer)
            _, acc_log = test(self.model, test_loader)

            print('Process {}, acc {:.5f}'.format(id, np.mean(acc_log)))


if __name__ == '__main__':
    model = ConvNet()
    group_size = 6

    def spawn_worker(id):
        return TrainWorker(model, group_size)

    runner = DistributedRunner(
        n_processes=group_size,
        spawn_worker=spawn_worker,
        master_addr='127.0.0.1',
        master_port='29500',
    )

    runner()
