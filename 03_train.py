from process_group import Worker, DistributedRunner
import torch
import torch.distributed as D
import torchvision
import numpy as np
from util import train_epoch, test, ConvNet


class TrainWorker(Worker):
    def __init__(self, model, group_size):
        self.train_dataset = torchvision.datasets.MNIST(
            root='./MNIST/',
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True)

        self.test_dataset = torchvision.datasets.MNIST(
            root='./MNIST/',
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True)

        self.model = model
        self.group_size = group_size

    def __call__(self, id, n_processes):
        torch.set_num_threads(1)

        def step_callback(model):
            for p in model.parameters():
                D.all_reduce(p.grad.data, op=D.ReduceOp.SUM)
                p.grad.data / self.group_size

        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        for i in range(10):
            print(f'Process {id}, epoch {i}')
            train_epoch(self.model, train_loader, optimizer, step_callback)
            _, acc_log = test(self.model, test_loader)

            print('Process {}, acc {:.5f}'.format(id, np.mean(acc_log)))


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

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
