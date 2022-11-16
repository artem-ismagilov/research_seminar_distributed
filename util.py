import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


def train_epoch(model, train_loader, optimizer, step_callback=None):
    loss_log, acc_log = [], []
    model.train()
    for batch_num, (x_batch, y_batch) in enumerate(train_loader):
        data = x_batch
        target = y_batch

        optimizer.zero_grad()
        output = model(data)
        pred = torch.max(output, 1)[1]
        acc = torch.eq(pred, y_batch).float().mean()
        acc_log.append(acc)

        loss = F.nll_loss(output, target).cpu()
        loss.backward()

        if step_callback is not None:
            step_callback(model)

        optimizer.step()

        loss = loss.item()
        loss_log.append(loss)

    return loss_log, acc_log


def test(model, test_loader):
    loss_log, acc_log = [], []
    model.eval()
    for batch_num, (x_batch, y_batch) in enumerate(test_loader):
        data = x_batch
        target = y_batch

        output = model(data)
        loss = F.nll_loss(output, target).cpu()

        pred = torch.max(output, 1)[1]
        acc = torch.eq(pred, y_batch).float().mean()
        acc_log.append(acc)

        loss = loss.item()
        loss_log.append(loss)

    return loss_log, acc_log


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.classifier = nn.Linear(4 * 4 * 16, 10)

    def forward(self, x):
        return F.log_softmax(self.classifier(self.features(x.reshape(-1, 1, 28, 28))), dim=-1)


def prepare_data():
    train_dataset = torchvision.datasets.MNIST(
            root='./MNIST/',
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True)

    test_dataset = torchvision.datasets.MNIST(
            root='./MNIST/',
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True)

    return train_dataset, test_dataset
