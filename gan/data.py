#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

def get_dataset(root='_DATA', transform=None, train=True):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
            ])
    ds = datasets.MNIST(root=root, download=True, train=train,
                              transform=transform)
    return ds


def get_loader(batch_size=32, transform=None, train=True, shuffle=True):
    ds = get_dataset(transform=transform, train=train)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_test_images(examples_per_class=10, classes=list(range(10))):
    loader = get_loader(train=False, shuffle=False)
    examples = [[] for _ in classes]

    for x, y in loader:
        to_cont = False
        for i, c in enumerate(classes):
            if len(examples[i]) < examples_per_class:
                examples[i] += list(x[y == c])
                to_cont = True
        if not to_cont:
            break

    examples = [torch.cat(l[:examples_per_class]) for l in examples]
    examples = torch.cat(examples)
    return examples


class NoiseMaker():
    def __init__(self, size_z):
        self.size_z = size_z

    def __call__(self, batch_size):
        return torch.randn(batch_size, self.size_z)
