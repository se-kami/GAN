#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn

class Generator(nn.Module):
    def __init__(self, size_z=128, size_hid=128, size_out=32*32):
        super().__init__()
        # params
        self.size_z = size_z
        self.size_hid = size_hid
        self.size_out = size_out
        # net
        self.net = nn.Sequential(
                nn.Linear(self.size_z, self.size_hid),
                nn.LeakyReLU(),
                nn.Linear(self.size_hid, 2*self.size_hid),
                nn.LeakyReLU(),
                nn.Linear(2*self.size_hid, self.size_hid),
                nn.LeakyReLU(),
                nn.Linear(self.size_hid, self.size_out),
                nn.Tanh(),
                )

    def forward(self, x):
        x = self.net(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, size_in=32*32, size_hid=128):
        super().__init__()
        # params
        self.size_in = size_in
        self.size_hid = size_hid
        # net
        self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.size_in, self.size_hid),
                nn.LeakyReLU(),
                nn.Linear(self.size_hid, 2*self.size_hid),
                nn.LeakyReLU(),
                nn.Linear(2*self.size_hid, self.size_hid),
                nn.LeakyReLU(),
                nn.Linear(self.size_hid, 1),
                )

    def forward(self, x):
        x = self.net(x)
        return x
