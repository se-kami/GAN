#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from tqdm import tqdm
from model import Discriminator
from model import Generator
from data import get_loader
from data import NoiseMaker
from data import get_test_images
from torch.optim import Adam
from time import time
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage


def train(model_d, model_g, opt_d, opt_g, loader, noise_maker, epochs, device,
          loss_fn, dir_exp, noise_to_viz=None):

    for epoch in tqdm(range(epochs)):
        loss_d = 0.0
        loss_g = 0.0
        total_correct = 0
        total = 0
        for index, (x, _) in enumerate(loader):
            total += x.shape[0]
            model_g.train()
            # move to device
            x = x.to(device)
            # optimize discriminator
            # discriminator on real images
            prediction_d = model_d(x).view(-1)
            loss_d_real = loss_fn(prediction_d, torch.ones_like(prediction_d))
            total_correct += (prediction_d > 0).long().sum().item()
            # make noise
            noise = noise_maker(x.shape[0])
            noise = noise.to(device)
            # generate from noise
            x_gen = model_g(noise)
            # discriminator on generated images
            prediction_d = model_d(x_gen.detach()).view(-1)
            loss_d_gen = loss_fn(prediction_d, torch.zeros_like(prediction_d))
            total_correct += (prediction_d > 0).long().sum().item()
            # total loss
            loss = 0.5 * loss_d_real + 0.5 * loss_d_gen
            loss_d += loss.item() * x.shape[0]
            # backward step
            opt_d.zero_grad()
            loss.backward()
            opt_d.step()

            # optimize generator
            prediction_d = model_d(x_gen).view(-1)
            loss_d_gen = loss_fn(prediction_d, torch.ones_like(prediction_d))
            loss = 1.0 * loss_d_gen
            loss_g += loss.item() * x.shape[0]
            # backward step
            opt_g.zero_grad()
            loss.backward()
            opt_g.step()

            # save images
            if index == 0 and noise_to_viz is not None:
                model_g.eval()
                with torch.no_grad():
                    x_gen = model_g(noise_to_viz)
                    size = int(x_gen[0].shape[0] ** 0.5)  # img side length
                    x_gen = x_gen.view(x_gen.shape[0], 1, size, size)
                    grid = make_grid(x_gen, nrow=4, normalize=True)
                    img = ToPILImage()(grid)
                    filename = f'image_{epoch:04d}.png'
                    filename = os.path.join(dir_exp, filename)
                    img.save(filename)

        postfix = f"{epoch:03d}: "
        postfix += f"[Loss D: {loss_d/total:.2f}]"
        postfix += f"[Loss G: {loss_g/total:.2f}]"
        postfix += f"[Domain accuracy: {total_correct/total/2:.2f}]"
        print(postfix)


if __name__ == '__main__':
    runs_dir = 'runs'
    # run name
    exp_name = str(time()).split('.')[0]
    dir_exp = os.path.join(runs_dir, exp_name)
    # make dirs
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(dir_exp)


    # training params
    lr = 3e-4
    momentum = 0.9
    epochs = 128
    batch_size = 512
    # model params
    size_in = 28*28
    size_z = 128
    size_hid = 128
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # models
    discriminator = Discriminator(size_in=size_in, size_hid=size_hid)
    generator = Generator(size_z=size_z, size_hid=size_hid, size_out=size_in)
    # move to device
    discriminator = discriminator.to(device)
    generator = generator.to(device)
    # optimizers
    opt_d = Adam(discriminator.parameters(), lr=lr)
    opt_g = Adam(generator.parameters(), lr=lr)
    # loss
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # noise
    examples_to_viz = 20
    noise_to_viz = torch.randn((examples_to_viz, size_z), device=device)
    noise_maker = NoiseMaker(size_z)
    # loaders
    loader = get_loader(batch_size=batch_size)

    # run training
    train(discriminator, generator, opt_d, opt_g, loader, noise_maker, epochs, device, loss_fn, dir_exp, noise_to_viz)
