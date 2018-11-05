
import pathlib

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.copula_VAE import CopulaVAEWithNormals
from utils.copula_sampling import sampling_from_gausiann_copula
from loaders.load_funtions import load_MNIST


def explore_latent_space(model, s_dim, input_shape):

    train_loader, _, _ = load_MNIST(shuffle=False)

    original_x = next(iter(train_loader))[0][0]
    original_x = original_x.view(original_x.numel())

    L = model.compute_L_x(original_x, 1)
    s = sampling_from_gausiann_copula(L, 1, s_dim)
    for alpha in [1]:
        s = s + alpha * torch.eye(s_dim))

        xs = model.p_x(s).reshape((-1, ) + input_shape).squeeze().detach().numpy()

        for i, x in enumerate(xs):

            plt.imshow(x)
            plt.savefig(f'img/{alpha}_{i}.png')


if __name__ == '__main__':

    path = pathlib.Path('outputs') / 'L2_Mnist' / 'model'
    input_shape = (1, 28, 28)
    s_dim = 40

    model = CopulaVAEWithNormals(dimension_latent_space=s_dim, input_shape=input_shape)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    explore_latent_space(model, s_dim, input_shape)