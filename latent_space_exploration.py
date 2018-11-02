
import pathlib

import torch

from models.copula_VAE import CopulaVAEWithNormals


def explore_latent_space(model):

    pass

if __name__ == '__main__':

    path = pathlib.Path('outputs') / 'L2_Mnist' / 'model'
    input_shape = (1, 28, 28)
    s_dim = 40

    model = CopulaVAEWithNormals(dimension_latent_space=s_dim, input_shape=input_shape)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    explore_latent_space(model)