import numpy as np
import torch
from torch import nn

from models.BaseCopulaVAE import BaseCopulaVAE, BaseDiagonalCopulaVAE
from utils.nn import View, ICDF


class ConvMarginalVAE(BaseCopulaVAE):

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 dataset_type,
                 marginals='gaussian',
                 device=torch.device("cpu")):
        super(ConvMarginalVAE, self).__init__(dimension_latent_space=dimension_latent_space,
                                              input_shape=input_shape,
                                              dataset_type=dataset_type,
                                              device=device)

        # Variables

        self.marginals = marginals
        nb_channel_in = self.input_shape[0]

        # Encoder q(s | x)
        # L(x) without the final activation function

        self.L_layers = nn.Sequential(
            nn.Conv2d(nb_channel_in, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
            nn.ReLU(True),
            View((256 * 1 * 1,)),  # B, 256
            nn.Linear(256, self.number_neurons_L),  # B, z_dim
        )

        # Decoder p(x|s)

        # F_l(s) for now we assume that everything is gaussian

        self.F = ICDF(self.dimension_latent_space, marginals=self.marginals)

        # F(z)

        self.p_x_mean = nn.Sequential(
            nn.Linear(dimension_latent_space, 256),  # B, 256
            View((256, 1, 1)),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nb_channel_in, 4, 2, 1),  # B, nc, 64, 64,
            nn.Sigmoid()
        )

        # TODO: Fix the variance stuff

    def compute_L_x(self, x, batch_size):
        x = x.view((-1,) + self.input_shape)
        return super(ConvMarginalVAE, self).compute_L_x(x, batch_size)

    def p_z(self, s):
        return self.F(s)

    def p_x(self, s):
        z = self.p_z(s)
        x = self.p_x_mean(z)

        return x.view((-1, np.prod(self.input_shape))), None


class ConvDiagonalMarginalVAE(BaseDiagonalCopulaVAE):

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 dataset_type,
                 marginals='gaussian',
                 device=torch.device("cpu")):
        super(ConvDiagonalMarginalVAE, self).__init__(dimension_latent_space=dimension_latent_space,
                                                      input_shape=input_shape,
                                                      dataset_type=dataset_type,
                                                      device=device)

        # Variables

        self.marginals = marginals
        nb_channel_in = self.input_shape[0]

        # Encoder q(s | x)
        # L(x) without the final activation function

        self.L_layers = nn.Sequential(
            nn.Conv2d(nb_channel_in, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
            nn.ReLU(True),
            View((256 * 1 * 1,)),  # B, 256
            nn.Linear(256, self.number_neurons_L),  # B, z_dim
        )

        # Decoder p(x|s)

        # F_l(s) for now we assume that everything is gaussian

        self.F = ICDF(self.dimension_latent_space, marginals=self.marginals)

        # F(z)

        self.p_x_mean = nn.Sequential(
            nn.Linear(dimension_latent_space, 256),  # B, 256
            View((256, 1, 1)),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nb_channel_in, 4, 2, 1),  # B, nc, 64, 64,
            nn.Sigmoid()
        )

        # TODO: Fix the variance stuff

    def compute_L_x(self, x, batch_size):
        x = x.view((-1,) + self.input_shape)
        return super(ConvDiagonalMarginalVAE, self).compute_L_x(x, batch_size)

    def p_z(self, s):
        return self.F(s)

    def p_x(self, s):
        z = self.p_z(s)
        x = self.p_x_mean(z)

        return x.view((-1, np.prod(self.input_shape))), None
