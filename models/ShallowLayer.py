import numpy as np

import torch
from torch import nn

from models.BaseCopulaVAE import BaseCopulaVAE, BaseDiagonalCopulaVAE
from utils.nn import ICDF


class ShallowMarginalVAE(BaseCopulaVAE):

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 dataset_type,
                 marginals='gaussian',
                 device=torch.device("cpu")):

        super(ShallowMarginalVAE, self).__init__(dimension_latent_space=dimension_latent_space,
                                                 input_shape=input_shape,
                                                 dataset_type=dataset_type,
                                                 device=device)

        self.marginals = marginals

        # Encoder q(s | x)
        # L(x) without the final activation function

        self.L_layers = nn.Sequential(
            nn.Linear(np.prod(self.input_shape), 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, self.number_neurons_L)
        )

        # Decoder p(x|s)

        # F(z)

        self.F = ICDF(self.dimension_latent_space, marginals=self.marginals)

        self.p_x_layers = nn.Sequential(
            nn.Linear(self.dimension_latent_space, 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Tanh(),
        )

        self.p_x_mean = nn.Sequential(
            nn.Linear(300, np.prod(self.input_shape)),
            nn.Sigmoid()
        )

        if not dataset_type == 'binary':
            self.p_x_log_var = nn.Linear(300, np.prod(self.input_shape))

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def p_z(self, s):

        return self.F(s)


class ShallowCopulaVAE(BaseCopulaVAE):

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 dataset_type,
                 device=torch.device("cpu")):

        super(ShallowCopulaVAE, self).__init__(dimension_latent_space=dimension_latent_space,
                                               input_shape=input_shape,
                                               dataset_type=dataset_type,
                                               device=device)

        # Encoder q(s | x)
        # L(x) without the final activation function

        self.L_layers = nn.Sequential(
            nn.Linear(np.prod(self.input_shape), 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, self.number_neurons_L)
        )

        # Decoder p(x|s)

        # F(z)

        self.F_s = nn.Sequential(
            nn.Linear(self.dimension_latent_space, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, self.dimension_latent_space)
        )

        self.p_x_layers = nn.Sequential(
            nn.Linear(self.dimension_latent_space, 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Tanh(),
        )

        self.p_x_mean = nn.Sequential(
            nn.Linear(300, np.prod(self.input_shape)),
            nn.Sigmoid()
        )

        if not dataset_type == 'binary':
            self.p_x_log_var = nn.Linear(300, np.prod(self.input_shape))

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def p_z(self, s):

        return self.F_s(s)


class ShallowDiagonalMarginalVAE(BaseDiagonalCopulaVAE):

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 dataset_type,
                 encoder_output_size=300,
                 marginals='gaussian',
                 device=torch.device("cpu")):

        super(ShallowDiagonalMarginalVAE, self).__init__(dimension_latent_space=dimension_latent_space,
                                                         input_shape=input_shape,
                                                         dataset_type=dataset_type,
                                                         device=device)

        self.marginals = marginals

        # Encoder q(s | x)
        # L(x) without the final activation function

        self.L_layers = nn.Sequential(
            nn.Linear(np.prod(self.input_shape), 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, self.number_neurons_L)
        )

        # Decoder p(x|s)

        # F(z)

        self.F = ICDF(self.dimension_latent_space, marginals=self.marginals)

        self.p_x_layers = nn.Sequential(
            nn.Linear(self.dimension_latent_space, 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Tanh(),
        )

        self.p_x_mean = nn.Sequential(
            nn.Linear(300, np.prod(self.input_shape)),
            nn.Sigmoid()
        )

        if not dataset_type == 'binary':
            self.p_x_log_var = nn.Linear(300, np.prod(self.input_shape))

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def p_z(self, s):

        return self.F(s)
