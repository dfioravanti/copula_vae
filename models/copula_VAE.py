from models.basic_model import BaseVAE

import numpy as np

import torch
from torch import nn

from utils.nn import OneToOne, GatedDense
from utils.copula_sampling import sampling_from_gausiann_copula
from utils.distributions import gaussian_icdf


class BaseCopulaVAE(BaseVAE):

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 encoder_output_size=300,
                 device=torch.device("cpu")):

        super(BaseCopulaVAE, self).__init__(dimension_latent_space=dimension_latent_space,
                                        input_shape=input_shape,
                                        device=device)

    def forward(self, x):

        s_x, L_x = self.q_s(x)

        # x_mean = p(x|z)
        p_x_mean, p_x_var = self.p_x(s_x)
        x_recontructed = p_x_mean

        return x_recontructed, L_x, p_x_mean, p_x_var

    def q_s(self, x):

        batch_size = x.shape[0]

        L_x = self._compute_L_x(x, batch_size)

        s = sampling_from_gausiann_copula(L_x, batch_size, self.dimension_latent_space)

        return s, L_x

    def p_x(self, s):

        # F_l(s)

        mean_z, var_z = self.mean_z(s), self.var_z(s) + 1e-7
        z = gaussian_icdf(mean_z, var_z, s)

        F_z = self.F_x_layers(z)

        p_x_mean = self.p_x_mean(F_z)
        p_x_mean = torch.clamp(p_x_mean, min=0.+1./512., max=1.-1./512.)
        p_x_var = self.p_x_var(F_z) + 1e-7

        return p_x_mean, p_x_var

    def _compute_L_x(self, x, batch_size):

        idx_trn = np.tril_indices(self.dimension_latent_space)
        idx_diag = np.diag_indices(self.dimension_latent_space)
        idx_not_diag = np.tril_indices(self.dimension_latent_space, -1)

        L_x = torch.zeros([batch_size, self.dimension_latent_space, self.dimension_latent_space]).to(self.device)

        L_x[:, idx_trn[0], idx_trn[1]] = self.L_layers(x)
        L_x[:, idx_diag[0], idx_diag[1]] = torch.sigmoid(L_x[:, idx_diag[0], idx_diag[1]]) + 1e-5
        L_x[:, idx_not_diag[0], idx_not_diag[1]] = L_x[:, idx_not_diag[0], idx_not_diag[1]]

        return L_x


class CopulaVAEWithNormals(BaseCopulaVAE):

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 encoder_output_size=300,
                 device=torch.device("cpu")):

        super(CopulaVAEWithNormals, self).__init__(dimension_latent_space=dimension_latent_space,
                                                   input_shape=input_shape,
                                                   device=device)

        self.encoder_output_size = encoder_output_size

        self.number_neurons_L = dimension_latent_space * (dimension_latent_space+1) // 2

        # Encoder q(s | x)
        # L(x) without the final activtion function

        self.L_layers = nn.Sequential(
            nn.Linear(np.prod(self.input_shape), 300),
            nn.ReLU(),
            nn.Linear(300, self.number_neurons_L)
        )

        # Decoder p(x|s)

        # F_l(s) for now we assume that everything is gaussian

        self.mean_z = nn.Linear(self.dimension_latent_space, self.dimension_latent_space)
        self.var_z = nn.Sequential(
            nn.Linear(self.dimension_latent_space, self.dimension_latent_space),
            nn.Softplus()
        )

        # F(z)

        self.F_x_layers = nn.Sequential(
            nn.Linear(self.dimension_latent_space, 300),
            nn.ReLU()
        )
        self.p_x_mean = nn.Sequential(
            nn.Linear(300, np.prod(self.input_shape)),
            nn.Sigmoid(),
        )
        self.p_x_var = nn.Sequential(
            nn.Linear(300, np.prod(self.input_shape)),
            nn.Softplus()
        )

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


class CopulaVAE(BaseCopulaVAE):

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 encoder_output_size=300,
                 device=torch.device("cpu")):

        super(BaseCopulaVAE, self).__init__(dimension_latent_space=dimension_latent_space,
                                            input_shape=input_shape,
                                            device=device)

        self.encoder_output_size = encoder_output_size

        self.number_neurons_L = dimension_latent_space * (dimension_latent_space+1) // 2

        # Encoder q(s | x)
        # L(x) without the final activtion function

        self.L_layers = nn.Sequential(
            nn.Linear(np.prod(self.input_shape), 300),
            nn.ReLU(),
            nn.Linear(300, self.number_neurons_L)
        )

        # Decoder p(x|s)

        # F_l(s) for now we assume that everything is gaussian

        self.z_s_layers = nn.Sequential(
            OneToOne(self.dimension_latent_space),
            nn.ReLU(),
            OneToOne(self.dimension_latent_space),
            nn.ReLU(),
            OneToOne(self.dimension_latent_space),
            nn.ReLU(),
            OneToOne(self.dimension_latent_space),
            nn.ReLU(),
            OneToOne(self.dimension_latent_space)
        )

        # F(z)

        self.F_x_layers = nn.Sequential(
            nn.Linear(self.dimension_latent_space, 300),
            nn.ReLU(),
            nn.Linear(300, np.prod(self.input_shape)),
            nn.Sigmoid()
        )

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def p_x(self, s):

        # F_l(s)

        z = self.z_s_layers(s)

        return self.F_x_layers(z)