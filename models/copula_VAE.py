from models.basic_model import BaseVAE

import numpy as np


import torch
from torch import nn

from utils.nn import GatedDense
from utils import copula_sampling
from utils.distributions import gaussian_icdf, log_normal_standard, log_normal_by_component


class CopulaVAEWithNormals(BaseVAE):

    def __init__(self,
                 number_latent_variables,
                 input_shape,
                 encoder_output_size=300,
                 device=torch.device("cpu")):

        super(CopulaVAEWithNormals, self).__init__(number_latent_variables=number_latent_variables,
                                                   input_shape=input_shape,
                                                   device=device)

        self.encoder_output_size = encoder_output_size

        # Encoder q(z|x)

        self.q_z_layers = nn.Sequential(
            GatedDense(np.prod(self.input_shape), 300),
            GatedDense(300, self.encoder_output_size)
        )

        self.mean = nn.Linear(self.encoder_output_size, self.number_latent_variables)
        self.log_var = nn.Sequential(
            nn.Linear(self.encoder_output_size, self.number_latent_variables),
            nn.Hardtanh(min_val=-6, max_val=2)
        )

        # Decoder p(x|z)

        self.p_x_layers = nn.Sequential(
            GatedDense(self.number_latent_variables, 300),
            GatedDense(300, 300)
        )

        self.output_decoder = nn.Sequential(
            nn.Linear(300, np.prod(self.input_shape)),
            nn.Sigmoid()
        )

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        mean_z_x, log_var_z_x = self.q_z(x)
        z_x = sampling_copula_with_normal_inverse(mean_z_x, log_var_z_x, self.number_latent_variables)

        # x_mean = p(x|z)
        x_recontructed = self.p_x(z_x)

        return x_recontructed, mean_z_x, log_var_z_x

    def q_z(self, x):
        x = self.q_z_layers(x)

        return self.mean(x), self.log_var(x)

    def p_x(self, z):
        z = self.p_x_layers(z)

        return self.output_decoder(z)


def sampling_copula_with_normal_inverse(means, log_vars, number_latent_variables, size=1):

    cov = np.eye(number_latent_variables)
    xs = torch.FloatTensor(copula_sampling.sampling_from_gausiann_copula(cov, size))

    if torch.cuda.is_available():
        xs = xs.to(means.get_device())

    return gaussian_icdf(means, log_vars, xs)


def compute_KL(means, log_vars, number_latent_variables, number_samples_kl):

    zs = sampling_copula_with_normal_inverse(means, log_vars, number_latent_variables, number_samples_kl)

    log_ps = log_normal_standard(zs)
    log_qs = log_normal_by_component(zs, means, log_vars)

    return torch.mean(log_ps - log_qs)
