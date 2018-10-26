from models.basic_model import BaseVAE

import numpy as np
import torch
from torch import nn
from utils.nn import GatedDense


class VAE(BaseVAE):

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 encoder_output_size=300,
                 device=torch.device("cpu")):

        super(VAE, self).__init__(dimension_latent_space=dimension_latent_space,
                                  input_shape=input_shape,
                                  device=device)

        self.encoder_output_size = encoder_output_size

        # Encoder q(z|x)

        self.q_z_layers = nn.Sequential(
            GatedDense(np.prod(self.input_shape), 300),
            GatedDense(300, self.encoder_output_size)
        )

        self.mean = nn.Linear(self.encoder_output_size, self.dimension_latent_space)
        self.log_var = nn.Sequential(
            nn.Linear(self.encoder_output_size, self.dimension_latent_space),
            nn.Hardtanh(min_val=-6, max_val=2)
        )

        # Decoder p(x|z)

        self.p_x_layers = nn.Sequential(
            GatedDense(self.dimension_latent_space, 300),
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
        z_x = self.sampling_normal_with_reparametrization(mean_z_x, log_var_z_x)

        # x_mean = p(x|z)
        x_recontructed = self.p_x(z_x)

        return x_recontructed, mean_z_x, log_var_z_x

    def q_z(self, x):
        x = self.q_z_layers(x)

        return self.mean(x), self.log_var(x)

    def p_x(self, z):
        z = self.p_x_layers(z)

        return self.output_decoder(z)

