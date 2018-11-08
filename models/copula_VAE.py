from models.basic_model import BaseVAE

import numpy as np

import torch
from torch import nn
from utils.nn import OneToOne, GatedDense, Flatten
from utils.copula_sampling import sampling_from_gausiann_copula
from utils.distributions import gaussian_icdf
from utils.utils_conv import compute_final_convolution_shape, build_convolutional_blocks


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
        x_recontructed = self.p_x(s_x)

        return x_recontructed, L_x

    def q_s(self, x):

        batch_size = x.shape[0]

        L_x = self.compute_L_x(x, batch_size)

        s = sampling_from_gausiann_copula(L_x, batch_size, self.dimension_latent_space)

        return s, L_x

    def p_x(self, s):

        # F_l(s)

        mean_z, var_z = self.mean_z(s), self.var_z(s) + 1e-5
        z = gaussian_icdf(mean_z, var_z, s)

        return self.F_x_layers(z)

    def compute_L_x(self, x, batch_size):

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
            nn.ReLU(),
            nn.Linear(300, np.prod(self.input_shape)),
            nn.Sigmoid()
        )

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


class CopulaVAEWithNormalsConvDecoder(BaseCopulaVAE):

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 encoder_output_size=300,
                 device=torch.device("cpu")):

        super(CopulaVAEWithNormalsConvDecoder, self).__init__(dimension_latent_space=dimension_latent_space,
                                                              input_shape=input_shape,
                                                              device=device)

        # Variables

        nb_channel_in = self.input_shape[0]
        self.encoder_output_size = encoder_output_size
        self.number_neurons_L = dimension_latent_space * (dimension_latent_space+1) // 2

        kernel_size = 4
        number_blocks = 2
        nb_channel = 32
        maxpolling = True
        conv_output_shape = compute_final_convolution_shape(self.input_shape[1], self.input_shape[2],
                                                            number_blocks, maxpolling, kernel_size)

        # Encoder q(s | x)
        # L(x) without the final activation function

        self.L_layers = build_convolutional_blocks(number_blocks, nb_channel_in, nb_channel, maxpolling, kernel_size)
        self.L_layers.add_module('flatten_0', Flatten())
        self.L_layers.add_module('linear_0',
                                 nn.Linear(nb_channel * conv_output_shape[0] * conv_output_shape[1],
                                           self.number_neurons_L))


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
            nn.ReLU(),
            nn.Linear(300, np.prod(self.input_shape)),
            nn.Sigmoid()
        )

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def compute_L_x(self, x, batch_size):

        x = x.view((-1, ) + self.input_shape)
        return super(CopulaVAEWithNormalsConvDecoder, self).compute_L_x(x, batch_size)


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
            nn.Linear(300, 300),
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