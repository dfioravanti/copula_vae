from models.BaseCopulaVAE import BaseCopulaVAE
import numpy as np

import torch
from torch import nn

from utils.nn import Flatten, Reshape, ICDF, PositiveLinear
from utils.utils_conv import compute_final_convolution_shape, build_convolutional_blocks,\
                             compute_final_deconv_shape, build_deconvolutional_blocks

from utils.inverse_distibutions import gaussian_icdf

class MarginalVAE(BaseCopulaVAE):

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 dataset_type,
                 encoder_output_size=300,
                 marginals='gaussian',
                 device=torch.device("cpu")):

        super(MarginalVAE, self).__init__(dimension_latent_space=dimension_latent_space,
                                          input_shape=input_shape,
                                          dataset_type=dataset_type,
                                          device=device)

        self.encoder_output_size = encoder_output_size
        self.marginals = marginals
        self.number_neurons_L = dimension_latent_space * (dimension_latent_space+1) // 2

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

        decoder_in_shape = (1, 7, 7)
        number_blocks_deconv = 2
        deconv_output_shape = compute_final_deconv_shape(decoder_in_shape[1], decoder_in_shape[2],
                                                         number_blocks=number_blocks_deconv, kernel_size=kernel_size)
        dim_decov_out = np.prod(deconv_output_shape)

        self.F_x_layers = nn.Sequential(
            nn.Linear(self.dimension_latent_space, np.prod(decoder_in_shape)),
            nn.ReLU(),
        )
        self.F_x_layers.add_module('reshape', Reshape(decoder_in_shape))
        self.F_x_layers.add_module('deconv', build_deconvolutional_blocks(number_blocks_deconv,
                                                                          nb_channels_in=decoder_in_shape[0],
                                                                          nb_channels=nb_channel,
                                                                          kernel_size=kernel_size))

        self.F_x_layers.add_module('flatten_out', Flatten())
        self.F_x_layers.add_module('dense_out', nn.Linear(nb_channel * dim_decov_out, np.prod(self.input_shape)))
        self.F_x_layers.add_module('sigmoid_out', nn.Sigmoid())

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
                 dataset_type,
                 encoder_output_size=300,
                 device=torch.device("cpu")):

        super(CopulaVAE, self).__init__(dimension_latent_space=dimension_latent_space,
                                        input_shape=input_shape,
                                        dataset_type=dataset_type,
                                            device=device)

        self.encoder_output_size = encoder_output_size
        self.number_neurons_L = dimension_latent_space * (dimension_latent_space+1) // 2

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
            PositiveLinear(self.dimension_latent_space, 300),
            nn.ReLU(),
            PositiveLinear(300, 300),
            nn.ReLU(),
            PositiveLinear(300, self.dimension_latent_space)
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
