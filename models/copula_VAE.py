from models.basic_model import BaseVAE

import numpy as np

import torch
from torch import nn

from utils.nn import OneToOne, Flatten, Reshape, GatedDense
from utils.copula_sampling import sampling_from_gausiann_copula
from utils.distributions import gaussian_0_I_icdf, log_density_Bernoulli, log_density_discretized_Logistic
from utils.utils_conv import compute_final_convolution_shape, build_convolutional_blocks,\
                             compute_final_deconv_shape, build_deconvolutional_blocks


class BaseCopulaVAE(BaseVAE):

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 dataset_type,
                 encoder_output_size=300,
                 device=torch.device("cpu")):

        super(BaseCopulaVAE, self).__init__(dimension_latent_space=dimension_latent_space,
                                            input_shape=input_shape,
                                            dataset_type=dataset_type,
                                            device=device)

    def forward(self, x):

        s_x, L_x = self.q_s(x)

        # x_mean = p(x|z)
        x_mean, x_log_var = self.p_x(s_x)

        return x_mean, x_log_var, L_x

    def q_s(self, x):

        batch_size = x.shape[0]
        L_x = self.compute_L_x(x, batch_size)

        s = sampling_from_gausiann_copula(L_x, batch_size, self.dimension_latent_space)

        return s, L_x

    def p_x(self, s):

        z = self.p_z(s)
        z = self.F_x_layers(z)

        if self.dataset_type == 'binary':
            return self.p_x_mean(z), None
        else:
            return self.p_x_mean(z), self.p_x_log_var(z)

    def compute_L_x(self, x, batch_size):

        idx_trn = np.tril_indices(self.dimension_latent_space)
        idx_diag = np.diag_indices(self.dimension_latent_space)
        idx_not_diag = np.tril_indices(self.dimension_latent_space, -1)

        L_x = torch.zeros([batch_size, self.dimension_latent_space, self.dimension_latent_space]).to(self.device)

        L_x[:, idx_trn[0], idx_trn[1]] = self.L_layers(x)
        L_x[:, idx_diag[0], idx_diag[1]] = torch.sigmoid(L_x[:, idx_diag[0], idx_diag[1]]) + 1e-5
        L_x[:, idx_not_diag[0], idx_not_diag[1]] = L_x[:, idx_not_diag[0], idx_not_diag[1]]

        return L_x

    # Evaluation

    def calculate_loss(self, x, beta=1, average=True):

        '''

        Function to compute the loss given x.
        For binary values we assume that p(x|z) is Bernoulli
        For continuous values we assume that p(x|z) is a mixture of Logistic with 256 bins
        TODO: [check this and add reference]

        :param x: a batch of input images with shape BCWH
        :param beta: a hyperparam for warmup
        :param average: whether to average loss over the batch or not

        :return: value of the loss function

        '''

        x_mean, x_log_var, L_x = self.forward(x)

        if self.dataset_type == 'binary':
            NLL = log_density_Bernoulli(x, x_mean, reduce_dim=1)
        elif self.dataset_type == 'gray' or self.dataset_type == 'continuous':
            NLL = - log_density_discretized_Logistic(x, x_mean, x_log_var, reduce_dim=1)

        k = L_x.shape[1]
        ixd_diag = np.diag_indices(k)
        diag_L_x = L_x[:, ixd_diag[0], ixd_diag[1]]

        tr_R = torch.sum(L_x ** 2, dim=(1, 2))
        tr_log_L = torch.sum(torch.log(diag_L_x), dim=1)
        KL = torch.mean((tr_R - k) / 2 - tr_log_L)

        loss = - NLL + beta * KL

        if average:
            loss = torch.mean(loss)
            NLL = torch.mean(NLL)
            KL = torch.mean(KL)

        return loss, NLL, KL

    # Evaluation

    def get_latent_code(self, x):

        z_x_mean, z_x_log_var = self.q_z(x)
        return self.sampling_from_normal(z_x_mean, z_x_log_var)

    def get_reconstruction(self, x):

        x_reconstructed, _, _, = self.forward(x)

        return x_reconstructed


class CopulaVAEWithNormals(BaseCopulaVAE):

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 dataset_type,
                 encoder_output_size=300,
                 device=torch.device("cpu")):

        super(CopulaVAEWithNormals, self).__init__(dimension_latent_space=dimension_latent_space,
                                                   input_shape=input_shape,
                                                   dataset_type=dataset_type,
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

        self.F_x_layers = nn.Sequential(
            GatedDense(self.dimension_latent_space, 300),
            GatedDense(300, 300)
        )

        self.p_x_mean = nn.Sequential(
            nn.Linear(300, np.prod(self.input_shape)),
            nn.Sigmoid()
        )

        if not dataset_type == 'binary':
            self.p_x_log_var = nn.Sequential(
                nn.Linear(300, np.prod(self.input_shape)),
                nn.Hardtanh(min_val=-4.5, max_val=0)
            )

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def p_z(self, s):

        return gaussian_0_I_icdf(s)


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