import numpy as np

import torch
from torch import nn

from models.BaseVAE import BaseVAE

from utils.distributions import log_density_bernoulli, log_density_discretized_Logistic, log_density_standard_Normal, \
    log_density_Normal, kl_normal_and_standard_normal
from utils.inverse_distibutions import normal_icdf
from utils.copula_sampling import sampling_from_gaussian_copula
from utils.settings import tollerance


class NewCopulaVAE(BaseVAE):

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 dataset_type,
                 device=torch.device("cpu")):

        super(NewCopulaVAE, self).__init__(dimension_latent_space=dimension_latent_space,
                                           input_shape=input_shape,
                                           dataset_type=dataset_type,
                                           device=device)

        # Encoder q(z|x)

        self.q_z_layers = nn.Sequential(
            nn.Linear(np.prod(self.input_shape), 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Tanh(),
        )

        self.L_layers = nn.Sequential(
            nn.Linear(300, self.dimension_latent_space),
        )
        self.mean = nn.Linear(300, self.dimension_latent_space)
        self.log_var = nn.Linear(300, self.dimension_latent_space)

        # Decoder p(x|z)

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

    # Model execution

    def forward(self, x):

        z_x, L_x, z_x_mean, z_x_log_var = self.q_z(x)
        x_mean, x_logvar = self.p_x(z_x)

        return x_mean, x_logvar, z_x, L_x, z_x_mean, z_x_log_var

    def q_z(self, x):

        batch_size = x.shape[0]
        x = self.q_z_layers(x)
        L_x, z_x_mean, z_x_log_var = self.L_x(x), self.mean(x), self.log_var(x)

        # TODO: Check that with I one gets a standard gaussian
        z_x = sampling_from_gaussian_copula(L=L_x, d=self.dimension_latent_space, n=batch_size)
        z_x = normal_icdf(z_x, loc=z_x_mean, scale=z_x_log_var)

        return z_x, L_x, z_x_mean, z_x_log_var

    def p_x(self, z):
        z = self.p_x_layers(z)

        if self.dataset_type == 'binary':
            return self.p_x_mean(z), None
        else:
            return self.p_x_mean(z), self.p_x_log_var(z)

    def L_x(self, x):

        batch_size = x.shape[0]

        idx_diag = np.diag_indices(self.dimension_latent_space)
        L_x = torch.zeros([batch_size, self.dimension_latent_space, self.dimension_latent_space]).to(self.device)
        # L_x[:, idx_diag[0], idx_diag[1]] = torch.sigmoid(self.L_layers(x)) + tollerance

        L_x[:, idx_diag[0], idx_diag[1]] = 1

        return L_x

    # Evaluation

    def get_L_x(self, x):

        _, L, _, _ = self.q_z(x)
        return L

    def get_reconstruction(self, x):

        x_reconstructed, _, _, _, _, _, = self.forward(x)

        return x_reconstructed

    def sampling_from_posterior(self, L, mean_marginals, log_var):

        pass

    def p_z(self, n):

        return torch.randn(n, self.dimension_latent_space)

        # eye = torch.eye(self.dimension_latent_space)
        # return normal_icdf(sampling_from_gaussian_copula(L=eye, d=self.dimension_latent_space, n=n))

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

        x_mean, x_log_var, z_x, L_x, z_x_mean, z_x_log_var = self.forward(x)

        if self.dataset_type == 'binary':
            RE = log_density_bernoulli(x, x_mean, reduce_dim=1)
        elif self.dataset_type == 'gray' or self.dataset_type == 'continuous':
            RE = - self.L2(x, x_mean)

        KL_copula = self.compute_KL_copula(L_x)
        KL_marginal = self.compute_KL_marginals(z_x, z_x_mean, z_x_log_var)
        KL = torch.mean(KL_copula + KL_marginal)

        # We are going to minimise so we need to take -ELBO
        loss = -RE + beta * KL

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)
            z_x_mean = torch.mean(z_x_mean)
            z_x_log_var = torch.mean(z_x_log_var)

        return loss, RE, KL, z_x_mean, z_x_log_var

    def compute_KL_copula(self, L):

        d = L.shape[1]
        ixd_diag = np.diag_indices(d)
        diag_L_x = L[:, ixd_diag[0], ixd_diag[1]]

        tr_R = torch.sum(L ** 2, dim=(1, 2))
        tr_log_L = torch.sum(torch.log(diag_L_x), dim=1)
        return (tr_R - d) / 2 - tr_log_L

    def compute_KL_marginals(self, z, means, log_vars):

        log_p_z = log_density_standard_Normal(z)
        log_q_z = log_density_Normal(z, means, log_vars)
        return torch.sum(log_q_z - log_p_z, dim=1)
