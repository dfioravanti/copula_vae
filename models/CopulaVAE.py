import numpy as np

import torch
from torch import nn

from models.BaseVAE import BaseVAE

from utils.distributions import log_density_bernoulli, log_density_standard_normal, log_density_normal, \
    log_density_laplace, log_density_log_normal, log_density_exponential
from utils.inverse_distibutions import normal_icdf, laplace_icdf, log_norm_icdf, exp_icdf
from utils.copula_sampling import sampling_from_gaussian_copula
from utils.settings import tollerance


class BaseCopulaVAE(BaseVAE):

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 dataset_type,
                 device=torch.device("cpu")):

        super(BaseCopulaVAE, self).__init__(dimension_latent_space=dimension_latent_space,
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
        KL = KL_copula + KL_marginal

        # We are going to minimise so we need to take -ELBO
        loss = -RE + beta * KL

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        return loss, RE, KL

    def compute_KL_copula(self, L):

        d = L.shape[1]
        ixd_diag = np.diag_indices(d)
        diag_L_x = L[:, ixd_diag[0], ixd_diag[1]]

        tr_R = torch.sum(L ** 2, dim=(1, 2))
        tr_log_L = torch.sum(torch.log(diag_L_x), dim=1)
        return (tr_R - d) / 2 - tr_log_L


class GaussianCopulaVAE(BaseCopulaVAE):

    def q_z(self, x):
        batch_size = x.shape[0]
        x = self.q_z_layers(x)
        L_x, z_x_mean, z_x_log_var = self.L_x(x), self.mean(x), self.log_var(x)

        z_x = sampling_from_gaussian_copula(L=L_x, d=self.dimension_latent_space, n=batch_size)
        z_x = normal_icdf(z_x, loc=z_x_mean, scale=torch.exp(z_x_log_var))

        return z_x, L_x, z_x_mean, z_x_log_var

    def p_z(self, n):
        eye = torch.eye(self.dimension_latent_space).to(self.device)
        return normal_icdf(sampling_from_gaussian_copula(L=eye, d=self.dimension_latent_space, n=n))

    def compute_KL_marginals(self, z, means, log_vars):
        log_p_z = log_density_normal(z)
        log_q_z = log_density_normal(z, means, torch.exp(log_vars))
        return torch.sum(log_q_z - log_p_z, dim=1)


class LaplaceCopulaVAE(BaseCopulaVAE):

    def q_z(self, x):
        batch_size = x.shape[0]
        x = self.q_z_layers(x)
        L_x, z_x_loc, z_x_scale = self.L_x(x), self.mean(x), torch.sigmoid(self.log_var(x)) + tollerance

        z_x = sampling_from_gaussian_copula(L=L_x, d=self.dimension_latent_space, n=batch_size)
        z_x = laplace_icdf(z_x, loc=z_x_loc, scale=z_x_scale)

        return z_x, L_x, z_x_loc, z_x_scale

    def p_z(self, n):
        eye = torch.eye(self.dimension_latent_space).to(self.device)
        return laplace_icdf(sampling_from_gaussian_copula(L=eye, d=self.dimension_latent_space, n=n))

    def compute_KL_marginals(self, z, means, log_vars):
        log_p_z = log_density_laplace(z)
        log_q_z = log_density_laplace(z, loc=means, scale=log_vars)
        return torch.sum(log_q_z - log_p_z, dim=1)


class LogNormalCopulaVAE(BaseCopulaVAE):

    def q_z(self, x):
        batch_size = x.shape[0]
        x = self.q_z_layers(x)
        L_x, z_x_mean, z_x_log_var = self.L_x(x), self.mean(x), self.log_var(x)

        z_x = sampling_from_gaussian_copula(L=L_x, d=self.dimension_latent_space, n=batch_size)
        z_x = log_norm_icdf(z_x, loc=z_x_mean, scale=torch.exp(z_x_log_var))

        return z_x, L_x, z_x_mean, z_x_log_var

    def p_z(self, n):
        eye = torch.eye(self.dimension_latent_space).to(self.device)
        return log_norm_icdf(sampling_from_gaussian_copula(L=eye, d=self.dimension_latent_space, n=n))

    def compute_KL_marginals(self, z, means, log_vars):
        log_p_z = log_density_log_normal(z)
        log_q_z = log_density_log_normal(z, means, torch.exp(log_vars))
        return torch.sum(log_q_z - log_p_z, dim=1)


class MixCopulaVAE(BaseCopulaVAE):

    def q_z(self, x):
        batch_size = x.shape[0]
        x = self.q_z_layers(x)
        L_x, z_x_mean, z_x_log_var = self.L_x(x), self.mean(x), self.log_var(x)

        s_x = sampling_from_gaussian_copula(L=L_x, d=self.dimension_latent_space, n=batch_size)
        z_1 = normal_icdf(s_x[:, :25], loc=z_x_mean[:, :25], scale=torch.exp(z_x_log_var[:, :25]))
        z_2 = log_norm_icdf(s_x[:, 25:], loc=z_x_mean[:, 25:], scale=torch.exp(z_x_log_var[:, 25:]))
        z_x = torch.cat((z_1, z_2), 1)

        return z_x, L_x, z_x_mean, z_x_log_var

    def p_z(self, n):
        eye = torch.eye(self.dimension_latent_space).to(self.device)
        z_1 = normal_icdf(sampling_from_gaussian_copula(L=eye, d=self.dimension_latent_space//2, n=n))
        z_2 = log_norm_icdf(sampling_from_gaussian_copula(L=eye, d=self.dimension_latent_space // 2, n=n))
        z = torch.cat((z_1, z_2), 1)
        return z

    def compute_KL_marginals(self, z, means, log_vars):
        log_p_z_1 = log_density_normal(z[:, :25])
        log_q_z_1 = log_density_normal(z[:, :25], means[:, :25], torch.exp(log_vars[:, :25]))
        log_p_z_2 = log_density_log_normal(z[:, 25:])
        log_q_z_2 = log_density_log_normal(z[:, 25:], means[:, 25:], torch.exp(log_vars[:, 25:]))
        return torch.sum(log_q_z_1 - log_p_z_1, dim=1) + torch.sum(log_q_z_2 - log_p_z_2, dim=1)


class ExponentialCopulaVAE(BaseCopulaVAE):

    def q_z(self, x):
        batch_size = x.shape[0]
        x = self.q_z_layers(x)
        L_x, z_x_rate = self.L_x(x), torch.sigmoid(self.mean(x)) + tollerance

        z_x = sampling_from_gaussian_copula(L=L_x, d=self.dimension_latent_space, n=batch_size)
        z_x = exp_icdf(z_x, rate=z_x_rate)

        return z_x, L_x, z_x_rate, 0

    def p_z(self, n):
        eye = torch.eye(self.dimension_latent_space).to(self.device)
        return exp_icdf(sampling_from_gaussian_copula(L=eye, d=self.dimension_latent_space, n=n))

    def compute_KL_marginals(self, z, means, log_vars):
        log_p_z = log_density_exponential(z)
        log_q_z = log_density_exponential(z, means)
        return torch.sum(log_q_z - log_p_z, dim=1)
