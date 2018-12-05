from models.BaseVAE import BaseVAE

import numpy as np
import torch
from torch import nn
from utils.distributions import log_density_bernoulli, log_density_discretized_Logistic, \
    log_density_Normal, log_density_standard_Normal


class VAE(BaseVAE):

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 dataset_type,
                 device=torch.device("cpu")):

        super(VAE, self).__init__(dimension_latent_space=dimension_latent_space,
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

        self.mean = nn.Linear(300, self.dimension_latent_space)
        self.log_var = nn.Sequential(
            nn.Linear(300, self.dimension_latent_space),
            nn.Hardtanh(min_val=-6., max_val=2.)
        )

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

        z_x_mean, z_x_log_var = self.q_z(x)
        z_x = self.sampling_from_normal(z_x_mean, z_x_log_var)
        x_mean, x_logvar = self.p_x(z_x)

        return x_mean, x_logvar, z_x, z_x_mean, z_x_log_var

    def q_z(self, x):

        x = self.q_z_layers(x)
        return self.mean(x), self.log_var(x)

    def p_x(self, z):
        z = self.p_x_layers(z)

        if self.dataset_type == 'binary':
            return self.p_x_mean(z), None
        else:
            return self.p_x_mean(z), self.p_x_log_var(z)

    def sampling_from_normal(self, mean, log_variance):

        zero_one_normal = torch.randn(self.dimension_latent_space, dtype=log_variance.dtype).to(self.device)
        variance = log_variance.exp()

        return zero_one_normal.mul(variance).add(mean)

    # Evaluation


    def log_desity_prior(self, z):

        return log_density_standard_Normal(z, reduce_dim=1)

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

        x_mean, x_log_var, z_x, z_x_mean, z_x_log_var = self.forward(x)

        if self.dataset_type == 'binary':
            RE = log_density_bernoulli(x, x_mean, reduce_dim=1)
        elif self.dataset_type == 'gray' or self.dataset_type == 'continuous':
            RE = - log_density_discretized_Logistic(x, x_mean, x_log_var, reduce_dim=1)

        log_p_z = self.log_desity_prior(z_x)
        log_q_z = log_density_Normal(z_x, z_x_mean, z_x_log_var, reduce_dim=1)
        KL = log_q_z - log_p_z

        # We are going to minimise so we need to take -ELBO
        loss = -RE + beta * KL

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        return loss, RE, KL


class DeepVAE(VAE):

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 dataset_type,
                 device=torch.device("cpu")):

        super(DeepVAE, self).__init__(dimension_latent_space=dimension_latent_space,
                                      input_shape=input_shape,
                                      dataset_type=dataset_type,
                                      device=device)

        self.q_z_layers = nn.Sequential(
            nn.Linear(np.prod(self.input_shape), 1200),
            nn.ReLU(),
            nn.Linear(1200, 1200),
            nn.ReLU(),
        )

        self.mean = nn.Linear(1200, self.dimension_latent_space)
        self.log_var = nn.Sequential(
            nn.Linear(1200, self.dimension_latent_space),
            nn.Hardtanh(min_val=-6., max_val=2.)
        )

        # Decoder p(x|z)

        self.p_x_layers = nn.Sequential(
            nn.Linear(self.dimension_latent_space, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
        )

        self.p_x_mean = nn.Sequential(
            nn.Linear(1200, np.prod(self.input_shape)),
            nn.Sigmoid()
        )

        if not dataset_type == 'binary':
            self.p_x_log_var = nn.Linear(1200, np.prod(self.input_shape))

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
