import numpy as np
import torch

from models.BaseVAE import BaseVAE
from utils.copula_sampling import sampling_from_gausiann_copula
from utils.distributions import log_density_bernoulli, log_density_discretized_Logistic


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
        z = self.p_x_layers(z)

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
            RE = log_density_bernoulli(x, x_mean, reduce_dim=1)
        elif self.dataset_type == 'gray' or self.dataset_type == 'continuous':
            RE = - log_density_discretized_Logistic(x, x_mean, x_log_var, reduce_dim=1)

        k = L_x.shape[1]
        ixd_diag = np.diag_indices(k)
        diag_L_x = L_x[:, ixd_diag[0], ixd_diag[1]]

        tr_R = torch.sum(L_x ** 2, dim=(1, 2))
        tr_log_L = torch.sum(torch.log(diag_L_x), dim=1)
        KL = torch.mean((tr_R - k) / 2 - tr_log_L)

        loss = - RE + beta * KL

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        return loss, RE, KL

    # Evaluation

    def get_latent_code(self, x):

        z_x_mean, z_x_log_var = self.q_z(x)
        return self.sampling_from_normal(z_x_mean, z_x_log_var)

    def get_reconstruction(self, x):

        x_reconstructed, _, _, = self.forward(x)

        return x_reconstructed
