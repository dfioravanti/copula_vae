
import abc

import numpy as np

import torch
import torch.nn as nn




class BaseVAE(nn.Module):

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 device=torch.device("cpu")):

        super(BaseVAE, self).__init__()

        self.dimension_latent_space = dimension_latent_space
        self.input_shape = input_shape
        self.device = device

    def sampling_normal_with_reparametrization(self, mean, log_variance):

        zero_one_normal = torch.randn(self.dimension_latent_space, dtype=log_variance.dtype).to(self.device)
        variance = log_variance.exp()

        return zero_one_normal.mul(variance).add(mean)

    @abc.abstractmethod
    def calculate_loss(self, xs, beta=1, loss=nn.MSELoss()):
        return

    @abc.abstractmethod
    def calculate_likelihood(self):
        return

    @abc.abstractmethod
    def calculate_lower_bound(self):
        return

    @abc.abstractmethod
    def forward(self, x):
        return

    @abc.abstractmethod
    def get_encoder_layers(self):
        return

    @abc.abstractmethod
    def get_decoder_layers(self):
        return


if __name__ == '__main__':

    test = BaseVAE(dimension_latent_space=1, input_shape=1)
    log_variance = torch.tensor(1, dtype=torch.float32)
    mean = torch.tensor(0, dtype=torch.float32)
    print(test.sampling_normal_with_reparametrization(mean=mean, log_variance=log_variance))
