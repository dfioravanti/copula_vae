
import abc

import numpy as np

import torch
import torch.nn as nn


class BaseVAE(nn.Module):

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 number_latent_variables,
                 input_size,
                 device=torch.device("cpu")):

        super(BaseVAE, self).__init__()

        self.number_latent_variables = number_latent_variables
        self.input_size = input_size
        self.device = device

    def sampling_normal_with_reparametrization(self, mean, log_variance):

        zero_one_normal = torch.randn(self.number_latent_variables, dtype=log_variance.dtype).to(self.device)
        variance = log_variance.mul(0.5).exp()

        return zero_one_normal.mul(variance).add(mean)

    def train_epoch(self,
                    optimizer,
                    loader,
                    epoch,
                    warmup=None,
                    verbose=True):

        self.train()

        batch_losses = batch_REs = batch_KLs = np.zeros(len(loader))

        if warmup is None:
            beta = 1
        else:
            beta = min(epoch/warmup, 1)

        if verbose:
            print(f'Training with beta = {beta}')

        for batch_idx, xs in enumerate(loader):

            optimizer.zero_grad()
            loss, reconstruction_error, KL = self.calculate_loss(xs, beta)
            loss.backward()
            optimizer.step()

            batch_losses[batch_idx] = loss
            batch_REs[batch_idx] = reconstruction_error
            batch_KLs[batch_idx] = KL

        epoch_loss = np.average(batch_losses)
        epoch_RE = np.average(batch_REs)
        epoch_KLs = np.average(batch_KLs)

        return epoch_loss, epoch_RE, epoch_KLs

    @abc.abstractmethod
    def calculate_loss(self, xs, beta, average=True):
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

    test = BaseVAE(number_latent_variables=1, input_size=1)
    log_variance = torch.tensor(1, dtype=torch.float32)
    mean = torch.tensor(0, dtype=torch.float32)
    print(test.sampling_normal_with_reparametrization(mean=mean, log_variance=log_variance))
