
import abc

import numpy as np

import torch
import torch.nn as nn

from utils.plot_utils import plot_grid_images_file


class BaseVAE(nn.Module):

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 number_latent_variables,
                 input_shape,
                 output_dir=None,
                 device=torch.device("cpu")):

        super(BaseVAE, self).__init__()

        self.output_dir = output_dir

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.number_latent_variables = number_latent_variables
        self.input_shape = input_shape
        self.device = device

    def sampling_normal_with_reparametrization(self, mean, log_variance):

        zero_one_normal = torch.randn(self.number_latent_variables, dtype=log_variance.dtype).to(self.device)
        variance = log_variance.mul(0.5).exp()

        return zero_one_normal.mul(variance).add(mean)

    def train_epoch(self,
                    epoch,
                    loader,
                    optimizer,
                    warmup=None,
                    verbose=True):

        self.train()

        batch_losses = batch_REs = batch_KLs = np.zeros(len(loader))

        beta = self._compute_beta(epoch, warmup)

        if verbose:
            print(f'Training with beta = {beta}')

        for batch_idx, (xs, _) in enumerate(loader):

            xs = xs.to(self.device)

            if batch_idx == 0 and self.output_dir is not None:
                self._plot_reconstruction(xs, epoch)

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

    def validation_epoch(self,
                         epoch,
                         loader,
                         warmup=None):

        self.eval()

        beta = self._compute_beta(epoch, warmup)

        batch_losses = batch_REs = batch_KLs = np.zeros(len(loader))

        for batch_idx, (xs, _) in enumerate(loader):

            xs = xs.to(self.device)

            loss, reconstruction_error, KL = self.calculate_loss(xs, beta)

            batch_losses[batch_idx] = loss
            batch_REs[batch_idx] = reconstruction_error
            batch_KLs[batch_idx] = KL

        val_loss = np.average(batch_losses)
        val_RE = np.average(batch_REs)
        val_KLs = np.average(batch_KLs)

        return val_loss, val_RE, val_KLs

    def _compute_beta(self, epoch, warmup):

        if warmup is None:
            beta = 1
        else:
            beta = min(epoch/warmup, 1)

        return beta

    def _plot_reconstruction(self,
                             xs,
                             epoch):

        xs_reconstructed, _, _ = self.forward(xs)

        xs_reconstructed = xs.view((-1,) + self.input_shape)

        filename = self.output_dir / f'Epoch {epoch}'

        plot_grid_images_file(xs_reconstructed.to('cpu').numpy()[0:10, :],
                              columns=5,
                              filename=filename)

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

    test = BaseVAE(number_latent_variables=1, input_shape=1)
    log_variance = torch.tensor(1, dtype=torch.float32)
    mean = torch.tensor(0, dtype=torch.float32)
    print(test.sampling_normal_with_reparametrization(mean=mean, log_variance=log_variance))
