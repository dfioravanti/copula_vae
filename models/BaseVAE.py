import abc

import numpy as np
from scipy.special import logsumexp

import torch
import torch.nn as nn


class BaseVAE(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 dimension_latent_space,
                 input_shape,
                 dataset_type,
                 device=torch.device("cpu")):

        super(BaseVAE, self).__init__()

        self.dimension_latent_space = dimension_latent_space
        self.input_shape = input_shape
        self.device = device

        if dataset_type not in ['continuous', 'gray', 'binary']:
            raise ValueError(f'We do not support {dataset_type} as dataset_type!')

        self.dataset_type = dataset_type

    @abc.abstractmethod
    def calculate_loss(self, x, beta=1, average=True):
        return None, None, None

    @abc.abstractmethod
    def calculate_lower_bound(self):
        return

    @abc.abstractmethod
    def forward(self, x):
        return

    def get_reconstruction(self, x):

        x_reconstructed, _, _, _, _, = self.forward(x)

        return x_reconstructed

    def get_latent_code(self, x):

        z_x_mean, z_x_log_var = self.q_z(x)
        return self.sampling_from_normal(z_x_mean, z_x_log_var)

    def calculate_likelihood(self, loader, number_samples, writer=None):

        size_dataset = len(loader.dataset)
        batch_size = loader.batch_size

        losses = np.zeros((batch_size, number_samples))
        likelihood_x = np.zeros(size_dataset)

        for i, (xs, _) in enumerate(loader):

            xs = xs.view(batch_size, -1).to(self.device)

            print(f'Computing likelihood for batch number {i}\n')

            for j in range(number_samples):
                loss, _, _ = self.calculate_loss(xs)
                losses[:, j] = loss.cpu().detach().numpy()

            if writer is not None:
                writer.add_scalar(tag='loss/test', scalar_value=loss, global_step=i)

            likelihood_x[i * batch_size:(i + 1) * batch_size] = logsumexp(losses, axis=1) - np.log(number_samples)

        return np.mean(likelihood_x)


if __name__ == '__main__':
    from loaders.load_funtions import load_MNIST
    from models.VAE import VAE

    import pathlib

    _, loader, _, dataset_type = load_MNIST('../datasets/')

    output_dit = pathlib.Path('../outputs/')
    input_shape = (1, 28, 28)

    model = VAE(dimension_latent_space=50, input_shape=input_shape, dataset_type=dataset_type)
    model.load_state_dict(torch.load('../outputs/trained/mnist_bin_standard_50/model', map_location='cpu'))
    model.eval()

    print(model.calculate_likelihood(loader, 100))
