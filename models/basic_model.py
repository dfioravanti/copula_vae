
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

    @abc.abstractmethod
    def get_encoder_layers(self):
        return

    @abc.abstractmethod
    def get_decoder_layers(self):
        return

    def calculate_likelihood(self, loader, number_samples, output_dir):

        size_dataset = len(loader.dataset)
        batch_size = loader.batch_size
        output_file = output_dir / 'output.txt'

        losses = np.zeros((size_dataset, number_samples))

        with open(output_file, 'a', buffering=1) as f:

            for i, (xs, _) in enumerate(loader):

                xs = xs.view(batch_size, -1).to(self.device)

                f.write(f'Computing likelihood for batch number {i}\n')

                for j in range(number_samples):

                    loss, _, _ = self.calculate_loss(xs)
                    losses[i*batch_size:(i+1)*batch_size, j] = -loss.cpu().detach().numpy()

            likelihood_x = logsumexp(losses, axis=1) - np.log(number_samples)

            return np.mean(likelihood_x)


if __name__ == '__main__':

    from loaders.load_funtions import load_MNIST
    from models.VAE import VAE

    import pathlib

    _, loader, _, dataset_type = load_MNIST('../datasets/')

    output_dit = pathlib.Path('../outputs/')
    input_shape = (1, 28, 28)

    model = VAE(dimension_latent_space=20, input_shape=input_shape, dataset_type=dataset_type)
    print(model.calculate_likelihood(loader, 10, output_dit))
