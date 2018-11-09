import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class ToBinary(object):
    """
    Convert a continuous tensor X between (0,1) to a discrete one Y
    where Y_i ~ Ber(X_i)
    """

    def __call__(self, x):

        p = x.numpy()
        y = np.random.binomial(1, p)
        return torch.from_numpy(y)


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in self.indices)

    def __len__(self):
        return len(self.indices)