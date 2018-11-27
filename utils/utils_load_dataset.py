"""
    This file provides some utilities needed to load the datasets.
"""

import torch
from torch.utils.data.sampler import Sampler


class SubsetSampler(Sampler):
    """
    Samples elements from a given list of indices.

    Parameters
    ----------
    mask (sequence):
         The sequence of indices to be sampled.
    """

    def __init__(self, mask):
        self.mask = mask

    def __iter__(self):
        return (self.mask[i] for i in torch.arange(start=0, end=len(self.mask)))

    def __len__(self):
        return len(self.mask)
