
import urllib.request

import torch
import torch.utils.data as data_utils

import numpy as np


class BinaryMNISTDataset(data_utils.dataset):

    """
        MNIST dataset processed to be just black and white without gray
    """

    def __init__(self, csv_file=None, root_dir=None, download=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.URL_test = 'http://www.cs.toronto.edu/~larocheh/public/datasets/' \
                        'binarized_mnist/binarized_mnist_test.amat'
        self.URL_train = 'http://www.cs.toronto.edu/~larocheh/public/datasets/' \
                         'binarized_mnist/binarized_mnist_train.amat'
        self.URL_validation = 'http://www.cs.toronto.edu/~larocheh/public/datasets/' \
                              'binarized_mnist/binarized_mnist_valid.amat'



        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

