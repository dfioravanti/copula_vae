
import pathlib

import urllib.request

import torch
import torch.utils.data as data_utils

import numpy as np
from utils.file_utils import exists_and_correct_hash


class BinaryMNISTDataset(data_utils.Dataset):

    """
        MNIST dataset processed to be just black and white without gray
    """

    URL_test = 'http://www.cs.toronto.edu/~larocheh/public/datasets/' \
               'binarized_mnist/binarized_mnist_test.amat'
    sha1_test = 'ca70bd7775e182596dec148d4dc1bc6bd6c4c5f4'

    URL_train = 'http://www.cs.toronto.edu/~larocheh/public/datasets/' \
                'binarized_mnist/binarized_mnist_train.amat'
    sha1_train = 'cf70514ec5f481e54ad3eb0e6c03c83326ff96f8'

    URL_validation = 'http://www.cs.toronto.edu/~larocheh/public/datasets/' \
                     'binarized_mnist/binarized_mnist_valid.amat'
    sha1_validation = '13418487742e6ea6d48db7b5187353a20b1b1f8c'

    def __init__(self, root_dir=None, download=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if root_dir is None:
            root_dir = pathlib.Path(__file__).parents[0] / 'datasets'

        if not isinstance(root_dir, pathlib.Path):
            root_dir = pathlib.Path(root_dir)

        self.dataset_path = root_dir / f'{self.__class__.__name__}'
        self.test_path = self.dataset_path / 'binarized_mnist_test.amat'
        self.train_path = self.dataset_path / 'binarized_mnist_train.amat'
        self.sha1_train
        self.valid_path = self.dataset_path / 'binarized_mnist_valid.amat'

        if download:

            self.dataset_path.mkdir(parents=True, exist_ok=True)
            print(f'dataset directory is {self.dataset_path}')

            print(f'Beginning file download for {self.__class__.__name__}')

            if not exists_and_correct_hash(self.test_path, BinaryMNISTDataset.sha1_test):
                print('Downloading test')
                urllib.request.urlretrieve(BinaryMNISTDataset.URL_test,
                                           self.test_path)
            else:
                print('test exists and the hash matches, skip')

            if not exists_and_correct_hash(self.train_path, BinaryMNISTDataset.sha1_train):
                print('Downloading train')
                urllib.request.urlretrieve(BinaryMNISTDataset.URL_train,
                                           self.train_path)
            else:
                print('train exists and the hash matches, skip')

            if not exists_and_correct_hash(self.valid_path, BinaryMNISTDataset.sha1_validation):
                print('Downloading validation')
                urllib.request.urlretrieve(BinaryMNISTDataset.URL_validation,
                                           self.valid_path)
            else:
                print('validation exists and the hash matches, skip')


if __name__ == '__main__':

    dataset = BinaryMNISTDataset('../datasets/', True)
    