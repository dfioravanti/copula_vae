
import pathlib

import urllib.request

import pandas as pd

import torch
import torch.utils.data as data_utils

import numpy as np
from utils.file_utils import exists_and_correct_hash


class BinaryMNISTDataset(data_utils.Dataset):

    """
        MNIST dataset processed to be just black and white without gray
        TODO: Fix this mess, now I am loading everything everytime. I should load only the right stuff
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

    shape = (1, 28, 28)

    def __init__(self,
                 root_dir=None,
                 download=True,
                 train=True,
                 test=False,
                 validation=False,
                 transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            download (binary): Should I download the file
            transform (callable): Optional transform to be applied on a sample.
        """

        if not ((train == True and test == validation == False) or
                (test == True and train == validation == False) or
                (validation == True and train == test == False)):
            raise ValueError("Please select one and only one between train, test and validation")

        self.transform = transform

        if root_dir is None:
            root_dir = pathlib.Path(__file__).parents[0] / 'datasets'

        if not isinstance(root_dir, pathlib.Path):
            root_dir = pathlib.Path(root_dir)

        self.dataset_path = root_dir / f'{self.__class__.__name__}'
        self.test_path = self.dataset_path / 'binarized_mnist_test.amat'
        self.train_path = self.dataset_path / 'binarized_mnist_train.amat'
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

        if train:
            train_df = pd.read_csv(self.train_path, sep=' ', index_col=False, header=None, dtype=np.float32)
            self.values = train_df.values
        elif test:
            test_df = pd.read_csv(self.test_path, sep=' ', index_col=False, header=None, dtype=np.float32)
            self.values = test_df.values
        elif validation:
            valid_df = pd.read_csv(self.valid_path, sep=' ', index_col=False, header=None, dtype=np.float32)
            self.values = valid_df.values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        sample = self.values[idx, :]

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':

    dataset = BinaryMNISTDataset('../datasets/', True, train=True)

    print(len(dataset))