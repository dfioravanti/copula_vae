
import sys
import pathlib

import numpy as np

import torch
import torch.utils.data as data_utils

from torchvision import transforms, datasets

from loaders import BinaryMNISTDataset

from utils.utils_load_dataset import ToBinary, SubsetSampler


def load_binary_MNIST(root_dir=None, batch_size=20, shuffle=True, transform=None, download=True):

    train_dataset = BinaryMNISTDataset.BinaryMNISTDataset(root_dir,
                                                          download=download,
                                                          train=True,
                                                          transform=transform)
    train_loader = data_utils.DataLoader(train_dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle)

    test_dataset = BinaryMNISTDataset.BinaryMNISTDataset(root_dir,
                                                         download=download,
                                                         train=False,
                                                         test=True,
                                                         transform=transform)
    test_loader = data_utils.DataLoader(test_dataset,
                                        batch_size=batch_size,
                                        shuffle=False)

    valid_dataset = BinaryMNISTDataset.BinaryMNISTDataset(root_dir,
                                                          download=download,
                                                          train=False,
                                                          validation=True,
                                                          transform=transform)
    valid_loader = data_utils.DataLoader(valid_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)

    dataset_type = "binary"

    return train_loader, test_loader, valid_loader, dataset_type


def load_MNIST(root_dir=None, dynamic_binarization=True,
               batch_size=20, transform=None, shuffle=True, download=True):

    """
    This function load the MNIST dataset

    :param root_dir: root directory where to store the dataset
    :param dynamic_binarization: If the dataset should dinamicaly binarized as in [ref]
    :param batch_size: see datasets.MNIST
    :param transform: see datasets.MNIST
    :param shuffle: Should the dataset be shuffled
    :param download: see datasets.MNIST
    :return:
    """

    if root_dir is None:
        root_dir = pathlib.Path(sys.argv[0]).parents[0] / 'datasets/MNIST'

    if transform is None:
        transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root_dir, train=True, download=download,
                                          transform=transform)

    test_dataset = datasets.MNIST(root_dir, train=False, download=download,
                                          transform=transform)

    if dynamic_binarization:
        dataset_type = "binary"
        train_data = torch.from_numpy(np.random.binomial(1, train_dataset.train_data.numpy() / 255))
        test_data = torch.from_numpy(np.random.binomial(1, test_dataset.test_data.numpy() / 255))
    else:
        dataset_type = "continuous"
        train_data = torch.from_numpy(train_dataset.train_data.numpy() / 255)
        test_data = torch.from_numpy(test_dataset.test_data.numpy() / 255)

    train_labels = train_dataset.train_labels
    test_labels = test_dataset.test_labels

    train_dataset = data_utils.TensorDataset(train_data.float(), train_labels)
    test_dataset = data_utils.TensorDataset(test_data.float(), test_labels)

    size_train = len(train_dataset)
    indices = list(range(size_train))
    split = int(np.floor(0.2 * size_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetSampler(train_idx)
    valid_sampler = SubsetSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False,
    )

    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler, shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle
    )

    return train_loader, test_loader, valid_loader, dataset_type


def load_FashionMNIST(root_dir=None, batch_size=20, shuffle=True, transform=None, download=True):

    if root_dir is None:
        root_dir = pathlib.Path(sys.argv[0]).parents[0] / 'datasets/Fashion'

    if transform is None:
        transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(root_dir, train=True, download=download,
                                          transform=transform)

    valid_dataset = datasets.FashionMNIST(root_dir, train=True, download=download,
                                          transform=transform)

    size_train = len(train_dataset)
    indices = list(range(size_train))
    split = int(np.floor(0.2 * size_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = data_utils.SubsetRandomSampler(train_idx)
    valid_sampler = data_utils.SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=shuffle
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler, shuffle=shuffle
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transform
        ])),
        batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader, valid_loader


def load_bedrooms(root_dir=None, batch_size=20, shuffle=True, transform=None):

    if root_dir is None:
        root_dir = pathlib.Path(sys.argv[0]).parents[0] / 'datasets'
        root_dir = str(root_dir)

    if transform is None:
        transform = transforms.ToTensor()

    train_dataset = datasets.LSUN(root_dir, classes=['classroom_train'], transform=transform)

    size_train = len(train_dataset)
    indices = list(range(size_train))
    split = int(np.floor(0.2 * size_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = data_utils.SubsetRandomSampler(train_idx)
    valid_sampler = data_utils.SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=shuffle
    )

    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler, shuffle=shuffle
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transform
        ])),
        batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader, valid_loader


def load_omniglot(root_dir=None, batch_size=20, shuffle=True, transform=None, download=True):

    if root_dir is None:
        root_dir = pathlib.Path(sys.argv[0]).parents[0] / 'datasets'
        root_dir = str(root_dir)

    if transform is None:
        transform = transforms.ToTensor()

    train_dataset = datasets.Omniglot(root_dir, transform=transform, download=download)

    size_train = len(train_dataset)
    indices = list(range(size_train))
    split = int(np.floor(0.2 * size_train))

    if split % batch_size != 0:
        raise ValueError(f'The batch size: {batch_size} does not divide the size of '
                         f'the train_dataset: {size_train-split} or the size of the validation_dataset: {split}')

    if shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = data_utils.SubsetRandomSampler(train_idx)
    valid_sampler = data_utils.SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=shuffle
    )

    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler, shuffle=shuffle
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transform
        ])),
        batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader, valid_loader


def load_cifar10(root_dir=None, batch_size=20, shuffle=True, transform=None, download=True):

    if root_dir is None:
        root_dir = pathlib.Path(sys.argv[0]).parents[0] / 'datasets'
        root_dir = str(root_dir)

    if transform is None:
        transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(root_dir, transform=transform, download=download)

    size_train = len(train_dataset)
    indices = list(range(size_train))
    split = int(np.floor(0.2 * size_train))

    if split % batch_size != 0:
        raise ValueError(f'The batch size: {batch_size} does not divide the size of '
                         f'the train_dataset: {size_train-split} or the size of the validation_dataset: {split}')

    if shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = data_utils.SubsetRandomSampler(train_idx)
    valid_sampler = data_utils.SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=shuffle
    )

    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler, shuffle=shuffle
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transform
        ])),
        batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader, valid_loader


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    train_loader, _, _, _ = load_MNIST('../datasets/', dynamic_binarization=True, batch_size=1, download=False, shuffle=False)

    sample = next(iter(train_loader))[0]
    print(sample.shape)
    sample = sample.reshape((28, 28))
    plt.imshow(sample, cmap='gray')
    plt.show()

