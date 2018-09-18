
import sys
import pathlib

import numpy as np

import torch
import torch.utils.data as data_utils

from torchvision import transforms, datasets

from loaders import BinaryMNISTDataset


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

    return train_loader, test_loader, valid_loader


def load_MNIST(root_dir=None, batch_size=20, shuffle=True, transform=None, download=True, random_seed=42):

    if root_dir is None:
        root_dir = pathlib.Path(sys.argv[0]).parents[0] / 'datasets'

    if transform is None:
        transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root_dir, train=True, download=download,
                                   transform=transform)

    valid_dataset = datasets.MNIST(root_dir, train=True, download=download,
                                   transform=transform)

    size_train = len(train_dataset)
    indices = list(range(size_train))
    split = int(np.floor(0.2 * size_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = data_utils.SubsetRandomSampler(train_idx)
    valid_sampler = data_utils.SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler
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

    train_loader, _, _, = load_binary_MNIST('../datasets/', batch_size=1, download=False)

    sample = next(iter(train_loader))
    print(sample.shape)
    sample = sample.reshape((28, 28))
    plt.imshow(sample, cmap='gray')
    plt.show()

