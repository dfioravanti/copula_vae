import argparse
import random

import numpy as np

from models.VAE import VAE
from utils import datasets

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

# Training settings
parser = argparse.ArgumentParser(description='VAE')

# arguments for optimization
parser.add_argument('--batch_size', type=int, default=100, metavar='BStrain',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='BStest',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=2000, metavar='E',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--early_stopping_epochs', type=int, default=50, metavar='ES',
                    help='number of epochs for early stopping')

parser.add_argument('--warmup', type=int, default=100, metavar='WU',
                    help='number of epochs for warmu-up')

# cuda
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')

# random seed
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

# model
parser.add_argument('--z_size', type=int, default=40, metavar='M1',
                    help='latent space size (default: 40)')

parser.add_argument('--prior', type=str, default='standard', metavar='P',
                    help='prior: standard, gaussian_copula')

# experiment
parser.add_argument('--S', type=int, default=5000, metavar='SLL',
                    help='number of samples used for approximating log-likelihood')
parser.add_argument('--MB', type=int, default=100, metavar='MBLL',
                    help='size of a mini-batch used for approximating log-likelihood')

# dataset
parser.add_argument('--dataset_name', type=str, default='binary_mnist', metavar='DN',
                    help='name of the dataset: binary_mnist, dynamic_mnist,'
                         ' omniglot, caltech101silhouettes, histopathologyGray, freyfaces, cifar10')

parser.add_argument('--dynamic_binarization', action='store_true', default=False,
                    help='allow dynamic binarization')

args = parser.parse_args()
args.cuda = args.no_cuda and torch.cuda.is_available()

args.device = torch.device("cuda") if args.cuda else torch.device("cpu")

random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def main(args):

    train_loader, validation_loader, test_loader, input_shape = load_dataset(args.dataset_name)

    if args.prior == 'standard':
        model = VAE(number_latent_variables=40,
                    input_shape=input_shape,
                    encoder_output_size=300,
                    device=args.device)

    model.train_dataset(train_loader,
                        validation_loader,
                        optimizer=Adam(model.parameters()),
                        epochs=50,
                        warmup=10,
                        verbose=True,
                        early_stopping_tolerance=10)


def load_dataset(dataset_name, batch_size=50):
    if dataset_name == 'binary_mnist':
        train_dataset = datasets.BinaryMNISTDataset('datasets/', True, train=True)
        test_dataset = datasets.BinaryMNISTDataset('datasets/', True, train=False, test=True)
        val_dataset = datasets.BinaryMNISTDataset('datasets/', True, train=False, validation=True)
        input_shape = datasets.BinaryMNISTDataset.shape

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True, num_workers=4)

    validation_loader = DataLoader(val_dataset,
                                   batch_size=batch_size,
                                   shuffle=True, num_workers=4)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True, num_workers=4)

    return train_loader, validation_loader, test_loader, input_shape


if __name__ == '__main__':
    main(args)
