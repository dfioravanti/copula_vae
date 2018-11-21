import argparse
import random
import pathlib
import json

from loaders import load_funtions
from utils.training import train_on_dataset

from models.VAE import VAE
from models.copula_VAE import MarginalVAE, CopulaVAE
from loaders.BinaryMNISTDataset import BinaryMNISTDataset

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

# Training settings
parser = argparse.ArgumentParser(description='VAE')

# arguments for optimization
parser.add_argument('--batch_size', type=int, default=100, metavar='BStrain',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=2000, metavar='E',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--early_stopping_epochs', type=int, default=50, metavar='ES',
                    help='number of epochs for early stopping')

parser.add_argument('--warmup', type=int, default=100, metavar='WU',
                    help='number of epochs for warmu-up')

# cuda
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')

parser.add_argument('--verbose', action='store_true', default=True,
                    help='enables verbose behaviour')

# random seed
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

# model
parser.add_argument('--s_size', type=int, default=50, metavar='M1',
                    help='latent space size (default: 50)')

parser.add_argument('--architecture', type=str, default='copula2',
                    help='architecture: standard, copula, copula2')

parser.add_argument('--marginals', type=str, default='gaussian',
                    help='architecture: gaussian, laplace, log_norm')

# experiment
parser.add_argument('--S', type=int, default=5000, metavar='SLL',
                    help='number of samples used for approximating log-likelihood')

# dataset
parser.add_argument('--dataset_name', type=str, default='mnist', metavar='DN',
                    help='name of the dataset: binary_mnist, mnist, bedrooms,'
                         ' omniglot, cifar10, fashionmnist')

parser.add_argument('--dynamic_binarization', action='store_true', default=True,
                    help='allow dynamic binarization')

parser.add_argument('--output_dir', type=str, default='./outputs',
                    help='Location of the output directory')

args = parser.parse_args()
args_as_json = json.dumps(vars(args))

args.cuda = args.cuda and torch.cuda.is_available()

if args.dynamic_binarization:
    folder_name = f'{args.dataset_name}_bin'
else:
    folder_name = f'{args.dataset_name}'
if args.architecture == 'copula':
    folder_name = f'{folder_name}_{args.marginals}_{args.architecture}_{args.s_size}'
else:
    folder_name = f'{folder_name}_{args.architecture}_{args.s_size}'

args.output_dir = pathlib.Path(args.output_dir) / folder_name
args.output_dir.mkdir(parents=True, exist_ok=True)

with open(args.output_dir / 'config.json', 'w') as f:
    f.write(args_as_json)

if args.warmup == 0:
    args.warmup = None

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def main(args):
    train_loader, validation_loader, test_loader, \
    input_shape, dataset_type = load_dataset(dataset_name=args.dataset_name,
                                             batch_size=args.batch_size)

    if args.architecture == 'standard':
        model = VAE(dimension_latent_space=args.s_size,
                    input_shape=input_shape,
                    encoder_output_size=300,
                    dataset_type=dataset_type,
                    device=args.device)

    elif args.architecture == 'copula':
        model = MarginalVAE(dimension_latent_space=args.s_size,
                            input_shape=input_shape,
                            encoder_output_size=300,
                            dataset_type=dataset_type,
                            marginals=args.marginals,
                            device=args.device)

    elif args.architecture == 'copula2':
        model = CopulaVAE(dimension_latent_space=args.s_size,
                          input_shape=input_shape,
                          encoder_output_size=300,
                          dataset_type=dataset_type,
                          device=args.device)
    else:
        raise ValueError(f'We do not support {args.architecture} as architecture')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(args.device)

    model, results_train, results_val = train_on_dataset(model=model,
                                                         loader_train=train_loader,
                                                         loader_validation=validation_loader,
                                                         optimizer=Adam(model.parameters(), lr=args.lr),
                                                         epochs=args.epochs,
                                                         warmup=args.warmup,
                                                         early_stopping_tolerance=args.early_stopping_epochs,
                                                         device=args.device,
                                                         output_dir=args.output_dir)

    save_training(model, results_train, results_val, args.output_dir)

    results_test = model.calculate_likelihood(train_loader,
                                              number_samples=args.S,
                                              output_dir=args.output_dir
                                              )

    results_test = np.expand_dims(results_test, axis=0)
    np.savetxt(args.output_dir / "test.txt", results_test, delimiter='\t')

    print('Done')


def load_dataset(dataset_name, batch_size=50):
    if dataset_name == 'binary_mnist':

        train_loader, test_loader, validation_loader = load_funtions.load_binary_MNIST(batch_size=batch_size)
        input_shape = BinaryMNISTDataset.shape

    elif dataset_name == 'mnist':

        train_loader, test_loader, validation_loader, dataset_type = \
            load_funtions.load_MNIST(batch_size=batch_size,
                                     dynamic_binarization=args.dynamic_binarization, shuffle=False)
        input_shape = (1, 28, 28)

    elif dataset_name == 'fashionmnist':

        train_loader, test_loader, validation_loader = load_funtions.load_FashionMNIST(batch_size=batch_size,
                                                                                       shuffle=False)
        input_shape = (1, 28, 28)

    elif dataset_name == 'omniglot':

        train_loader, test_loader, validation_loader = load_funtions.load_omniglot(batch_size=batch_size, shuffle=False)
        input_shape = (1, 105, 105)

    elif dataset_name == 'cifar10':

        train_loader, test_loader, validation_loader = load_funtions.load_cifar10(batch_size=batch_size, shuffle=False)
        input_shape = (3, 32, 32)

    elif dataset_name == 'bedrooms':

        train_loader, test_loader, validation_loader = load_funtions.load_bedrooms(batch_size=batch_size, shuffle=False)
        input_shape = (3, 28, 28)

    else:
        raise ValueError('Wrond dataset name')

    return train_loader, validation_loader, test_loader, input_shape, dataset_type


def save_training(model, results_train, results_val, output_dir):
    epoch_train_loss, epoch_train_RE, epoch_train_KLs = results_train
    epoch_val_loss, epoch_val_RE, epoch_val_KLs = results_val

    torch.save(model.state_dict(), output_dir / 'model')
    np.savetxt(output_dir / "loss_train.txt", epoch_train_loss, delimiter='\t')
    np.savetxt(output_dir / "RE_train.txt", epoch_train_RE, delimiter='\t')
    np.savetxt(output_dir / "KL_train.txt", epoch_train_KLs, delimiter='\t')
    np.savetxt(output_dir / "loss_val.txt", epoch_val_loss, delimiter='\t')
    np.savetxt(output_dir / "RE_val.txt", epoch_val_RE, delimiter='\t')
    np.savetxt(output_dir / "KL_val.txt", epoch_val_KLs, delimiter='\t')


if __name__ == '__main__':
    main(args)
