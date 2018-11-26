import argparse
import hashlib
import json
import pathlib

import torch
import torch.nn as nn

from models.LayerBauer import CopulaVAE, MarginalVAE
from models.LayerBetaVAE import ConvMarginalVAE
from models.VAE import VAE


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='VAE')

    # Output and logging

    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Location of the output directory')
    parser.add_argument('--tensorbord', action='store_true', default=True,
                        help='Should TensorBoard be used (default: True)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='enables verbose behaviour')
    parser.add_argument('--to_file', action='store_true', default=True,
                        help='Redirects stdout to file output.txt (default: True)')

    # random seed
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    # arguments for optimization

    parser.add_argument('--batch_size', type=int, default=100, metavar='BStrain',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=2000, metavar='E',
                        help='number of epochs to train (default: 2000)')
    parser.add_argument('--lr', type=float, default=3e-3, metavar='LR',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--early_stopping_epochs', type=int, default=50, metavar='ES',
                        help='number of epochs for early stopping')
    parser.add_argument('--warmup', type=int, default=100, metavar='WU',
                        help='number of epochs for warmu-up')

    # model
    parser.add_argument('--s_size', type=int, default=50, metavar='M1',
                        help='latent space size (default: 50)')

    parser.add_argument('--architecture', type=str, default='standard',
                        help='architecture: standard, conv, copula, copula2')

    parser.add_argument('--marginals', type=str, default='gaussian',
                        help='architecture: gaussian, laplace, log_norm, cauchy, exp')

    # cuda
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')

    # experiment
    parser.add_argument('--S', type=int, default=5000, metavar='SLL',
                        help='number of samples used for approximating log-likelihood')

    parser.add_argument('--load_checkpoint', action='store_true', default=True,
                        help='check and load checkpoints')

    # dataset
    parser.add_argument('--dataset_name', type=str, default='mnist', metavar='DN',
                        help='name of the dataset: binary_mnist, mnist, bedrooms,'
                             ' omniglot, cifar10, fashionmnist, dSprites')

    parser.add_argument('--dynamic_binarization', action='store_true', default=True,
                        help='allow dynamic binarization')

    parser.add_argument('--shuffle', action='store_true', default=False,
                        help='shuffle the dataset (default: False)')

    args = parser.parse_args()
    args_as_json = json.dumps(vars(args))

    args.cuda = args.cuda and torch.cuda.is_available()

    if args.dynamic_binarization:
        folder_name = f'{args.dataset_name}_bin'
    else:
        folder_name = f'{args.dataset_name}'
    if args.architecture == 'copula':
        folder_name = f'{folder_name}_{args.marginals}_{args.architecture}'
    else:
        folder_name = f'{folder_name}_{args.architecture}'

    folder_name = f'{folder_name}_{args.s_size}'

    args.output_dir = pathlib.Path(args.output_dir) / folder_name
    args.log_dir = args.output_dir / 'logs'
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_dir / 'config.json', 'w') as f:
        f.write(args_as_json)

    if args.warmup == 0:
        args.warmup = None

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return args


def get_model(args, input_shape, dataset_type, output_dir=None):
    if args.architecture == 'standard':
        model = VAE(dimension_latent_space=args.s_size,
                    input_shape=input_shape,
                    encoder_output_size=300,
                    dataset_type=dataset_type,
                    device=args.device)

    elif args.architecture == 'conv':
        model = ConvMarginalVAE(dimension_latent_space=args.s_size,
                                input_shape=input_shape,
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

    if output_dir is not None:
        with open(output_dir / 'model.txt', 'w') as f:
            print(model, file=f)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(args.device)

    return model


def choose_loss_function(loss='L2'):
    if loss == 'L1':
        return nn.L1Loss()
    elif loss == 'L2':
        return nn.MSELoss()
    elif loss == 'BCE':
        return nn.BCELoss()
    else:
        raise NotImplementedError("Unknown loss function")
