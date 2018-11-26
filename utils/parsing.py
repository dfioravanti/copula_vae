import argparse
from datetime import datetime
import yaml
from types import SimpleNamespace
import sys
from pathlib import Path

import torch
import torch.nn as nn

from models.LayerBauer import CopulaVAE, MarginalVAE
from models.LayerBetaVAE import ConvMarginalVAE
from models.VAE import VAE
from utils.HashTools import mnemonify_hash, string_to_md5


def get_args():

    main_path = Path(sys.argv[0]).parent
    args = parse_config(main_path)

    args_as_yaml = yaml.dump(vars(args))
    mnemonic_name_arguments = mnemonify_hash(string_to_md5(args_as_yaml))

    # Change some args to be more useful
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.warmup == 0:
        args.warmup = None

    # Create folders

    if args.dynamic_binarization:
        experiment_description = f'{args.dataset_name}_bin'
    else:
        experiment_description = f'{args.dataset_name}'
    if args.architecture == 'copula':
        experiment_description = f'{experiment_description}_{args.marginals}_{args.architecture}'
    else:
        experiment_description = f'{experiment_description}_{args.architecture}'

    experiment_description = f'{experiment_description}_{args.latent_size}'

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.output_dir = Path(args.output_dir) / experiment_description / mnemonic_name_arguments / current_time
    args.log_dir = args.output_dir / 'logs'
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config file for future use

    with open(args.output_dir / 'config.yml', 'w') as f:
        f.write(args_as_yaml)

    return args


def parse_config(main_dir):
    cfg = SimpleNamespace()

    with open(main_dir / 'defaults.yml', 'r') as f:
        defaults = yaml.load(f)

    for _, section in defaults.items():
        for arg, value in section.items():
            cfg.__setattr__(arg, value)

    experiment_config_path = main_dir / 'experiment.yml'
    if experiment_config_path.is_file():

        with open(experiment_config_path, 'r') as f:
            defaults = yaml.load(f)

        for _, section in defaults.items():
            for arg, value in section.items():
                cfg.__setattr__(arg, value)

    return cfg


def parse_command_line():
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

    return parser.parse_args()


def get_model(args, input_shape, dataset_type, output_dir=None):
    if args.architecture == 'standard':
        model = VAE(dimension_latent_space=args.latent_size,
                    input_shape=input_shape,
                    encoder_output_size=300,
                    dataset_type=dataset_type,
                    device=args.device)

    elif args.architecture == 'conv':
        model = ConvMarginalVAE(dimension_latent_space=args.latent_size,
                                input_shape=input_shape,
                                dataset_type=dataset_type,
                                device=args.device)

    elif args.architecture == 'copula':
        model = MarginalVAE(dimension_latent_space=args.latent_size,
                            input_shape=input_shape,
                            encoder_output_size=300,
                            dataset_type=dataset_type,
                            marginals=args.marginals,
                            device=args.device)

    elif args.architecture == 'copula2':
        model = CopulaVAE(dimension_latent_space=args.latent_size,
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
