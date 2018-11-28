"""
    This file provides functions needed to parse the configurations and initialise the experiment
"""

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import yaml

from models.ConvLayer import ConvMarginalVAE, ConvDiagonalMarginalVAE
from models.DeepLayer import DeepMarginalVAE, DeepDiagonalMarginalVAE
from models.ShallowLayer import ShallowCopulaVAE, ShallowMarginalVAE, ShallowDiagonalMarginalVAE
from models.VAE import VAE
from utils.HashTools import mnemonify_hash, string_to_md5


def get_args_and_setting_up():
    """
    Load the arguments from the config files, sets up the directory structure
    and saves the configuration for future use.

    Returns
    -------
    SimpleNamespace
        A namespace containing all the selected options.
    """
    main_path = Path(sys.argv[0]).parent
    args = parse_config(main_path)

    args_as_yaml = yaml.dump(vars(args))
    mnemonic_name_arguments = mnemonify_hash(string_to_md5(args_as_yaml))

    # Change some args to be more useful

    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.warmup == 0:
        args.warmup = None
    if not args.checkpointing:
        args.frequency_checkpoints = None
    if args.frequency_checkpoints == 0:
        args.frequency_checkpoints = args.epochs // 100 * 5

    # Create folders

    experiment_description = f'{args.dataset_name}'
    if args.dynamic_binarization:
        experiment_description = f'{experiment_description}_bin'

    experiment_description = f'{experiment_description}_{args.architecture}_{args.type_vae}'

    if args.type_vae == 'copula' or args.type_vae == 'diagonal':
        experiment_description = f'{experiment_description}_{args.marginals}'

    experiment_description = f'{experiment_description}_{args.latent_size}'

    if args.extra_text:
        experiment_description = f'{experiment_description}_{args.extra_text}'

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.output_dir = Path(args.output_dir) / experiment_description / mnemonic_name_arguments / current_time
    args.log_dir = args.output_dir / 'logs'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.frequency_checkpoints is not None:
        args.checkpoint_dir = args.output_dir / "checkpoints"
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config file for future use

    with open(args.output_dir / 'config.yml', 'w') as f:
        f.write(args_as_yaml)

    return args


def parse_config(path):
    """
    Loads the default configuration and, if needed, it overwrites
    them with the one defined for the single experiment.

    Parameters
    ----------
    path: Path
        Path to the directory containing the configuration files

    Returns
    -------
    SimpleNamespace
        A namespace containing all the selected options.
    """

    cfg = SimpleNamespace()

    with open(path / 'defaults.yml', 'r') as f:
        defaults = yaml.load(f)

    for _, section in defaults.items():
        for arg, value in section.items():
            cfg.__setattr__(arg, value)

    experiment_config_path = path / 'experiment.yml'
    if experiment_config_path.is_file():

        with open(experiment_config_path, 'r') as f:
            defaults = yaml.load(f)

        for _, section in defaults.items():
            for arg, value in section.items():
                cfg.__setattr__(arg, value)

    return cfg


def get_model(args, input_shape, dataset_type, output_dir=None):
    """
    Selects and initialise a pytorch model based on the selected arguments.
    Raises a NotImplementedError if the required model is not available.

    Parameters
    ----------
    args: SimpleNamespace
        Namespace containing the arguments of the experiment
    input_shape: tuple of int
        Shape of the input WITHOUT batch size
    dataset_type: str
        Which type of dataset are we using, e.i. binary or continuous
    output_dir: Path
        path to the output directory

    Returns
    -------
    nn.Module:
        The selected model correctly initialised

    """

    model = None

    if args.type_vae == 'standard':

        if args.architecture == 'shallow':
            model = VAE(dimension_latent_space=args.latent_size,
                        input_shape=input_shape,
                        encoder_output_size=300,
                        dataset_type=dataset_type,
                        device=args.device)

    elif args.type_vae == 'copula':

        if args.architecture == 'shallow':
            model = ShallowMarginalVAE(dimension_latent_space=args.latent_size,
                                       input_shape=input_shape,
                                       dataset_type=dataset_type,
                                       marginals=args.marginals,
                                       device=args.device)

        elif args.architecture == 'deep':
            model = DeepMarginalVAE(dimension_latent_space=args.latent_size,
                                    input_shape=input_shape,
                                    dataset_type=dataset_type,
                                    marginals=args.marginals,
                                    device=args.device)

        elif args.architecture == 'conv':
            model = ConvMarginalVAE(dimension_latent_space=args.latent_size,
                                    input_shape=input_shape,
                                    dataset_type=dataset_type,
                                    device=args.device)

    elif args.type_vae == 'diagonal':

        if args.architecture == 'shallow':
            model = ShallowDiagonalMarginalVAE(dimension_latent_space=args.latent_size,
                                               input_shape=input_shape,
                                               dataset_type=dataset_type,
                                               marginals=args.marginals,
                                               device=args.device)
        elif args.architecture == 'deep':
            model = DeepDiagonalMarginalVAE(dimension_latent_space=args.latent_size,
                                            input_shape=input_shape,
                                            dataset_type=dataset_type,
                                            marginals=args.marginals,
                                            device=args.device)
        elif args.architecture == 'conv':
            model = ConvDiagonalMarginalVAE(dimension_latent_space=args.latent_size,
                                            input_shape=input_shape,
                                            dataset_type=dataset_type,
                                            device=args.device)

    elif args.type_vae == 'copulaV2':
        if args.architecture == 'shallow':
            model = ShallowCopulaVAE(dimension_latent_space=args.latent_size,
                                     input_shape=input_shape,
                                     dataset_type=dataset_type,
                                     device=args.device)

    if model is None:
        error = f'We do not support {args.type_vae} with {args.architecture} as architecture'
        raise NotImplementedError(error)

    if output_dir is not None:
        with open(output_dir / 'model.txt', 'w') as f:
            print(model, file=f)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(args.device)

    return model
