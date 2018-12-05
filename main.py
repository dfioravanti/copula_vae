import sys
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam

from loaders.load_funtions import load_dataset
from utils.parsing import get_args_and_setting_up, get_model
from utils.training import train_on_dataset


def main(args):
    train_loader, validation_loader, test_loader, \
    input_shape, dataset_type = load_dataset(args)

    print('Loaded datased')

    writer = SummaryWriter(str(args.log_dir)) if args.log_dir is not None else None
    if writer is not None:
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)))

    model = get_model(args=args,
                      input_shape=input_shape,
                      dataset_type=dataset_type,
                      output_dir=args.output_dir)
    optimizer = Adam(model.parameters(), lr=args.lr)

    model = train_on_dataset(model=model, loader_train=train_loader, loader_validation=validation_loader,
                             optimizer=optimizer, epochs=args.epochs, warmup=args.warmup,
                             grace_early_stopping=args.grace_early_stopping, output_dir=args.output_dir,
                             frequency_checkpoints=args.frequency_checkpoints, checkpoint_dir=args.checkpoint_dir,
                             writer=writer, device=args.device)

    torch.save(model.state_dict(), args.output_dir / 'model_trained')

    results_test = model.calculate_likelihood(test_loader,
                                              number_samples=args.samples_ll,
                                              writer=writer)

    if writer is not None:
        writer.add_scalar('test/NLL', results_test)
        writer.export_scalars_to_json(args.log_dir / 'scalars.json')
        writer.close()


if __name__ == '__main__':

    # Parse the config file

    args = get_args_and_setting_up()

    # Set random seed

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Decide if to redirect stdout to file or not
    # needed if printing on terminal is not available

    if args.to_file:
        with open(args.output_dir / 'output.txt', 'w', buffering=1) as f:
            sys.stdout = f
            main(args)
    else:
        main(args)
