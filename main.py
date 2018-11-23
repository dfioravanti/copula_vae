import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam

from loaders.load_funtions import load_dataset
from utils.parsing import get_args, get_model
from utils.training import train_on_dataset


def main(args):
    train_loader, validation_loader, test_loader, \
    input_shape, dataset_type = load_dataset(args)

    writer = SummaryWriter(args.log_dir) if args.log_dir is not None else None
    if writer is not None:
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)))

    model = get_model(args=args,
                      input_shape=input_shape,
                      dataset_type=dataset_type)

    model = train_on_dataset(model=model,
                             loader_train=train_loader,
                             loader_validation=validation_loader,
                             optimizer=Adam(model.parameters(), lr=args.lr),
                             epochs=args.epochs,
                             warmup=args.warmup,
                             early_stopping_tolerance=args.early_stopping_epochs,
                             device=args.device,
                             writer=writer)

    torch.save(model.state_dict(), args.output_dir / 'model_trained')

    results_test = model.calculate_likelihood(train_loader,
                                              number_samples=args.S,
                                              output_dir=args.output_dir
                                              )

    if writer is not None:
        writer.add_scalar('test/NLL', results_test)


if __name__ == '__main__':

    args = get_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    main(args)
