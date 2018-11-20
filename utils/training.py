import math

import numpy as np
import torch
import torch.nn as nn

from utils.plot_utils import plot_grid_images_file


def train_epoch(model,
                epoch,
                loader,
                optimizer,
                beta,
                device=torch.device("cpu"),
                output_dir=None):
    model.train()

    batch_losses = np.zeros(len(loader))
    batch_RE = np.zeros_like(batch_losses)
    batch_KL = np.zeros_like(batch_losses)

    if isinstance(model, nn.DataParallel):
        input_shape = model.module.input_shape
    else:
        input_shape = model.input_shape

    for batch_idx, (xs, _) in enumerate(loader):

        optimizer.zero_grad()

        # TODO: move reshape inside the model.
        # So if for some reason I decide to start with CNN it still works.
        # Inputs and outputs should always be  of the original shape

        xs = xs.view(loader.batch_size, -1).to(device)

        if batch_idx == 0 and output_dir is not None:

            xs_reconstructed = model.get_reconstruction(xs)
            plot_reconstruction(xs[:10],
                                xs_reconstructed[:10],
                                input_shape,
                                epoch,
                                output_dir)

        loss, NLL, KL = model.calculate_loss(xs, beta=beta)
        loss.backward()
        optimizer.step()

        batch_losses[batch_idx] = loss
        batch_RE[batch_idx] = NLL
        batch_KL[batch_idx] = KL

    epoch_loss = np.average(batch_losses)
    epoch_RE = np.average(batch_RE)
    epoch_KL = np.average(batch_KL)

    return epoch_loss, epoch_RE, epoch_KL


def validation_epoch(model,
                     beta,
                     loader,
                     device):
    model.eval()

    batch_losses = np.zeros(len(loader))
    batch_RE = np.zeros_like(batch_losses)
    batch_KL = np.zeros_like(batch_losses)

    for batch_idx, (xs, _) in enumerate(loader):
        xs = xs.view(loader.batch_size, -1).to(device)

        loss, NLL, KL = model.calculate_loss(xs, beta=beta)

        batch_losses[batch_idx] = loss
        batch_RE[batch_idx] = NLL
        batch_KL[batch_idx] = KL

    val_loss = np.average(batch_losses)
    val_RE = np.average(batch_RE)
    val_KL = np.average(batch_KL)

    return val_loss, val_RE, val_KL


def compute_beta(epoch, warmup):
    if warmup is None:
        beta = 1
    else:
        beta = min(epoch / warmup, 1)

    return beta


def plot_reconstruction(xs,
                        xs_reconstructed,
                        input_shape,
                        epoch,
                        output_dir):

    xs = xs.view((-1,) + input_shape)
    xs_reconstructed = xs_reconstructed.view((-1,) + input_shape)

    image_dir = output_dir / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    filename = image_dir / f'Epoch {epoch}'

    plot_grid_images_file(xs.to('cpu').detach().numpy()[0:10, :],
                          xs_reconstructed.to('cpu').detach().numpy()[0:10, :],
                          columns=10,
                          filename=filename)


def train_on_dataset(model,
                     loader_train,
                     loader_validation,
                     optimizer,
                     epochs=50,
                     warmup=None,
                     early_stopping_tolerance=10,
                     device=torch.device("cpu"),
                     output_dir=None):

    best_loss = math.inf
    early_stopping_strikes = 0

    train_loss = np.zeros(epochs)
    train_RE = np.zeros_like(train_loss)
    train_KL = np.zeros_like(train_loss)
    val_loss = np.zeros_like(train_loss)
    val_RE = np.zeros_like(train_loss)
    val_KL = np.zeros_like(train_loss)

    path_output_file = output_dir / 'output.txt'

    with open(path_output_file, 'w', buffering=1) as f:

        for epoch in range(epochs):

            beta = compute_beta(epoch, warmup)

            f.write(f'Training with beta = {beta}\n')

            train_loss[epoch], train_RE[epoch], train_KL[epoch] = train_epoch(model=model,
                                                                            loader=loader_train,
                                                                            optimizer=optimizer,
                                                                            epoch=epoch,
                                                                            beta=beta,
                                                                            device=device,
                                                                            output_dir=output_dir)

            val_loss[epoch], val_RE[epoch], val_KL[epoch] = validation_epoch(model=model,
                                                                           beta=beta,
                                                                           loader=loader_validation,
                                                                           device=device)

            f.write(f'epoch: {epoch}/{epochs}\n'
                    f'train loss: {train_loss[epoch]} and val loss: {val_loss[epoch]}\n'
                    f'train RE: {train_RE[epoch]} and val RE: {val_RE[epoch]}\n'
                    f'train KL: {train_KL[epoch]} and val KL: {val_KL[epoch]}\n\n')

            if val_loss[epoch] < best_loss:

                early_stopping_strikes = 0
                best_loss = val_loss[epoch]

            elif warmup is not None and epoch > warmup:

                early_stopping_strikes += 1
                if early_stopping_strikes >= early_stopping_tolerance:
                    break

    return model, (train_loss, train_RE, train_KL), (val_loss, val_RE, val_KL)
