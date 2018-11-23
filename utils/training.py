import math

import numpy as np

import torch
import torch.nn as nn
from torchvision.utils import make_grid

from utils.plot_utils import plot_grid_images_file


def train_epoch(model,
                epoch,
                loader,
                optimizer,
                beta,
                device=torch.device("cpu"),
                writer=None):
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
        #       So if for some reason I decide to start with CNN it still works.
        #       Inputs and outputs should always be  of the original shape

        xs = xs.view(loader.batch_size, -1).to(device)

        if batch_idx == 0 and writer is not None:

            recs = model.get_reconstruction(xs)

            shape = (-1,) + input_shape
            n = min(xs.size(0), 10)

            grid = make_grid(recs.reshape(shape), nrow=n)

            writer.add_image(f'reconstructions/train', grid, epoch)

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
                     writer=None):
    best_loss = math.inf
    early_stopping_strikes = 0

    model.train()

    for epoch in range(epochs):

        beta = compute_beta(epoch, warmup)

        loss_train, RE_train, KL_train = train_epoch(model=model,
                                                     loader=loader_train,
                                                     optimizer=optimizer,
                                                     epoch=epoch,
                                                     beta=beta,
                                                     device=device,
                                                     writer=writer)

        loss_val, RE_val, KL_val = validation_epoch(model=model,
                                                    beta=beta,
                                                    loader=loader_validation,
                                                    device=device)

        # logging

        if writer is not None:
            writer.add_scalar('loss/train', loss_train, epoch)
            writer.add_scalar('RE/train', RE_train, epoch)
            writer.add_scalar('KL/train', KL_train, epoch)

            writer.add_scalar('loss/val', loss_val, epoch)
            writer.add_scalar('RE/val', RE_val, epoch)
            writer.add_scalar('KL/val', KL_val, epoch)

        # Early stopping

        if loss_val < best_loss:

            early_stopping_strikes = 0
            best_loss = loss_val

        elif warmup is not None and epoch > warmup:

            early_stopping_strikes += 1
            if early_stopping_strikes >= early_stopping_tolerance:
                break

    return model
