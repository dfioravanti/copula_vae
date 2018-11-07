import math

import numpy as np
import torch
import torch.nn as nn

from utils.plot_utils import plot_grid_images_file


def train_epoch(model,
                epoch,
                loader,
                optimizer,
                loss_function,
                beta,
                device=torch.device("cpu"),
                output_dir=None):
    model.train()

    batch_losses = np.zeros(len(loader))
    batch_REs = np.zeros_like(batch_losses)
    batch_KLs = np.zeros_like(batch_losses)

    if isinstance(model, nn.DataParallel):
        input_shape = model.module.input_shape
    else:
        input_shape = model.input_shape

    for batch_idx, (xs, _) in enumerate(loader):

        optimizer.zero_grad()

        xs = xs.view(loader.batch_size, -1).to(device)
        xs_reconstructed, L_x = model(xs)

        if batch_idx == 0 and output_dir is not None:

            plot_reconstruction(xs[:10],
                                xs_reconstructed[:10],
                                input_shape,
                                epoch,
                                output_dir)

        loss, reconstruction_error, KL = calculate_loss(xs,
                                                        xs_reconstructed,
                                                        L_x,
                                                        loss_function=loss_function,
                                                        beta=beta)
        loss.backward()
        optimizer.step()

        batch_losses[batch_idx] = loss
        batch_REs[batch_idx] = reconstruction_error
        batch_KLs[batch_idx] = KL

    epoch_loss = np.average(batch_losses)
    epoch_RE = np.average(batch_REs)
    epoch_KLs = np.average(batch_KLs)

    return epoch_loss, epoch_RE, epoch_KLs


def validation_epoch(model,
                     loss_function,
                     beta,
                     loader,
                     device):
    model.eval()

    batch_losses = np.zeros(len(loader))
    batch_REs = np.zeros_like(batch_losses)
    batch_KLs = np.zeros_like(batch_losses)

    for batch_idx, (xs, _) in enumerate(loader):
        xs = xs.view(loader.batch_size, -1).to(device)
        xs_reconstructed, L_x = model(xs)

        loss, reconstruction_error, KL = calculate_loss(xs,
                                                        xs_reconstructed,
                                                        L_x,
                                                        loss_function=loss_function,
                                                        beta=beta)

        batch_losses[batch_idx] = loss
        batch_REs[batch_idx] = reconstruction_error
        batch_KLs[batch_idx] = KL

    val_loss = np.average(batch_losses)
    val_RE = np.average(batch_REs)
    val_KLs = np.average(batch_KLs)

    return val_loss, val_RE, val_KLs


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


def calculate_loss(xs, xs_reconstructed, L_x, loss_function, beta):

    k = L_x.shape[1]

    ixd_diag = np.diag_indices(k)
    diag_L_x = L_x[:, ixd_diag[0], ixd_diag[1]]

    RE = loss_function(xs, xs_reconstructed)
    tr_R = torch.sum(L_x ** 2, dim=(1, 2))
    tr_log_L = torch.sum(torch.log(diag_L_x), dim=1)
    KL = torch.mean((tr_R - k) / 2 - tr_log_L)
    return RE + beta * KL, RE, KL


def train_on_dataset(model,
                     loader_train,
                     loader_validation,
                     optimizer,
                     loss,
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
                                                                            loss_function=loss,
                                                                            epoch=epoch,
                                                                            beta=beta,
                                                                            device=device,
                                                                            output_dir=output_dir)

            val_loss[epoch], val_RE[epoch], val_KL[epoch] = validation_epoch(model=model,
                                                                           loss_function=loss,
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