import math

import numpy as np
import torch
import torch.nn as nn

from utils.plot_utils import plot_grid_images_file
from models.copula_VAE import compute_KL


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

    for batch_idx, (xs, _) in enumerate(loader):

        optimizer.zero_grad()

        xs = xs.view(loader.batch_size, -1).to(device)
        xs_reconstructed, mean_z_x, log_var_z_x = model(xs)

        if batch_idx == 0 and output_dir is not None:
            print(f'plotting')

            if isinstance(model, nn.DataParallel):
                input_shape = model.module.input_shape
            else:
                input_shape = model.input_shape

            plot_reconstruction(xs_reconstructed[:10],
                                input_shape,
                                epoch,
                                output_dir)

        loss, reconstruction_error, KL = calculate_loss(xs,
                                                        xs_reconstructed,
                                                        loss_function=loss_function,
                                                        beta=beta,
                                                        mean_z_x=mean_z_x,
                                                        log_var_z_x=log_var_z_x,
                                                        number_latent_variables=model.number_latent_variables,
                                                        number_samples_kl=1)
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
        xs_reconstructed, mean_z_x, log_var_z_x = model(xs)

        loss, reconstruction_error, KL = calculate_loss(xs,
                                                        xs_reconstructed,
                                                        loss_function=loss_function,
                                                        beta=beta,
                                                        mean_z_x=mean_z_x,
                                                        log_var_z_x=log_var_z_x,
                                                        number_latent_variables=model.number_latent_variables,
                                                        number_samples_kl=1)

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


def plot_reconstruction(xs_reconstructed,
                        input_shape,
                        epoch,
                        output_dir):
    xs_reconstructed = xs_reconstructed.view((-1,) + input_shape)

    filename = output_dir / f'Epoch {epoch}'

    plot_grid_images_file(xs_reconstructed.to('cpu').detach().numpy()[0:10, :],
                          columns=5,
                          filename=filename)


def calculate_loss(xs, xs_reconstructed, loss_function, beta,
                   mean_z_x, log_var_z_x, number_latent_variables, number_samples_kl):

    RE = loss_function(xs, xs_reconstructed)
    KL = compute_KL(mean_z_x, log_var_z_x, number_latent_variables, number_samples_kl)

    return RE + beta * KL, RE, KL


def train_on_dataset(model,
                     loader_train,
                     loader_validation,
                     optimizer,
                     loss,
                     epochs=50,
                     warmup=None,
                     verbose=True,
                     early_stopping_tolerance=10,
                     device=torch.device("cpu"),
                     output_dir=None):
    best_loss = math.inf
    early_stopping_strikes = 0

    for epoch in range(epochs):

        beta = compute_beta(epoch, warmup)
        if verbose:
            print(f'Training with beta = {beta}')

        epoch_train_loss, epoch_train_RE, epoch_train_KLs = train_epoch(model=model,
                                                                        loader=loader_train,
                                                                        optimizer=optimizer,
                                                                        loss_function=loss,
                                                                        epoch=epoch,
                                                                        beta=beta,
                                                                        device=device,
                                                                        output_dir=output_dir)

        epoch_val_loss, epoch_val_RE, epoch_val_KLs = validation_epoch(model=model,
                                                                       loss_function=loss,
                                                                       beta=beta,
                                                                       loader=loader_validation,
                                                                       device=device)

        if verbose:
            print(f'epoch: {epoch}/{epochs} => train loss: {epoch_train_loss} and val loss: {epoch_val_loss}')

        if epoch_val_loss < best_loss:

            early_stopping_strikes = 0
            best_loss = epoch_val_loss

        elif warmup is not None and epoch > warmup:

            early_stopping_strikes += 1
            if early_stopping_strikes >= early_stopping_tolerance:
                break
