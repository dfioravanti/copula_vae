import math

import numpy as np

import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image


def train_epoch(model, epoch, loader, optimizer, beta, rec_img_path=None, writer=None, device=torch.device("cpu")):
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

            save_image(tensor=recs.reshape(shape),
                       filename=rec_img_path / f'epoch_{epoch:0=2d}.png',
                       nrow=n)

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


def validation_epoch(model, beta, loader, device=torch.device("cpu")):
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


def train_on_dataset(model, loader_train, loader_validation, optimizer, epochs=50, warmup=None,
                     early_stopping_tolerance=10, output_dir=None, writer=None, checkpoint=None,
                     device=torch.device("cpu")):
    best_loss = math.inf
    early_stopping_strikes = 0
    starting_epoch = 0

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']

    model.train()

    rec_img_path = output_dir / 'rec_img'
    rec_img_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(starting_epoch, epochs):

        beta = compute_beta(epoch, warmup)

        loss_train, RE_train, KL_train = train_epoch(model=model, epoch=epoch, loader=loader_train, optimizer=optimizer,
                                                     beta=beta, rec_img_path=rec_img_path, writer=writer, device=device)

        loss_val, RE_val, KL_val = validation_epoch(model=model, beta=beta, loader=loader_validation, device=device)

        print(f'epoch: {epoch}/{epochs}\n'
              f'beta: {beta}\n'
              f'loss train: {loss_train} and loss val: {loss_val}\n'
              f'RE train: {RE_train} and RE val: {RE_val}\n'
              f'KL train: {KL_train} and KL val: {KL_val}\n')

        # logging

        if writer is not None:
            writer.add_scalar(tag='loss/train', scalar_value=loss_train, global_step=epoch)
            writer.add_scalar(tag='loss/val', scalar_value=loss_val, global_step=epoch)

            writer.add_scalar(tag='RE/train', scalar_value=RE_train, global_step=epoch)
            writer.add_scalar(tag='RE/val', scalar_value=RE_val, global_step=epoch)

            writer.add_scalar(tag='KL/train', scalar_value=KL_train, global_step=epoch)
            writer.add_scalar(tag='KL/val', scalar_value=KL_val, global_step=epoch)

        # Early stopping and checkpointing best model

        if loss_val < best_loss:

            early_stopping_strikes = 0
            best_loss = loss_val

            if warmup is None or epoch > warmup:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_train,
                }, output_dir / 'best.tar')

        elif warmup is not None and epoch > warmup:

            early_stopping_strikes += 1
            if early_stopping_strikes >= early_stopping_tolerance:
                break

    return model
