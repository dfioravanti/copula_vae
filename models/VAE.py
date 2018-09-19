from models.basic_model import BaseVAE

import math

import numpy as np
import torch
from torch import nn
from utils.nn import GatedDense


class VAE(BaseVAE):

    def __init__(self,
                 number_latent_variables,
                 input_shape,
                 encoder_output_size=300,
                 output_dir=None,
                 device=torch.device("cpu")):

        super(VAE, self).__init__(number_latent_variables=number_latent_variables,
                                  input_shape=input_shape,
                                  output_dir=output_dir,
                                  device=device)

        self.encoder_output_size = encoder_output_size

        # Encoder q(z|x)

        self.q_z_layers = nn.Sequential(
            GatedDense(np.prod(self.input_shape), 300),
            GatedDense(300, self.encoder_output_size)
        )

        self.mean = nn.Linear(self.encoder_output_size, self.number_latent_variables)
        self.log_var = nn.Sequential(
            nn.Linear(self.encoder_output_size, self.number_latent_variables),
            nn.Hardtanh(min_val=-6, max_val=2)
        )

        # Decoder p(x|z)

        self.p_x_layers = nn.Sequential(
            GatedDense(self.number_latent_variables, 300),
            GatedDense(300, 300)
        )

        self.output_decoder = nn.Sequential(
            nn.Linear(300, np.prod(self.input_shape)),
            nn.Sigmoid()
        )

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        mean_z_x, log_var_z_x = self.q_z(x)
        z_x = self.sampling_normal_with_reparametrization(mean_z_x, log_var_z_x)

        # x_mean = p(x|z)
        x_recontructed = self.p_x(z_x)

        return x_recontructed, mean_z_x, log_var_z_x

    def q_z(self, x):
        x = self.q_z_layers(x)

        return self.mean(x), self.log_var(x)

    def p_x(self, z):
        z = self.p_x_layers(z)

        return self.output_decoder(z)

    def calculate_loss(self, xs, beta=1, loss=nn.L1Loss()):
        xs_recontructed, mean_z_x, log_var_z_x = self.forward(xs)

        RE = loss(xs_recontructed, xs)
        KL = torch.mean(-0.5 * torch.sum(1 + log_var_z_x - mean_z_x.pow(2) - log_var_z_x.exp()))

        return RE + beta * KL, RE, KL

    def train_on_dataset(self,
                         loader_train,
                         loader_validation,
                         optimizer,
                         epochs=50,
                         warmup=None,
                         verbose=True,
                         early_stopping_tolerance=10):

        best_loss = math.inf
        early_stopping_strikes = 0

        for epoch in range(epochs):

            epoch_train_loss, epoch_train_RE, epoch_train_KLs = self.train_epoch(epoch=epoch,
                                                                                 loader=loader_train,
                                                                                 optimizer=optimizer,
                                                                                 warmup=warmup,
                                                                                 verbose=verbose)

            epoch_val_loss, epoch_val_RE, epoch_val_KLs = self.validation_epoch(epoch=epoch,
                                                                                loader=loader_validation)

            if verbose:
                print(f'epoch: {epoch}/{epochs} => train loss: {epoch_train_loss} and val loss: {epoch_val_loss}')

            if epoch_val_loss < best_loss:

                early_stopping_strikes = 0
                best_loss = epoch_val_loss

            elif warmup is not None and epoch > warmup:

                early_stopping_strikes += 1
                if early_stopping_strikes >= early_stopping_tolerance:
                    break
