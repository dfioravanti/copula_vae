from models.basic_model import BaseVAE

import numpy as np
import torch
from torch import nn


class VAE(BaseVAE):

    def __init__(self,
                 number_latent_variables,
                 input_size,
                 encoder_output_size,
                 device=torch.device("cpu")):

        super(VAE, self).__init__(number_latent_variables=number_latent_variables,
                                  input_size=input_size,
                                  device=device)

        self.encoder_output_size = encoder_output_size

        # Encoder q(z|x)

        self.q_z_layers = nn.Sequential(
            nn.ReLU(nn.Linear(np.prod(self.input_size), 300)),
            nn.ReLU(nn.Linear(300, self.encoder_output_size))
        )

        self.q_z_mean = nn.Linear(self.encoder_output_size, self.number_latent_variables)
        self.q_z_log_var = nn.Linear(self.encoder_output_size, self.number_latent_variables)

        # Decoder p(x|z)

if __name__ == '__main__':

    vae = VAE(number_latent_variables=1,
              input_size=1,
              encoder_output_size=20)
