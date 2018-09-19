
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class GatedDense(nn.Module):
    def __init__(self, input_size, output_size):
        super(GatedDense, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)

    def forward(self, x):

        h = self.h(x)
        g = self.sigmoid(self.g(x))

        return h * g