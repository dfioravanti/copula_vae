
import math

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


class OneToOne(nn.Module):

    def __init__(self, in_features, bias=True):
        super(OneToOne, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(in_features))
        self.mask = torch.eye(in_features)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):

        masked_weight = self.mask * self.weight
        return F.linear(input, masked_weight) + self.bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
