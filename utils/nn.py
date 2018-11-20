
import math

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from utils.inverse_distibutions import gaussian_icdf, laplace_icdf


class Flatten(nn.Module):

    """
    Layer that flatters the input
    """

    def forward(self, input):
        return input.view(input.size(0), -1)


class Reshape(nn.Module):

    """
    Layer that reshapes a given input to a desired shape
    """

    def __init__(self, shape):
        super(Reshape, self).__init__()

        self.shape = shape

    def forward(self, input):
        return input.view((-1,) + self.shape)


class ICDF(nn.Module):

    def __init__(self, in_features, marginals='gaussian'):

        super(ICDF, self).__init__()

        if marginals not in ['gaussian', 'laplace', 'log_norm']:
            raise ValueError(f'ICDF does not support {marginals}')

        self.loc = Parameter(torch.Tensor(in_features))
        self.scale = Parameter(torch.Tensor(in_features))
        self.distribution = marginals

        self.reset_parameters()

    def reset_parameters(self):

        self.loc.data = torch.zeros(self.loc.data.shape)
        self.scale.data = torch.zeros(self.scale.data.shape)

    def forward(self, x):

        if self.distribution == 'gaussian':
            return gaussian_icdf(values=x, loc=self.loc, scale=torch.exp(self.scale))
        if self.distribution == 'laplace':
            return laplace_icdf(values=x, loc=self.loc, scale=torch.exp(self.scale))
        if self.distribution == 'log_norm':
            return gaussian_icdf(values=torch.log(x), loc=self.loc, scale=torch.exp(self.scale))


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

        if torch.cuda:
            self.mask = self.mask.to(self.weight.get_device())

        masked_weight = self.mask * self.weight
        return F.linear(input, masked_weight) + self.bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
