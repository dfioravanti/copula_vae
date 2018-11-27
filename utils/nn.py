"""
    This file provides layers for our neural networks
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from utils.inverse_distibutions import gaussian_icdf, laplace_icdf, cauchy_icdf, exp_icdf, logNorm_icdf


class Flatten(nn.Module):
    """
    Layer that flatters the input
    """

    def forward(self, input):
        return input.view(input.size(0), -1)


class View(nn.Module):
    """
    Layer that reshapes a given input to a desired shape
    """

    def __init__(self, shape):
        super(View, self).__init__()

        self.shape = shape

    def forward(self, input):
        return input.view((-1,) + self.shape)


class ICDF(nn.Module):
    """
    Layer that applies a selected ICDF component by component to the input.
    """

    # TODO: Print the marginal name with extra_rep

    def __init__(self, in_features, marginals='gaussian'):

        super(ICDF, self).__init__()

        if marginals not in ['gaussian', 'laplace', 'log_norm', 'cauchy', 'exp']:
            raise ValueError(f'ICDF does not support {marginals}')

        self.loc = Parameter(torch.Tensor(in_features))
        self.scale = Parameter(torch.Tensor(in_features))
        self.distribution = marginals

        self.reset_parameters()

    def reset_parameters(self):

        self.loc.data = torch.zeros(self.loc.data.shape)
        self.scale.data = torch.zeros(self.scale.data.shape)

    def forward(self, x):

        # log(exp(x) + 1) is used to force the output to be positive

        if self.distribution == 'gaussian':
            return gaussian_icdf(p=x, loc=self.loc, scale=torch.log(torch.exp(self.scale) + 1))
        elif self.distribution == 'laplace':
            return laplace_icdf(p=x, loc=self.loc, scale=torch.log(torch.exp(self.scale) + 1))
        elif self.distribution == 'log_norm':
            return logNorm_icdf(p=x, loc=self.loc, scale=torch.log(torch.exp(self.scale) + 1))
        elif self.distribution == 'cauchy':
            return cauchy_icdf(p=x, loc=self.loc, scale=torch.log(torch.exp(self.scale) + 1))
        elif self.distribution == 'exp':
            return exp_icdf(p=x, rate=torch.log(torch.exp(self.scale) + 1))


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


class PositiveLinear(nn.Linear):
    r"""Applies a transformation to the incoming data of the following form: :math:`y_i = xlog(exp(A)+1)^T`
        where log and exp are elementwise operations.

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``

        Shape:
            - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
              additional dimensions
            - Output: :math:`(N, *, out\_features)` where all but the last dimension
              are the same shape as the input.

        Attributes:
            weight: the learnable weights of the module of shape
                `(out_features x in_features)`
            bias:   the learnable bias of the module of shape `(out_features)`

        Examples::

            >>> m = nn.PositiveLinear(20, 30)
            >>> input = torch.randn(128, 20)
            >>> output = m(input)
            >>> print(output.size())
        """

    def forward(self, input):
        transformed_weight = torch.clamp(self.weight, min=0)
        transformed_bias = torch.clamp(self.bias, min=0)
        return F.linear(input, transformed_weight, self.bias)
