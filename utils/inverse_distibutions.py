import math

import torch

from utils.distributions import tollerance

# We need this to avoid numerical instability
tollerance = 1e-7


def gaussian_icdf(values, loc, scale):

    """
    Compute the inverse cdf of a gaussian.
    The clamp is there to avoid numerical instability

    :param loc:
    :param scale:
    :param values:

    :return:
    """

    values = torch.clamp(values, min=0 + tollerance, max=1 - tollerance)

    return loc + scale * torch.erfinv(2 * values - 1) * math.sqrt(2)


def gaussian_0_I_icdf(values):

    """
    Compute the inverse cdf of a gaussian with 0 mean and I covariance matrix.
    The clamp is there to avoid numerical instability

    :param means:
    :param sigmas:
    :param values:

    :return:
    """

    values = torch.clamp(values, min=0 + tollerance, max=1 - tollerance)

    return torch.erfinv(2 * values - 1) * math.sqrt(2)


def gaussian_0_I_cdf(values):

    return 0.5 * (1 + torch.erf((values) / math.sqrt(2)))


def laplace_icdf(values, loc, scale):

    term = values - 0.5
    return loc - scale * term.sign() * torch.log1p(1-2 * term.abs())
