import math

import torch

from utils.distributions import tollerance

# We need this to avoid numerical instability
tollerance = 1e-7


# CDF

def standard_normal_cdf(values):
    return 0.5 * (1 + torch.erf((values) / math.sqrt(2)))


# ICDF

def standard_normal_icdf(p):
    """
    Compute the inverse cdf of a gaussian with 0 mean and I covariance matrix.
    The clamp is there to avoid numerical instability

    Parameters
    ----------
    p: float
        Probability value in [0,1]

    Returns
    -------
        float: quartile of p
    """

    p = torch.clamp(p, min=0 + tollerance, max=1 - tollerance)

    return torch.erfinv(2 * p - 1) * math.sqrt(2)


def gaussian_icdf(p, loc, scale):
    """
    Compute the inverse cdf of a gaussian with mean as loc and variance as scale.

    Parameters
    ----------
    p:  float
        Probability value in [0,1]
    loc: float
        The mean of the distribution
    scale: Float > 0
        The variance of the distribution

    Returns
    -------
        float: quartile of p
    """

    return loc + scale * standard_normal_icdf(p)


def logNorm_icdf(p, loc, scale):
    return torch.exp(loc + scale * standard_normal_icdf(p))


def laplace_icdf(p, loc, scale):
    term = p - 0.5
    return loc - scale * term.sign() * torch.log1p(-2 * term.abs())


def cauchy_icdf(p, loc, scale):
    return torch.tan(math.pi * (p - 0.5)) * scale + loc


def exp_icdf(p, rate):
    return -torch.log(1 - p) / rate
