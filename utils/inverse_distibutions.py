"""
    This file provides the functions to compute cdf and icdf of given distributions.
"""

import math

import torch

from utils.settings import tollerance


# CDF

def standard_normal_cdf(value):
    """
    Compute the componentwise cdf of a gaussian with 0 mean and I covariance matrix

    Parameters
    ----------
    value: float(s)
        Value to compute the cdf of

    Returns
    -------
    float(s):
        The cdf applied to value
    """
    return 0.5 * (1 + torch.erf(value / math.sqrt(2)))


# ICDF

def standard_normal_icdf(p):
    """
    Compute the componentwise inverse cdf of a gaussian with 0 mean and I covariance matrix.
    The clamp is there to avoid numerical instability

    Parameters
    ----------
    p: float(s)
        Probability value in [0,1]

    Returns
    -------
        float(s): quartile of p
    """

    p = torch.clamp(p, min=0 + tollerance, max=1 - tollerance)

    return torch.erfinv(2 * p - 1) * math.sqrt(2)


def normal_icdf(p, loc=0, scale=1):
    """
    Compute the componentwise inverse cdf of a gaussian with mean as loc and variance as scale.

    Parameters
    ----------
    p:  float(s)
        Probability value in [0,1]
    loc: float(s)
        The mean of the distribution
    scale: float(s) > 0
        The variance of the distribution

    Returns
    -------
        float(s): quartile of p
    """

    return loc + scale * standard_normal_icdf(p)


def logNorm_icdf(p, loc, scale):
    """
    Compute the componentwise inverse cdf of a logNormal with mean as loc and variance as scale.

    Parameters
    ----------
    p:  float(s)
        Probability value in [0,1]
    loc: float(s)
        The mean of the distribution
    scale: float(s) > 0
        The variance of the distribution

    Returns
    -------
        float(s): quartile of p
    """

    return torch.exp(loc + scale * standard_normal_icdf(p))


def laplace_icdf(p, loc, scale):
    """
        Compute the componentwise inverse cdf of a laplace with mean as loc and b as scale.

        Parameters
        ----------
        p:  float(s)
            Probability value in [0,1]
        loc: float(s)
            The mean of the distribution
        scale: float(s) > 0
            The variance of the distribution

        Returns
        -------
            float(s): quartile of p
    """

    term = p - 0.5
    return loc - scale * term.sign() * torch.log1p(-2 * term.abs())


def cauchy_icdf(p, loc, scale):
    """
        Compute the componentwise inverse cdf of a cauchy with mean as loc and variance as scale.

        Parameters
        ----------
        p:  float(s)
            Probability value in [0,1]
        loc: float(s)
            The mean of the distribution
        scale: float(s) > 0
            The variance of the distribution

        Returns
        -------
            float(s): quartile of p
    """

    return torch.tan(math.pi * (p - 0.5)) * scale + loc


def exp_icdf(p, rate):
    """
        Compute the componentwise inverse cdf of a exponential with mean as loc and b as scale.

        Parameters
        ----------
        p:  float(s)
            Probability value in [0,1]
        rate: float(s) > 0
            The rate of the distribution

        Returns
        -------
            float(s): quartile of p
    """

    return -torch.log(1 - p) / rate
