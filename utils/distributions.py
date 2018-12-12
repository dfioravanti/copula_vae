import math
from numbers import Number

import torch

from utils.settings import tollerance
from utils.inverse_distibutions import standard_normal_icdf
from utils.operations import batch_inverse, batch_eye_like


def gaussian_cdf(means, sigmas, values):
    dim = values.shape[0]

    means = means.repeat((dim, 1))
    sigmas = sigmas.repeat((dim, 1))

    return 0.5 * (1 + torch.erf((values - means) * sigmas.reciprocal() / math.sqrt(2)))


# Density functions

def log_density_gaussian_copula(xs, R_inv):
    """
    Computes the logarithm of the density function of a gaussian copula
    with correlation matrix R. Without the additive constant!

    See https://en.wikipedia.org/wiki/Copula_(probability_theory)#Gaussian_copula for the formula

    Parameters
    ----------
    xs: Tensor
        Input values
    R_inv: Tensor
        The inverse of correlation matrix R

    Returns
    -------
    Tensor:
        The log density without the additive constant

    """

    eye = batch_eye_like(xs)
    R_inv = batch_inverse(R_inv)
    R_inv = R_inv.t() @ R_inv

    icdf_gaussian_xs = standard_normal_icdf(xs)

    return -0.5 * icdf_gaussian_xs.t() @ (R_inv - eye) @ icdf_gaussian_xs


def log_density_normal(x, mean=0, var=1, average=False, reduce_dim=None):
    """

    :param x:
    :param mean:
    :param var:
    :param average:
    :param reduce_dim:
    :return:
    """

    if isinstance(var, Number):
        var = torch.tensor(var).float()
        if x.is_cuda:
            var = var.to(x.get_device())

    log_densities = -0.5 * (torch.log(var) + torch.pow(x - mean, 2) / var)

    if reduce_dim is None:
        return log_densities
    elif average:
        return torch.mean(log_densities, reduce_dim)
    else:
        return torch.sum(log_densities, reduce_dim)


def log_density_standard_normal(x, average=False, reduce_dim=None):
    """

    :param x:
    :param average:
    :param reduce_dim:
    :return:
    """

    log_densities = -0.5 * torch.pow(x, 2)

    if reduce_dim is None:
        return log_densities
    elif average:
        return torch.mean(log_densities, reduce_dim)
    else:
        return torch.sum(log_densities, reduce_dim)


def log_density_log_normal(x, mean=0, var=1, average=False, reduce_dim=None):
    """

    :param x:
    :param mean:
    :param var:
    :param average:
    :param reduce_dim:
    :return:
    """

    return log_density_normal(torch.log(x), mean, var, average, reduce_dim)


def log_density_laplace(value, loc=0, scale=1, average=False, reduce_dim=None):
    """

    Parameters
    ----------
    value
    scale
    loc
    average
    reduce_dim

    Returns
    -------

    """

    if isinstance(scale, Number):
        var = torch.tensor(scale).float()
        if value.is_cuda:
            scale = var.to(value.get_device())

    log_densities = torch.log(2 * scale) - torch.abs(value - loc) / scale

    if reduce_dim is None:
        return log_densities
    elif average:
        return torch.mean(log_densities, reduce_dim)
    else:
        return torch.sum(log_densities, reduce_dim)


def log_density_bernoulli(x, ps, average=False, reduce_dim=None):
    """

    Function to compute the log density for a vector n given a vector of n probability.
    We assume that all the components are iid Bernoulli with probability ps[i]

    :param x: Values where we want to evaluate the log density
    :param ps: The probability p for the ith component.
    :param average: True if we should return the average density over x
    :param reduce_dim: If we pass multiple xs we need to set this to 1 in order to average or sum correctly
    :return:
    """

    ps = torch.clamp(ps, min=(0 + tollerance), max=(1 - tollerance))
    qs = 1 - ps

    log_densities = x * torch.log(ps) + (1 - x) * torch.log(qs)

    if average:
        return torch.mean(log_densities, reduce_dim)
    else:
        return torch.sum(log_densities, reduce_dim)


def log_density_discretized_Logistic(x, mean, logvar, bin_size=1 / 256, average=False, reduce=True, reduce_dim=None):
    # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
    scale = torch.exp(logvar)
    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
    cdf_plus = torch.sigmoid(x + bin_size / scale)
    cdf_minus = torch.sigmoid(x)

    log_logist_256 = - torch.log(cdf_plus - cdf_minus + 1.e-7)

    if reduce:
        if average:
            return torch.mean(log_logist_256, reduce_dim)
        else:
            return torch.sum(log_logist_256, reduce_dim)
    else:
        return log_logist_256


def kl_normal_and_standard_normal(means, log_vars):
    """
    Compute the KL between two X = N(means, exp(log_vars)I) and Z = N(0, I)
    the formula used can be found at page 41 of Kingma "Variational inference & deep learning".

    Parameters
    ----------
    means: Tensors
        (batch_size, n) numbers representing the mean of the component i of the gaussian X
    log_vars: Tensor
        (batch_size, n) numbers representing the log variance of the component i of the gaussian X
    Returns
    -------
        Tensor:
        batch_size numbers representing KL(X, Z)
    """

    kl = -0.5 * (1 + 2 * log_vars - means ** 2 - torch.exp(log_vars) ** 2)

    return torch.sum(kl, dim=1)
