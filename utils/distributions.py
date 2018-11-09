
import math
import numpy as np

import torch

# We need this to avoid numerical instability
tollerance = 1e-5

def gaussian_icdf(means, sigmas, values):

    """
    Compute the inverse cdf of a gaussian.
    The clamp is there to avoid numerical instability

    :param means:
    :param sigmas:
    :param values:

    :return:
    """

    values = torch.clamp(values, min=0 + tollerance, max=1 - tollerance)

    return means + sigmas * torch.erfinv(2 * values - 1) * math.sqrt(2)


def gaussian_0_I_cdf(values):

    return 0.5 * (1 + torch.erf((values) / math.sqrt(2)))


def gaussian_cdf(means, sigmas, values):

    dim = values.shape[0]

    means = means.repeat((dim, 1))
    sigmas = sigmas.repeat((dim, 1))

    return 0.5 * (1 + torch.erf((values - means) * sigmas.reciprocal() / math.sqrt(2)))


def log_normal_by_component(x, mean, log_var, average=False):

    #log_normal = -0.5 * np.log(2*np.pi) - log_var + torch.pow(x - mean, 2) / torch.exp(log_var))

    if average:
        return torch.mean(log_normal)
    else:
        return torch.sum(log_normal)


def log_normal_standard(x, average=False):

    log_normal = -0.5 * torch.pow(x, 2)

    if average:
        return torch.mean(log_normal)
    else:
        return torch.sum(log_normal)

# Density functions


def log_density_Normal(x, mean, log_var, average=False, reduce_dim=None):

    """

    :param x:
    :param mean:
    :param log_var:
    :param average:
    :param reduce_dim:
    :return:
    """

    log_densities = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))

    if average:
        return torch.mean(log_densities, reduce_dim)
    else:
        return torch.sum(log_densities, reduce_dim)


def log_density_standard_Normal(x, average=False, reduce_dim=None):

    """

    :param x:
    :param average:
    :param reduce_dim:
    :return:
    """

    log_densities = -0.5 * torch.pow(x, 2)

    if average:
        return torch.mean(log_densities, reduce_dim)
    else:
        return torch.sum(log_densities, reduce_dim)


def log_density_Bernoulli(x, mean, average=False, reduce_dim=None):

    """

    :param x:
    :param mean:
    :param average:
    :param reduce_dim:
    :return:
    """

    ps = torch.clamp(mean, min=(0 + tollerance), max=(1 - tollerance))
    qs = 1-ps

    log_densities = x*torch.log(ps) + (1-x)*torch.log(qs)

    if average:
        return torch.mean(log_densities, reduce_dim)
    else:
        return torch.sum(log_densities, reduce_dim)


def log_density_discretized_Logistic(x, mean, logvar, bin_size=1/256, average=False, reduce=True, reduce_dim=None):

    # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
    scale = torch.exp(logvar)
    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
    cdf_plus = torch.sigmoid(x + bin_size/scale)
    cdf_minus = torch.sigmoid(x)

    log_logist_256 = - torch.log(cdf_plus - cdf_minus + 1.e-7)

    if reduce:
        if average:
            return torch.mean(log_logist_256, reduce_dim)
        else:
            return torch.sum(log_logist_256, reduce_dim)
    else:
        return log_logist_256
