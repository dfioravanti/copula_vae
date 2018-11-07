
import math
import numpy as np

import torch


def gaussian_icdf(means, sigmas, values):

    values = torch.clamp(values, min=0.0001, max=0.9999)

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


def log_Logistic_256(x, mean, logvar, average=False, reduce=True, dim=None):

    bin_size = 1. / 256.

    # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
    scale = torch.exp(logvar)
    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
    cdf_plus = torch.sigmoid(x + bin_size/scale)
    cdf_minus = torch.sigmoid(x)

    log_logist_256 = - torch.log(cdf_plus - cdf_minus + 1.e-7)

    if reduce:
        if average:
            return torch.mean(log_logist_256, dim)
        else:
            return torch.sum(log_logist_256, dim)
    else:
        return log_logist_256
