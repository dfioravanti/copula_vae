
import math

import torch


def gaussian_icdf(means, sigmas, values):

    return means + sigmas * torch.erfinv(2 * values - 1) * math.sqrt(2)


def gaussian_0_I_cdf(values):

    return 0.5 * (1 + torch.erf((values) / math.sqrt(2)))


def gaussian_cdf(means, sigmas, values):

    dim = values.shape[0]

    means = means.repeat((dim, 1))
    sigmas = sigmas.repeat((dim, 1))

    return 0.5 * (1 + torch.erf((values - means) * sigmas.reciprocal() / math.sqrt(2)))


def log_normal_by_component(x, mean, log_var, average=False):

    log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))

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