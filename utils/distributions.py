
import math

import numpy as np
import torch


def gaussian_icdf(means, sigmas, value):

    dim = value.shape[0]

    means = means.repeat((dim, 1))
    sigmas = sigmas.repeat((dim, 1))

    return means + sigmas * torch.erfinv(2 * value - 1) * math.sqrt(2)


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