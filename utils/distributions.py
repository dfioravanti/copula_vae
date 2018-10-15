
import math

import torch


def gaussian_icdf(means, sigmas, value):

    return means + sigmas * torch.erfinv(2 * value - 1) * math.sqrt(2)