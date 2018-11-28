"""
    This file contains all the functions used to compute copulas and related auxiliary functions
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import levy_stable

import torch

from utils.inverse_distibutions import standard_normal_cdf
from utils.settings import tollerance


def partition(number, classes):
    sizes = np.zeros(classes, dtype=np.int)

    sizes[:] = number // classes

    for i in range(number % classes):
        sizes[i] += 1

    return sizes


def psi_theta(psi, theta):
    return lambda t: psi(t, theta)


def psi_gumbel(t, theta):
    return np.exp(-t ** (1 / theta))


def marshall_olkin_sampling(psi_theta, F, d):
    v = F.rvs(size=1)
    xs = uniform(size=d)

    return psi_theta(-np.log(xs) / v)


def sampling_from_gumbel_copula(theta, d):
    F = levy_stable(1 / theta, 1, 0, (np.cos(np.pi / (2 * theta)) ** theta))
    psi = psi_theta(psi_gumbel, theta)
    return marshall_olkin_sampling(psi, F, d)


def sampling_from_gumbel_partially_nested_copula(d, s, thetas):
    if not np.all(thetas >= 1):
        raise ValueError("All the thetas must be bigger than one")

    if not np.all(np.diff(thetas) >= 0):
        raise ValueError("Thetas must be increasing")

    if not len(thetas) == s + 1:
        raise ValueError("Length of thetas must be equas to s+1")

    sizes = partition(d, s)
    xs = np.zeros(d)

    psi_0 = psi_theta(psi_gumbel, thetas[0])

    v_0 = levy_stable(1 / thetas[0], 1, 0, (np.cos(np.pi / (2 * thetas[0])) ** thetas[0])).rvs(size=1)

    used_elemets = 0

    for i in range(1, s + 1):
        future_used_elements = used_elemets + sizes[i - 1]

        theta_i = thetas[i] / thetas[0]
        xs[used_elemets:future_used_elements] = sampling_from_gumbel_copula(theta_i, sizes[i - 1])

        used_elemets = future_used_elements

    return psi_0(-np.log(xs) / v_0)


def sampling_from_gausiann_copula(L, d, n=1):
    """
    Samples (n, d) elements from a gaussian copula with correlation matrix R = LL^t

    For one sample the algorithm used is the following
        1) Decompose R with Cholesky (we assume that the decomposition is computed outside)
        2) a) Generate a vector Z = (Z_1,...,Z_d)^t of independent standard normal variates
           b) X = LZ
           c) return (Phi(X_1), ..., Phi(X_d))

    more info https://stats.stackexchange.com/questions/37424/how-to-simulate-from-a-gaussian-copula

    Parameters
    ----------
    L: FloatTensor
        Positively defined lower triangular matrix obtained by the Cholesky decomposition of our
        desired correlation matrix R
    d: int
        Dimension of [0,1]^d where we are sampling the copula
    n: int
        Number of samples to be samples

    Returns
    -------
    FloatTensor tuple
        (n, d) elements coming from the copula
    """

    size = (n, 1, d)
    L = torch.clamp(L, min=0 + tollerance, max=1 - tollerance)

    zs = torch.randn(size=size)
    if L.is_cuda:
        zs = zs.to(L.get_device())

    xs = (zs @ L).view(n, -1)

    return standard_normal_cdf(xs)


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from utils.inverse_distibutions import normal_icdf

    np.random.seed(123)
    d = 3
    n = 60000

    L = np.matrix([[1, 0, 0],
                   [0, 1, 0],
                   [0.5, 0, 1]]
                  )
    print(L @ L.T)
    L = torch.tensor(L).float()

    xs = sampling_from_gausiann_copula(L, d, n)
    zs = normal_icdf(xs)

    fig = plt.figure()
    sns.distplot(zs[:, 0])
    sns.distplot(zs[:, 1])
    sns.distplot(zs[:, 2])
    plt.show()
    zs = pd.DataFrame(zs.numpy())
    sns.pairplot(zs)
    plt.show()
