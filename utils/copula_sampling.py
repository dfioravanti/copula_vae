import numpy as np
from numpy.random import uniform
from scipy.stats import levy_stable
from scipy import stats

import torch
from utils.distributions import gaussian_0_I_cdf


def partition(number, classes):

    sizes = np.zeros(classes, dtype=np.int)

    sizes[:] = number // classes

    for i in range(number % classes):
        sizes[i] += 1

    return sizes


def psi_theta(psi, theta):
    return lambda t: psi(t, theta)


def psi_gumbel(t, theta):
    return np.exp(-t ** (1/theta))


def marshall_olkin_sampling(psi_theta, F, d):
    v = F.rvs(size=1)
    xs = uniform(size=d)

    return psi_theta(-np.log(xs) / v)


def sampling_from_gumbel_copula(theta, d):
    F = levy_stable(1/theta, 1, 0, (np.cos(np.pi/(2*theta)) ** theta))
    psi = psi_theta(psi_gumbel, theta)
    return marshall_olkin_sampling(psi, F, d)


def sampling_from_gumbel_partially_nested_copula(d, s, thetas):

    if not np.all(thetas >= 1):
        raise ValueError("All the thetas must be bigger than one")

    if not np.all(np.diff(thetas) >= 0):
        raise ValueError("Thetas must be increasing")

    if not len(thetas) == s+1:
        raise ValueError("Length of thetas must be equas to s+1")

    sizes = partition(d, s)
    xs = np.zeros(d)

    psi_0 = psi_theta(psi_gumbel, thetas[0])

    v_0 = levy_stable(1 / thetas[0], 1, 0, (np.cos(np.pi / (2 * thetas[0])) ** thetas[0])).rvs(size=1)

    used_elemets = 0

    for i in range(1, s+1):

        future_used_elements = used_elemets + sizes[i-1]

        theta_i = thetas[i] / thetas[0]
        xs[used_elemets:future_used_elements] = sampling_from_gumbel_copula(theta_i, sizes[i-1])

        used_elemets = future_used_elements

    return psi_0(-np.log(xs) / v_0)


def sampling_from_gausiann_copula(L, batch_size, n):

    size = (batch_size, 1, n)

    es = torch.randn(size=size)
    if L.is_cuda:
        es = es.to(L.get_device())

    xs = (es @ L).view(batch_size, -1)

    return gaussian_0_I_cdf(xs)


if __name__ == '__main__':

    import matplotlib

    matplotlib.use('TkAgg')
    import seaborn as sns
    import pandas as pd

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    np.random.seed(123)
    n = 20000

    cov = np.matrix([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
                    )

    #cov = np.eye(3)

    xs = sampling_from_gausiann_copula(cov, n)
    zs = np.zeros_like(xs)

    fig = plt.figure()
    #ax = fig.gca(projection='3d')

    f = g = h = stats.norm(0, 1)
#    g = stats.poisson(1)
 #   h = stats.bernoulli(0.5)

    zs[:, 0], zs[:, 1], zs[:, 2] = f.ppf(xs[:, 0]), g.ppf(xs[:, 1]), h.ppf(xs[:, 2])

    #plt.scatter(zs[:, 0], zs[:, 1], color='blue')
    #ts[:, 0], ts[:, 1], ts[:, 2] = f.ppf(xs[:, 0]), g.ppf(xs[:, 1]), h.ppf(xs[:, 2])

    ys = np.random.normal(size=(n, 2))
    #plt.scatter(ys[:, 0], ys[:, 1], color='red')

    #

    df = pd.DataFrame(zs)
    sns.distplot(ys[:, 0])
    sns.distplot(zs[:, 0])
    plt.show()
