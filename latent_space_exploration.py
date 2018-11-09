
import pathlib

import torch
import numpy as np
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib import offsetbox

from models.copula_VAE import CopulaVAEWithNormals
from utils.copula_sampling import sampling_from_gausiann_copula
from loaders.load_funtions import load_MNIST


def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def explore_latent_space(model, s_dim, input_shape):

    path = pathlib.Path('outputs') / "Img"
    path.mkdir(parents=True, exist_ok=True)

    size = 10000

    _, test_loader, _ = load_MNIST(batch_size=size, shuffle=False)

    x = np.zeros((size, 1, 28, 28))
    y = np.zeros((size), dtype=np.int)

    for i, t in enumerate(test_loader):
        x[i*size:(i+1)*size] = t[0]
        y[i * size:(i + 1) * size] = t[1]

    x = x.reshape((size, -1))

    size_sampled = 1000
    idx = np.random.choice(size, size_sampled, replace=False)

    small_x = x[idx]
    small_y = y[idx]

    x_embedded = TSNE(n_components=2).fit_transform(small_x)

    plot_embedding(X=x_embedded, y=small_y)
    plt.savefig(path / f'mnist_tsne.pdf', format='pdf')

    x = torch.tensor(small_x).type(torch.FloatTensor)

    L = model.compute_L_x(x, size_sampled)
    s = sampling_from_gausiann_copula(L, size_sampled, s_dim)

    x_embedded = TSNE(n_components=2).fit_transform(s.detach().numpy())
    plot_embedding(X=x_embedded, y=small_y)
    plt.savefig(path / f's_tsne.pdf', format='pdf')

    z = model.p_z(s)

    x_embedded = TSNE(n_components=2).fit_transform(z.detach().numpy())
    plot_embedding(X=x_embedded, y=small_y)
    plt.savefig(path / f'z_tsne.pdf', format='pdf')

    exit(0)


if __name__ == '__main__':

    path = pathlib.Path('outputs') / 'mnist_stupid_decoder' / 'model'
    input_shape = (1, 28, 28)
    s_dim = 40

    model = CopulaVAEWithNormals(dimension_latent_space=s_dim, input_shape=input_shape)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    explore_latent_space(model, s_dim, input_shape)