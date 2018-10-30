
import matplotlib.pyplot as plt
import numpy as np


def plot_grid_images_file(xs, recontructed, columns=1, filename=None):

    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    recontructed: List of np.arrays compatible with plt.imshow.

    columns: Number of columns in figure (number of rows is
            set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """

    number_images = len(recontructed)
    xs = np.squeeze(xs)
    recontructed = np.squeeze(recontructed)

    fig = plt.figure()

    for n, (x, rec) in enumerate(zip(xs, recontructed)):

        ax1 = fig.add_subplot(2, columns, n + 1, frameon=False)
        ax2 = fig.add_subplot(2, columns, n + 1 + number_images, frameon=False)

        if x.ndim == 2:
            plt.gray()
        else:
            x = np.moveaxis(x, 0, -1)
            rec = np.moveaxis(rec, 0, -1)

        ax1.imshow(x)
        ax2.imshow(rec)

        for x in [ax1, ax2]:
            x.yaxis.set_ticks([])
            x.xaxis.set_ticks([])

    fig.set_size_inches(np.array(fig.get_size_inches()) * number_images)

    if filename is None:
        plt.show()
    else:
        plt.savefig(f'{filename}.png', bbox_inches='tight')

    plt.clf()
    plt.close()