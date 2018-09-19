
import matplotlib.pyplot as plt
import numpy as np


def plot_grid_images_file(images, columns=1, titles=None, filename=None):

    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    columns: Number of columns in figure (number of rows is
            set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """

    assert ((titles is None) or (len(images) == len(titles)))

    number_images = len(images)

    if titles is None:
        titles = [f'Image ({i})' for i in range(1, number_images + 1)]

    images = np.squeeze(images)

    fig = plt.figure()
    rows = np.ceil(number_images / columns)

    for n, (image, title) in enumerate(zip(images, titles)):

        ax = fig.add_subplot(rows, columns, n + 1)

        if image.ndim == 2:
            plt.gray()

        ax.imshow(image)
        ax.set_title(title)

    fig.set_size_inches(np.array(fig.get_size_inches()) * number_images)

    if filename is None:
        plt.show()
    else:
        plt.savefig(f'{filename}.png', bbox_inches='tight')

    plt.clf()
    plt.close()