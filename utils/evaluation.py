from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt


def plot_samples(model, output_dir=None, writer=None, n_samples=100, n_rows=10):
    shape = (-1,) + model.input_shape

    z = model.p_z(n=n_samples)
    x, _ = model.p_x(z=z)
    x = x.detach().reshape(shape)
    grid = make_grid(x, nrow=n_rows)

    if writer is not None:
        writer.add_image(f'sample_prior', grid, 0)

    if output_dir is None:

        plt.figure(figsize=(10, 10))
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.show()

    else:

        img_dir = output_dir / 'samples'
        img_dir.mkdir(parents=True, exist_ok=True)
        save_image(tensor=x,
                   filename=img_dir / 'grid.png',
                   nrow=n_rows)
        # TODO: Think if I need to save single images


if __name__ == '__main__':
    import pathlib
    import torch
    from models.CopulaVAE import CopulaVAE
    from models.VAE import VAE

    path = pathlib.Path('../outputs') / 'trained' / 'fashionmnist_bin_shallow_copula_gaussian_50' / 'best.tar'
    input_shape = (1, 28, 28)
    z_dim = 50

    checkpoint = torch.load(path, map_location='cpu')

    model = CopulaVAE(dimension_latent_space=z_dim, input_shape=input_shape, dataset_type="binary")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # check_plots(model)
    plot_samples(model, pathlib.Path('../'))
