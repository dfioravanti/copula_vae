from loaders import BinaryMNISTDataset
import torch.utils.data as data_utils


def load_binary_MNIST(root_dir=None, batch_size=20, shuffle=True, transform=None, download=True):

    train_dataset = BinaryMNISTDataset.BinaryMNISTDataset(root_dir,
                                                          download=download,
                                                          train=True,
                                                          transform=transform)
    train_loader = data_utils.DataLoader(train_dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle)

    test_dataset = BinaryMNISTDataset.BinaryMNISTDataset(root_dir,
                                                         download=download,
                                                         train=False,
                                                         test=True,
                                                         transform=transform)
    test_loader = data_utils.DataLoader(test_dataset,
                                        batch_size=batch_size,
                                        shuffle=False)

    valid_dataset = BinaryMNISTDataset.BinaryMNISTDataset(root_dir,
                                                          download=download,
                                                          train=False,
                                                          validation=True,
                                                          transform=transform)
    valid_loader = data_utils.DataLoader(valid_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)

    return train_loader, test_loader, valid_loader


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    train_loader, _, _, = load_binary_MNIST('../datasets/', batch_size=1, download=False)

    sample = next(iter(train_loader))
    print(sample.shape)
    sample = sample.reshape((28, 28))
    plt.imshow(sample, cmap='gray')
    plt.show()

