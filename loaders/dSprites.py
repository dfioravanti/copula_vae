import sys
import pathlib
import urllib

import numpy as np

import torch.utils.data as data_utils

from utils.HashTools import exists_and_correct_sha1


class dSpritesDataset(data_utils.Dataset):
    URL = 'https://github.com/deepmind/dsprites-dataset/blob/master/' \
          'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true'

    sha1 = '3e0a46a9238e425f46a6ef9e9bdfd7cedb35d50f'

    shape = (1, 64, 64)

    def __init__(self,
                 root_dir=None,
                 download=True,
                 verbose=False):

        super(dSpritesDataset, self).__init__()

        if root_dir is None:
            root_dir = pathlib.Path(sys.argv[0]).parents[0] / 'datasets'
        elif not isinstance(root_dir, pathlib.Path):
            root_dir = pathlib.Path(root_dir)

        self.dataset_root_path = root_dir / 'dSprites'
        self.dataset_path = self.dataset_root_path / 'dSprites.npz'

        if download:
            self.dataset_root_path.mkdir(parents=True, exist_ok=True)

            if verbose:
                print(f'dataset directory is {self.dataset_root_path}')

            if not exists_and_correct_sha1(self.dataset_path, dSpritesDataset.sha1):
                if verbose:
                    print('Downloading')

                urllib.request.urlretrieve(dSpritesDataset.URL, self.dataset_path)
            else:
                if verbose:
                    print('test exists and the hash matches, skip')

        self.data = np.load(self.dataset_path)['imgs']
