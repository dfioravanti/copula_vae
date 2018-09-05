
import os

import hashlib


def compute_hash(path):

    buf_size = 65536  # lets read stuff in 64kb chunks!

    sha1 = hashlib.sha1()

    with open(path, 'rb') as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break

            sha1.update(data)

    return sha1.hexdigest()


def exists_and_correct_hash(path, hash):

    return True if os.path.isfile(path) and compute_hash(path) == hash else False

