"""
    This file contains various pyTorch hacks to go around limitations
"""

import torch


def batch_eye_like(tensor):
    """
    Creates a sequence of identity tensors indicted by the batch size with the
    shape of the input tensor

    Parameters
    ----------
    tensor: Tensor (b, n, ..., n)
            b tensors with the same shape

    Returns
    -------
    Tensor (b, n, ..., n)
        b identity tensors
    """

    return tensor.new_ones(tensor.size(-1)).diag().expand_as(tensor)


def batch_inverse(matrices):
    """
    This function computes the inverse of a bunch of tensors using batch size as index.
    Once https://github.com/pytorch/pytorch/pull/9949 is merged this is useless

    Example:
        A, B are tensors
        batch_inverse([A,B]) = [inv(A), inv(B)]

    Parameters
    ----------
    matrices: Tensor (b, -1)
        b Tensors where the first component is interpreted as the batch size.

    Returns
    -------
    Tensor (b, -1)
        b Tensors representing the inverse of the matrix in b_mat
    """

    eye = batch_eye_like(matrices)
    inverses, _ = torch.gesv(eye, matrices)
    return inverses
