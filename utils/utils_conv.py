from math import floor

import torch
from torch import nn

def compute_conv_output_shape(h_in, w_in, kernel_size=1, stride=1, padding=0, dilation=1):

    """
    Utility function for computing output shape of convolutional layes
    takes h and w and returns the new h and w
    """

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(padding) is not tuple:
        padding = (padding, padding)

    if type(dilation) is not tuple:
        dilation = (dilation, dilation)

    h_out = floor(((h_in + (2 * padding[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1) / stride[0]) + 1)
    w_out = floor(((w_in + (2 * padding[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1) / stride[1]) + 1)

    return h_out, w_out


def compute_maxpooling_output_shape(h_in, w_in, kernel_size=1, stride=None, padding=0, dilation=1):

    if stride is None:
        stride = kernel_size

    return compute_conv_output_shape(h_in, w_in, kernel_size, stride, padding, dilation)


def compute_final_convolution_shape(h_in, w_in, number_blocks, maxpolling=False,
                                    kernel_size=1, stride=1, padding=0, dilation=1):

    h_out, w_out = h_in, w_in

    for _ in range(number_blocks):
        h_out, w_out = compute_conv_output_shape(h_out, w_out, kernel_size=kernel_size,
                                                 stride=stride, padding=padding, dilation=dilation)

        if maxpolling:
            h_out, w_out = compute_maxpooling_output_shape(h_out, w_out, kernel_size=2)

    return h_out, w_out


def compute_deconv_output_shape(h_in, w_in, kernel_size=1, stride=1, padding=0, output_padding=0):

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(padding) is not tuple:
        padding = (padding, padding)

    if type(output_padding) is not tuple:
        output_padding = (output_padding, output_padding)

    h_out = (h_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0] + output_padding[0]
    w_out = (w_in - 1) * stride[1] - 2 * padding[1] + kernel_size[1] + output_padding[1]

    return h_out, w_out


def compute_final_deconv_shape(h_in, w_in, number_blocks,
                               kernel_size=1, stride=1, padding=0, output_padding=0):

    h_out, w_out = h_in, w_in

    for _ in range(number_blocks):
        h_out, w_out = compute_deconv_output_shape(h_out, w_out, kernel_size=kernel_size,
                                                   stride=stride, padding=padding, output_padding=output_padding)

    return h_out, w_out


def build_convolutional_blocks(number_blocks, nb_channels_in, nb_channels, maxpooling=False,
                               kernel_size=1, stride=1, padding=0, dilation=1):

    final_block = nn.Sequential()
    final_block.add_module('conv_0', nn.Conv2d(nb_channels_in, nb_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation))
    final_block.add_module('relu_0', nn.ReLU())
    if maxpooling:
        final_block.add_module('max_pooling', nn.MaxPool2d(kernel_size=2))
    final_block.add_module('dropout_0', nn.Dropout())

    for i in range(1, number_blocks):

        final_block.add_module(f'conv_{i}', nn.Conv2d(nb_channels, nb_channels,
                                                      kernel_size=kernel_size, stride=stride, padding=padding,
                                                      dilation=dilation))
        final_block.add_module(f'relu_{i}', nn.ReLU())
        if maxpooling:
            final_block.add_module(f'max_pooling_{i}', nn.MaxPool2d(kernel_size=2))
        final_block.add_module(f'dropout_{i}', nn.Dropout())

    return final_block


def build_deconvolutional_blocks(number_blocks, nb_channels_in, nb_channels,
                                 kernel_size=1, stride=1, padding=0, output_padding=0):

    final_block = nn.Sequential()
    final_block.add_module('deconv_0', nn.ConvTranspose2d(nb_channels_in, nb_channels,
                                                          kernel_size=kernel_size, stride=stride, padding=padding,
                                                          output_padding=output_padding))
    final_block.add_module('relu_0', nn.ReLU())
    final_block.add_module('dropout_0', nn.Dropout())

    for i in range(1, number_blocks):

        final_block.add_module(f'deconv_{i}', nn.ConvTranspose2d(nb_channels, nb_channels,
                                                                 kernel_size=kernel_size, stride=stride,
                                                                 padding=padding, output_padding=output_padding))
        final_block.add_module(f'relu_{i}', nn.ReLU())
        final_block.add_module(f'dropout_{i}', nn.Dropout())

    return final_block