import math
import os
from typing import Tuple, Union, List, Sequence
from textwrap import wrap
from torchvision import transforms

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.utils import _pair

from bindsnet.datasets import *


def msg_wrapper(msg: List[str], style: int) -> None:
    """
    Wrap the message with a border.

    :param msg: List of messages.
    :param style: Pick the style of the border.
    """
    width = max([len(sentence) for sentence in msg])

    if style == 1:
        print('\n' + '#' * width)
        for sentence in msg:
            for line in wrap(sentence, width):
                print('{0:^{1}}'.format(line, width))
        print('#' * width + '\n')
    elif style == 2:
        print('\n+-' + '-' * width + '-+')
        for sentence in msg:
            for line in wrap(sentence, width):
                print('| {0:^{1}} |'.format(line, width))
        print('+-' + '-'*(width) + '-+' + '\n')


def load_data(
    dataset_name: str,
    encoder: torch.Tensor = None,
    train: bool = True,
    intensity: float = 128,
) -> torch.utils.data.Dataset:
    """
    Load dataset of choice.

    :param dataset_name: Name of dataset.
    :param encoder: Spike encoder for generating spike trains.
    :param train: True for train data, False for test data.
    :param intensity: Intensity for transformation of data.
    :return: Return dataset.
    """
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(ROOT_PATH, "..", "data", dataset_name)

    try:

        if dataset_name == "MNIST":
            # Load MNIST data.
            dataset = MNIST(
                encoder,
                None,
                root=DATA_PATH,
                train=train,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
                ),
            )
        elif dataset_name == "FashionMNIST":
            # Load FashionMNIST data.
            dataset = FashionMNIST(
                encoder,
                None,
                root=DATA_PATH,
                train=train,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
                ),
            )

        return dataset

    except:
        raise NameError("Name \"%s\" is not defined" % dataset_name)
        #raise NameError("name \"{}\" is not defined".format(dataset_name))


def make_dirs(path: str) -> None:
    """
    Setup directories within path.

    :param path: Name of path.
    """
    os.makedirs(path, exist_ok=True)
    # Alternative way:
    # if not os.path.isdir(path):
    #     os.makedirs(path)


def get_network_const(n_neurons: int, default_value: List[float]) -> float:
    """
    Get time constant of threshold potential decay & decay factor
    for different sized network.

    :param n_neurons: Number of excitatory, inhibitory neurons.
    :param default_value: Array of default value for
        constant theta_plus, tc_theta_decay.
    :return: Return constant theta_plus, tc_theta_decay.
    """
    # Num. of neurons : (theta plus, theta decay time constant)
    const_choices = {
        100  : (0.07, 6e6),
        400  : (0.07, 6e6),
        1600 : (0.07, 8e6),
        6400 : (0.05, 2e7),
        10000: (0.05, 2e7),
    }
    const = const_choices.get(n_neurons, default_value)

    return const[0], const[1]


def get_lrate(
    n_neurons: int,
    default_lrates: Tuple[Union[float, Sequence[float]]],
) -> float:
    """
    Get learning rate for different sized network.

    :param n_neurons: Number of excitatory, inhibitory neurons.
    :param default_lrates: Default value of learning rates.
    :return: Return pre- and post-synaptic learning rates for each layer.
    """
    # Num. of neurons : (nu_exc, nu_sl)
    lrate_choices = {
        100  : (5e-3, 4e-2),
        400  : (5e-3, 4e-2),
        1600 : (2e-2, 8e-2),
        6400 : (1e-2, 4e-2),
        10000: (7.5e-3, 4e-2),
    }
    lrate = lrate_choices.get(n_neurons, default_lrates)

    # lrate[0]: learning rates for exc layer, lrate[1]: learning rates for SL layer.
    return lrate[0], lrate[1]


def sample_from_class(
    dataset: torch.utils.data.Dataset,
    n_samples: int,
) -> torch.utils.data.Dataset:
    """
    Stratified sampling. Create a dataset with ``n_samples`` numbers of each class
    from the original dataset.

    :param dataset: Original dataset.
    :param n_samples: Number of samples to use from each class.
    """
    class_counts = {}
    new_data = []
    new_labels = []

    for index, data in enumerate(dataset.data):
        label = dataset.targets[index]
        c = label.item()
        class_counts[c] = class_counts.get(c, 0) + 1
        if class_counts[c] <= n_samples:
            new_data.append(torch.unsqueeze(data, 0))
            new_labels.append(torch.unsqueeze(label, 0))

    dataset.data = torch.cat(new_data)
    dataset.targets = torch.cat(new_labels)

    return dataset


def im2col_indices(
    x: Tensor,
    kernel_height: int,
    kernel_width: int,
    padding: Tuple[int, int] = (0, 0),
    stride: Tuple[int, int] = (1, 1),
) -> Tensor:
    # language=rst
    """
    im2col is a special case of unfold which is implemented inside of Pytorch.

    :param x: Input image tensor to be reshaped to column-wise format.
    :param kernel_height: Height of the convolutional kernel in pixels.
    :param kernel_width: Width of the convolutional kernel in pixels.
    :param padding: Amount of zero padding on the input image.
    :param stride: Amount to stride over image by per convolution.
    :return: Input tensor reshaped to column-wise format.
    """
    return F.unfold(x, (kernel_height, kernel_width), padding=padding, stride=stride)


def col2im_indices(
    cols: Tensor,
    x_shape: Tuple[int, int, int, int],
    kernel_height: int,
    kernel_width: int,
    padding: Tuple[int, int] = (0, 0),
    stride: Tuple[int, int] = (1, 1),
) -> Tensor:
    # language=rst
    """
    col2im is a special case of fold which is implemented inside of Pytorch.

    :param cols: Image tensor in column-wise format.
    :param x_shape: Shape of original image tensor.
    :param kernel_height: Height of the convolutional kernel in pixels.
    :param kernel_width: Width of the convolutional kernel in pixels.
    :param padding: Amount of zero padding on the input image.
    :param stride: Amount to stride over image by per convolution.
    :return: Image tensor in original image shape.
    """
    return F.fold(
        cols, x_shape, (kernel_height, kernel_width), padding=padding, stride=stride
    )


def get_square_weights(
    weights: Tensor, n_sqrt: int, side: Union[int, Tuple[int, int]]
) -> Tensor:
    # language=rst
    """
    Return a grid of a number of filters ``sqrt ** 2`` with side lengths ``side``.

    :param weights: Two-dimensional tensor of weights for two-dimensional data.
    :param n_sqrt: Square root of no. of filters.
    :param side: Side length(s) of filter.
    :return: Reshaped weights to square matrix of filters.
    """
    if isinstance(side, int):
        side = (side, side)

    square_weights = torch.zeros(side[0] * n_sqrt, side[1] * n_sqrt)
    for i in range(n_sqrt):
        for j in range(n_sqrt):
            n = i * n_sqrt + j

            if not n < weights.size(1):
                break

            x = i * side[0]
            y = (j % n_sqrt) * side[1]
            filter_ = weights[:, n].contiguous().view(*side)
            square_weights[x : x + side[0], y : y + side[1]] = filter_

    return square_weights


def get_square_assignments(assignments: Tensor, n_sqrt: int) -> Tensor:
    # language=rst
    """
    Return a grid of assignments.

    :param assignments: Vector of integers corresponding to class labels.
    :param n_sqrt: Square root of no. of assignments.
    :return: Reshaped square matrix of assignments.
    """
    square_assignments = torch.mul(torch.ones(n_sqrt, n_sqrt), -1.0)
    for i in range(n_sqrt):
        for j in range(n_sqrt):
            n = i * n_sqrt + j

            if not n < assignments.size(0):
                break

            square_assignments[
                i : (i + 1), (j % n_sqrt) : ((j % n_sqrt) + 1)
            ] = assignments[n]

    return square_assignments


def reshape_locally_connected_weights(
    w: Tensor,
    n_filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    conv_size: Union[int, Tuple[int, int]],
    locations: Tensor,
    input_sqrt: Union[int, Tuple[int, int]],
) -> Tensor:
    # language=rst
    """
    Get the weights from a locally connected layer and reshape them to be two-dimensional and square.

    :param w: Weights from a locally connected layer.
    :param n_filters: No. of neuron filters.
    :param kernel_size: Side length(s) of convolutional kernel.
    :param conv_size: Side length(s) of convolution population.
    :param locations: Binary mask indicating receptive fields of convolution population neurons.
    :param input_sqrt: Sides length(s) of input neurons.
    :return: Locally connected weights reshaped as a collection of spatially ordered square grids.
    """
    kernel_size = _pair(kernel_size)
    conv_size = _pair(conv_size)
    input_sqrt = _pair(input_sqrt)

    k1, k2 = kernel_size
    c1, c2 = conv_size
    i1, i2 = input_sqrt
    c1sqrt, c2sqrt = int(math.ceil(math.sqrt(c1))), int(math.ceil(math.sqrt(c2)))
    fs = int(math.ceil(math.sqrt(n_filters)))

    w_ = torch.zeros((n_filters * k1, k2 * c1 * c2))

    for n1 in range(c1):
        for n2 in range(c2):
            for feature in range(n_filters):
                n = n1 * c2 + n2
                filter_ = w[
                    locations[:, n],
                    feature * (c1 * c2) + (n // c2sqrt) * c2sqrt + (n % c2sqrt),
                ].view(k1, k2)
                w_[feature * k1 : (feature + 1) * k1, n * k2 : (n + 1) * k2] = filter_

    if c1 == 1 and c2 == 1:
        square = torch.zeros((i1 * fs, i2 * fs))

        for n in range(n_filters):
            square[
                (n // fs) * i1 : ((n // fs) + 1) * i2,
                (n % fs) * i2 : ((n % fs) + 1) * i2,
            ] = w_[n * i1 : (n + 1) * i2]

        return square
    else:
        square = torch.zeros((k1 * fs * c1, k2 * fs * c2))

        for n1 in range(c1):
            for n2 in range(c2):
                for f1 in range(fs):
                    for f2 in range(fs):
                        if f1 * fs + f2 < n_filters:
                            square[
                                k1 * (n1 * fs + f1) : k1 * (n1 * fs + f1 + 1),
                                k2 * (n2 * fs + f2) : k2 * (n2 * fs + f2 + 1),
                            ] = w_[
                                (f1 * fs + f2) * k1 : (f1 * fs + f2 + 1) * k1,
                                (n1 * c2 + n2) * k2 : (n1 * c2 + n2 + 1) * k2,
                            ]

        return square


def reshape_conv2d_weights(weights: torch.Tensor) -> torch.Tensor:
    # language=rst
    """
    Flattens a connection weight matrix of a Conv2dConnection

    :param weights: Weight matrix of Conv2dConnection object.
    :param wmin: Minimum allowed weight value.
    :param wmax: Maximum allowed weight value.
    """
    sqrt1 = int(np.ceil(np.sqrt(weights.size(0))))
    sqrt2 = int(np.ceil(np.sqrt(weights.size(1))))
    height, width = weights.size(2), weights.size(3)
    reshaped = torch.zeros(
        sqrt1 * sqrt2 * weights.size(2), sqrt1 * sqrt2 * weights.size(3)
    )

    for i in range(sqrt1):
        for j in range(sqrt1):
            for k in range(sqrt2):
                for l in range(sqrt2):
                    if i * sqrt1 + j < weights.size(0) and k * sqrt2 + l < weights.size(
                        1
                    ):
                        fltr = weights[i * sqrt1 + j, k * sqrt2 + l].view(height, width)
                        reshaped[
                            i * height
                            + k * height * sqrt1 : (i + 1) * height
                            + k * height * sqrt1,
                            (j % sqrt1) * width
                            + (l % sqrt2) * width * sqrt1 : ((j % sqrt1) + 1) * width
                            + (l % sqrt2) * width * sqrt1,
                        ] = fltr

    return reshaped


def reshape_local_connection_2d_weights(
    w: torch.Tensor,
    n_filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    conv_size: Union[int, Tuple[int, int]],
    input_sqrt: Union[int, Tuple[int, int]],
) -> torch.Tensor:
    # language=rst
    """
    Reshape a slice of weights of a LocalConnection2D slice for plotting.
    :param w: Slice of weights from a LocalConnection2D object.
    :param n_filters: Number of filters (output channels).
    :param kernel_size: Side length(s) of convolutional kernel.
    :param conv_size: Side length(s) of convolution population.
    :param input_sqrt: Sides length(s) of input neurons.
    :return: A slice of LocalConnection2D weights reshaped as a collection of spatially ordered square grids.
    """

    k1, k2 = kernel_size
    c1, c2 = conv_size
    i1, i2 = input_sqrt

    fs = int(np.ceil(np.sqrt(n_filters)))

    w_ = torch.zeros((n_filters * k1, k2 * c1 * c2))

    for n1 in range(c1):
        for n2 in range(c2):
            for feature in range(n_filters):
                n = n1 * c2 + n2
                filter_ = w[feature, n1, n2, :, :].view(k1, k2)
                w_[feature * k1 : (feature + 1) * k1, n * k2 : (n + 1) * k2] = filter_

    if c1 == 1 and c2 == 1:
        square = torch.zeros((i1 * fs, i2 * fs))

        for n in range(n_filters):
            square[
                (n // fs) * i1 : ((n // fs) + 1) * i2,
                (n % fs) * i2 : ((n % fs) + 1) * i2,
            ] = w_[n * i1 : (n + 1) * i2]

        return square
    else:
        square = torch.zeros((k1 * fs * c1, k2 * fs * c2))

        for n1 in range(c1):
            for n2 in range(c2):
                for f1 in range(fs):
                    for f2 in range(fs):
                        if f1 * fs + f2 < n_filters:
                            square[
                                k1 * (n1 * fs + f1) : k1 * (n1 * fs + f1 + 1),
                                k2 * (n2 * fs + f2) : k2 * (n2 * fs + f2 + 1),
                            ] = w_[
                                (f1 * fs + f2) * k1 : (f1 * fs + f2 + 1) * k1,
                                (n1 * c2 + n2) * k2 : (n1 * c2 + n2 + 1) * k2,
                            ]

        return square
