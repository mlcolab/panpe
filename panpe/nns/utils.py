# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch import nn

_ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU,
    "lrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "selu": nn.SELU,
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
}


def activation_by_name(name, **kwargs):
    try:
        return _ACTIVATION_FUNCTIONS[name.lower()](**kwargs)
    except KeyError:
        raise KeyError(f"Unknown activation function {name}")
