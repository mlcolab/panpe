# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random

import numpy as np

import torch
from torch import Tensor, tensor

try:
    from IPython import get_ipython

    if (
        "IPKernelApp" not in get_ipython().config
    ):  # Check if IPython Kernel is being used
        raise ImportError("Not in a notebook")
    from tqdm.notebook import tqdm, trange  # Use tqdm.notebook if in a notebook
except (ImportError, AttributeError):
    from tqdm import tqdm, trange  # Fallback to standard tqdm if not in a notebook


__all__ = [
    "to_np",
    "to_t",
    "set_seed",
    "tqdm",
    "trange",
]


def set_seed(seed):
    """
    Set seed globally for all possible random generators and set unique seeds for each CUDA device.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Check if CUDA is available and set a unique seed for each device
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            unique_seed = seed + device_index  # Create a unique seed for each device
            with torch.cuda.device(device_index):
                torch.cuda.manual_seed(unique_seed)

        # Ensure reproducibility on CUDA (might reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_np(arr):
    """
    Converts a torch tensor to a numpy array. If the input is already a numpy array, it is returned unchanged.
    """
    if isinstance(arr, Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def to_t(arr, device=None, dtype=None):
    """
    Converts a numpy array to a torch tensor. If the input is already a tensor, it is returned unchanged.
    """
    if not isinstance(arr, Tensor):
        return tensor(arr, device=device, dtype=dtype)
    return arr
