# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch.fft import fft, ifft
from torch import Tensor


def get_tau_estimation(y: Tensor):
    acf = batched_autocorr_func(y)
    taus = torch.cumsum(acf, dim=0) * 2 - 1
    window = auto_window_from_taus(taus)
    return taus[window]


def batched_autocorr_func(y: Tensor, window: int = None, reduce: bool = True):
    assert len(y.shape) == 2

    batch_size, sequence_size = y.shape
    n = next_pow_two(sequence_size)

    # Compute the FFT and then the auto-correlation function
    f = fft(y - y.mean(-1)[..., None], n=2 * n)
    acf = torch.real(ifft(f * torch.conj(f)))

    if window:
        acf = acf[:, :window]

    # Optionally normalize
    if reduce:
        acf = torch.mean(acf, 0)
        acf = acf / acf[0]
    else:
        acf = acf / acf[:, 0]

    return acf


def auto_window_from_taus(taus: Tensor, c: float = 5):
    num_taus = taus.shape[0]
    m = torch.arange(num_taus, device=taus.device, dtype=taus.dtype) < c * taus
    if torch.any(m):
        return torch.where(~m)[0].min()
    return num_taus - 1


def next_pow_two(n):
    """
    Find the next power of two for a given number.

    Args:
        n: int, the number to find the next power of two for

    Returns:
        int, the next power of two for the given number

    """
    i = 1
    while i < n:
        i = i << 1
    return i
