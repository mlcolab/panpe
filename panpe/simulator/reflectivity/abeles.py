# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from functools import reduce

import torch
from torch import Tensor


def abeles(
    q: Tensor,
    thickness: Tensor,
    roughness: Tensor,
    sld: Tensor,
):
    c_dtype = torch.complex128 if q.dtype is torch.float64 else torch.complex64

    batch_size, num_layers = thickness.shape

    if sld.shape[-1] == num_layers + 1:
        # add zero ambient sld
        sld = torch.cat([torch.zeros(batch_size, 1).to(sld), sld], -1)
    if sld.shape[-1] != num_layers + 2:
        raise ValueError(
            "Number of SLD values does not equal to num_layers + 2 (substrate + ambient)."
        )

    sld = sld[:, None]

    # add zero thickness for ambient layer:
    thickness = torch.cat([torch.zeros(batch_size, 1).to(thickness), thickness], -1)[
        :, None
    ]

    roughness = roughness[:, None] ** 2

    sld = (sld - sld[..., :1]) * 1e-6 + 1e-36j

    k_z0 = (q / 2).to(c_dtype)

    if k_z0.dim() == 1:
        k_z0.unsqueeze_(0)

    if k_z0.dim() == 2:
        k_z0.unsqueeze_(-1)

    k_n = torch.sqrt(k_z0**2 - 4 * math.pi * sld)

    # k_n.shape - (batch, q, layers)

    k_n, k_np1 = k_n[..., :-1], k_n[..., 1:]

    beta = 1j * thickness * k_n

    exp_beta = torch.exp(beta)
    exp_m_beta = torch.exp(-beta)

    rn = (k_n - k_np1) / (k_n + k_np1) * torch.exp(-2 * k_n * k_np1 * roughness)

    c_matrices = torch.stack(
        [
            torch.stack([exp_beta, rn * exp_m_beta], -1),
            torch.stack([rn * exp_beta, exp_m_beta], -1),
        ],
        -1,
    )

    # maybe faster to swap axes and provide a single tensor to reduce
    c_matrices = [c.squeeze(-3) for c in c_matrices.split(1, -3)]

    m = reduce(torch.matmul, c_matrices)

    r = (m[..., 1, 0] / m[..., 0, 0]).abs() ** 2
    r = torch.clamp_max_(r, 1.0)

    return r


# @torch.jit.script # commented so far due to complex numbers issue
def abeles_compiled(
    q: Tensor,
    thickness: Tensor,
    roughness: Tensor,
    sld: Tensor,
):
    return abeles(q, thickness, roughness, sld)
