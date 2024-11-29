# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import Tensor

from panpe.visualization.plot_data import plot_data


class MeasuredData:
    def __init__(self, q: Tensor, data: Tensor, sigmas: Tensor, phi: Tensor):
        self.q = torch.atleast_2d(q)
        self.data = torch.atleast_2d(data)
        self.sigmas = torch.atleast_2d(sigmas)
        self.phi = torch.atleast_2d(phi)

        assert self.q.shape == self.data.shape == self.sigmas.shape
        assert self.q.shape[0] == self.phi.shape[0] == 1
        assert self.q.ndim == self.phi.ndim == 2

    def to_dict(self) -> dict[str, Tensor]:
        return dict(q=self.q, data=self.data, sigmas=self.sigmas, phi=self.phi)

    def plot(self, **kwargs):
        plot_data(self, **kwargs)
