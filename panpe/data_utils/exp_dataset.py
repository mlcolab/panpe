# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import h5py

from panpe.inference.measured_data import MeasuredData


class ExpDataset:
    def __init__(self, h5path: str, device: str = "cuda"):
        self.h5path = h5path
        self.device = device

        with h5py.File(self.h5path, "r") as f:
            self.keys = sorted(list(f.keys()), key=int)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = self.keys[key]
        with h5py.File(self.h5path, "r") as f:
            return self._group2data(f[key])

    def __len__(self):
        return len(self.keys)

    def _group2data(self, group):
        return MeasuredData(
            q=_g2t(group, "q", device=self.device),
            data=_g2t(group, "data", device=self.device),
            sigmas=_g2t(group, "sigmas", device=self.device),
            phi=_g2t(group, "phi", device=self.device),
        )


def _g2t(g, k, device="cuda"):
    return torch.from_numpy(g[k][()]).float().to(device)
