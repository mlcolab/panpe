# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import Tensor, nn


class DataScaler(nn.Module):
    def scale(self, data: Tensor) -> Tensor:
        raise NotImplementedError

    def restore(self, data: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, data: Tensor) -> Tensor:
        return self.scale(data)


class IdentityScaler(DataScaler):
    def scale(self, data: Tensor) -> Tensor:
        return data

    def restore(self, data: Tensor) -> Tensor:
        return data


class ScalarLogAffineScaler(DataScaler):
    def __init__(self, weight: float = 0.1, bias: float = 0.5, eps: float = 1e-10):
        super().__init__()
        self.eps = eps

        self.register_buffer("weight", torch.tensor(weight))
        self.register_buffer("bias", torch.tensor(bias))

    def scale(self, data: Tensor) -> Tensor:
        return torch.log10(data + self.eps) * self.weight + self.bias

    def restore(self, data: Tensor) -> Tensor:
        return 10 ** ((data - self.bias) / self.weight) - self.eps


class AffineScaler(DataScaler):
    """
    Affine scaler from [min_bounds, max_bounds] range to scaled_range.

    Examples:
        >>> scaler = AffineScaler(1., 5., scaled_range=(0., 1.))
        >>> scaler.scale(torch.tensor([1., 2., 3.]))
        tensor([0.0000, 0.2500, 0.5000])
        >>> scaler.restore(torch.tensor([0., 0.25, 0.5]))
        tensor([1., 2., 3.])
        >>> scaler = AffineScaler(1., 5., scaled_range=(-1., 1.))
        >>> scaler.scale(torch.tensor([1., 2., 3.]))
        tensor([-1.0000, -0.5000,  0.0000])
        >>> scaler.restore(torch.tensor([-1.,  -0.5,  0.]))
        tensor([1., 2., 3.])

    """

    def __init__(
        self,
        min_bounds: Tensor or float,
        max_bounds: Tensor or float,
        scaled_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        min_bounds = torch.as_tensor(min_bounds)
        max_bounds = torch.as_tensor(max_bounds)
        scaled_range = torch.as_tensor(scaled_range)

        self.register_buffer("min_bounds", min_bounds)
        self.register_buffer("max_bounds", max_bounds)
        self.register_buffer(
            "delta", (max_bounds - min_bounds) / (scaled_range[1] - scaled_range[0])
        )
        self.register_buffer("scaled_min", scaled_range[0])

    def scale(self, data: Tensor) -> Tensor:
        return (data - self.min_bounds) / self.delta + self.scaled_min

    def restore(self, data: Tensor) -> Tensor:
        return (data - self.scaled_min) * self.delta + self.min_bounds


class ScalerDict(DataScaler):
    def __init__(self, **scalers):
        super().__init__()
        self.scalers = nn.ModuleDict(scalers)

    def scale(self, data: dict) -> dict:
        return {
            k: self.scalers[k](v) if k in self.scalers else v for k, v in data.items()
        }

    def restore(self, data: dict) -> dict:
        return {
            k: self.scalers[k].restore(v) if k in self.scalers else v
            for k, v in data.items()
        }


if __name__ == "__main__":
    import doctest

    doctest.testmod()
