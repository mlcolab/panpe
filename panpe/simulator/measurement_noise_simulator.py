# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from math import log10, sqrt

import torch
from torch import Tensor, nn


class MeasurementNoiseSimulator(nn.Module):
    """
    A base class for measurement noise simulators.
    """

    def sample_sigmas(self, data: Tensor) -> Tensor:
        raise NotImplementedError

    def sample(self, data: Tensor, sigmas: Tensor) -> Tensor:
        raise NotImplementedError

    def log_prob(self, data: Tensor, sigmas: Tensor, sim_curves: Tensor) -> Tensor:
        raise NotImplementedError


class NormalNoiseSimulator(MeasurementNoiseSimulator):
    """
    A measurement noise simulator that assumes normal distribution of the noise.
    """

    def __init__(
        self,
        rel_sigma_range: tuple[float, float] = (0.1, 0.3),
        logdist: bool = False,
    ):
        super().__init__()
        self.rel_sigma_range = rel_sigma_range
        self.logdist = logdist

    def sample_sigmas(self, data: Tensor) -> Tensor:
        return _sample_sigmas(
            data, self.rel_sigma_range[0], self.rel_sigma_range[1], self.logdist
        )

    def sample(self, data: Tensor, sigmas: Tensor) -> Tensor:
        return torch.normal(mean=data, std=sigmas).clamp_min_(0.0)

    def log_prob(self, data: Tensor, sigmas: Tensor, sim_curves: Tensor) -> Tensor:
        log_probs = (
            -(((data - sim_curves) / sigmas) ** 2) / 2
            - torch.log(sigmas)
            - torch.log(torch.tensor(2 * torch.pi)) / 2
        )

        log_zero_cdf = torch.log(
            1 - 0.5 * (1 + torch.erf(-data / (sigmas * sqrt(2.0))))
        )

        return (log_probs - log_zero_cdf).sum(-1)


def _sample_sigmas(
    data: Tensor, min_sigma: float, max_sigma: float, logdist: bool
) -> Tensor:
    """
    Samples sigmas from a uniform or log-uniform distribution.
    """
    if not logdist:
        rel_sigmas = torch.rand_like(data) * (max_sigma - min_sigma) + min_sigma
    else:
        rel_sigmas_log = torch.rand_like(data) * (
            log10(max_sigma) - log10(min_sigma)
        ) + log10(min_sigma)
        rel_sigmas = 10**rel_sigmas_log

    sigmas = rel_sigmas * data

    return sigmas
