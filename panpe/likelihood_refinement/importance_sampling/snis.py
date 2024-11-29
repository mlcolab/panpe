# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import Tensor

from panpe.likelihood_refinement.importance_sampling.streaming_snis import StreamingSNIS


class SNISBackend:
    """
    A base class for storing parameters and log probabilities for self-normalized importance sampling.
    """

    @property
    def params(self) -> Tensor:
        raise NotImplementedError

    @property
    def log_q(self) -> Tensor:
        raise NotImplementedError

    @property
    def log_p(self) -> Tensor:
        raise NotImplementedError

    @property
    def log_weights(self) -> Tensor:
        return self.log_p - self.log_q

    def update(self, log_p: Tensor, log_q: Tensor, params: Tensor) -> None:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def size(self) -> int:
        return self.log_p.shape[0]


class SNISInMemoryBackend(SNISBackend):
    """
    A class for storing parameters and log probabilities for self-normalized importance sampling in memory.
    """

    def __init__(self, device: str = "cpu"):
        self._params = []
        self._log_q = []
        self._log_p = []
        self._device = device

    def update(self, log_p: Tensor, log_q: Tensor, params: Tensor) -> None:
        self._params.append(params.to(self._device))
        self._log_q.append(log_q.to(self._device))
        self._log_p.append(log_p.to(self._device))

    @property
    def params(self) -> Tensor:
        return torch.cat(self._params, dim=0)

    @property
    def log_p(self) -> Tensor:
        return torch.cat(self._log_p, dim=0)

    @property
    def log_q(self) -> Tensor:
        return torch.cat(self._log_q, dim=0)

    def reset(self):
        self._params = []
        self._log_q = []
        self._log_p = []

    @property
    def size(self) -> int:
        return sum([p.shape[0] for p in self._params])


class SNISEmptyBackend(SNISBackend):
    """
    Does not store anything (for streaming SNIS).
    """

    def update(self, log_p: Tensor, log_q: Tensor, params: Tensor) -> None:
        pass

    @property
    def params(self) -> Tensor:
        return torch.empty((0,))

    @property
    def log_p(self) -> Tensor:
        return torch.empty((0,))

    @property
    def log_q(self) -> Tensor:
        return torch.empty((0,))

    def reset(self):
        pass


class ImportanceSampling:
    def __init__(self, backend: SNISBackend = None):
        self.backend = backend or SNISInMemoryBackend()
        self.streaming_snis = StreamingSNIS()

    @property
    def size(self) -> int:
        return self.backend.size

    def update(self, log_p: Tensor, log_q: Tensor, params: Tensor) -> None:
        self.streaming_snis.update(log_p, log_q, params)
        finite_indices = torch.isfinite(log_p)
        if finite_indices.any():
            self.backend.update(
                log_p[finite_indices], log_q[finite_indices], params[finite_indices]
            )

    def sample(self, num_samples: int) -> tuple[Tensor, Tensor]:
        log_weights = self.backend.log_weights
        weights = (log_weights - torch.logsumexp(log_weights, dim=0)).exp()
        indices = torch.multinomial(weights, num_samples, replacement=True)
        return self.backend.params[indices], self.backend.log_weights[indices]

    def reset(self):
        self.streaming_snis = StreamingSNIS()
        self.backend.reset()

    @property
    def neff(self) -> int:
        return self.streaming_snis.neff

    @property
    def eff(self) -> float:
        return self.streaming_snis.eff

    @property
    def acceptance_ratio(self) -> float:
        return self.streaming_snis.acceptance_ratio

    @property
    def log_evidence(self):
        return self.streaming_snis.log_evidence

    @property
    def params(self) -> Tensor:
        return self.backend.params

    @property
    def log_p(self) -> Tensor:
        return self.backend.log_p

    @property
    def log_q(self) -> Tensor:
        return self.backend.log_q

    @property
    def log_weights(self) -> Tensor:
        return self.backend.log_weights
