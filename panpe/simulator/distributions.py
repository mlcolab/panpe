# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import Tensor

from nflows.distributions import Distribution


class BoxUniform(Distribution):
    # BoxUniform distribution from nflows is not a subclass of nn.Module, so we have to redefine it here.
    # It is less general than the original BoxUniform, as it assumes that the last dimension of samples
    # is the parameter dimension.
    """
    Box uniform distribution.
    """

    def __init__(self, low: list or Tensor, high: list or Tensor):
        super().__init__()
        self.register_buffer("low", torch.as_tensor(low))
        self.register_buffer("high", torch.as_tensor(high))
        self.register_buffer("_log_p", -torch.log(self.high - self.low).sum(-1))

    def _log_prob(self, value: Tensor, context=None):
        log_prob = torch.empty_like(value[..., 0]).fill_(self._log_p)
        log_prob[torch.any((value < self.low) | (value > self.high), -1)] = -float(
            "inf"
        )
        return log_prob

    def _sample(self, num_samples: int, context: Tensor = None):

        if context is not None:
            device = context.device
            dtype = context.dtype
        else:
            device = self.low.device
            dtype = self.low.dtype

        samples = (
            torch.rand(num_samples, self.low.shape[-1], device=device, dtype=dtype)
            * (self.high - self.low)
            + self.low
        )

        return samples


class TruncatedExponential(Distribution):
    """
    Truncated exponential distribution with support [0, 1]:
    p(x) = lam * exp(-lam * x) / (1 - exp(-lam)).

    It assumes that the parameter dimension is the last one, and lam parameter is shared among them.
    """

    def __init__(self, ndim: int, lam: float or Tensor = 0.05):
        super().__init__()
        self.register_buffer("ndim", torch.as_tensor(ndim))
        self.register_buffer("lam", torch.as_tensor(lam))

    def _sample(self, num_samples: int, context=None):
        if context is not None:
            device = context.device
            dtype = context.dtype
        else:
            device = self.lam.device
            dtype = self.lam.dtype

        return truncated_exponential_sampler(
            self.lam, num_samples, self.ndim.item(), device=device, dtype=dtype
        )

    def _log_prob(self, inputs: Tensor, context=None):
        log_prob = (
            torch.log(self.lam)
            - self.lam * inputs
            - torch.log(1 - torch.exp(-self.lam))
        )
        return log_prob.sum(-1)


class DefaultWidthSampler(Distribution):
    """
    Default width sampler for the hyperprior. Samples a mixture of truncated exponential and uniform distributions:
    p(x) = weight * exp(-lam * x) / (1 - exp(-lam)) + (1 - weight).
    """

    def __init__(self, ndim: int):
        super().__init__()
        self.register_buffer("ndim", torch.as_tensor(ndim))
        self.register_buffer("lam", torch.as_tensor(0.05))
        self.register_buffer("weight", torch.as_tensor(0.5))

    def _sample(self, num_samples: int, context=None):
        if context is not None:
            device = context.device
            dtype = context.dtype
        else:
            device = self.lam.device
            dtype = self.lam.dtype

        return default_width_sampler(
            num_samples,
            self.ndim.item(),
            weight=self.weight.item(),
            lam=self.lam.item(),
            device=device,
            dtype=dtype,
        )

    def _log_prob(self, inputs: Tensor, context=None):
        return torch.logaddexp(
            torch.log(self.weight)
            + torch.log(self.lam)
            - self.lam * inputs
            - torch.log(1 - torch.exp(-self.lam)),
            torch.log(1 - self.weight),
        )


class MixtureWidthSampler(Distribution):
    """
    A mixture of uniform U(0, 1) and zeros:
    p(x) = weight * U(0, 1) + (1 - weight) * delta(x).
    """

    def __init__(self, ndim: int, weight: float = 0.9):
        super().__init__()
        self.register_buffer("ndim", torch.as_tensor(ndim))
        self.register_buffer("weight", torch.as_tensor(weight))

    def _sample(self, num_samples: int, context=None):
        if context is not None:
            device = context.device
            dtype = context.dtype
        else:
            device = self.weight.device
            dtype = self.weight.dtype

        ndim = self.ndim.item()

        return torch.where(
            torch.rand(num_samples, ndim, device=device, dtype=dtype) < self.weight,
            torch.rand(num_samples, ndim, device=device, dtype=dtype),
            torch.zeros(num_samples, ndim, device=device, dtype=dtype),
        )

    def _log_prob(self, inputs: Tensor, context=None):
        raise NotImplementedError


def truncated_exponential_sampler(
    lam: float or Tensor,
    *shape,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """
    Sample from a truncated exponential distribution with support [0, 1]:
    p(x) = lam * exp(-lam * x) / (1 - exp(-lam)).

    Examples:
    >>> from scipy import stats
    >>> gen = torch.Generator().manual_seed(0)

    # when lam is small, it is indistinguishable from the usual exponential distribution:
    >>> stats.kstest(truncated_exponential_sampler(1e-5, 10000, generator=gen) * 1e5, stats.expon.cdf).pvalue > 0.05
    True

    # when lam is large, it is indistinguishable from the uniform distribution:
    >>> stats.kstest(truncated_exponential_sampler(1e5, 10000, generator=gen), stats.uniform.cdf).pvalue > 0.05
    True
    """

    if isinstance(lam, Tensor):
        device = lam.device
        dtype = lam.dtype

    generator = generator or torch.Generator(device=device)

    if isinstance(lam, float):
        lam = torch.empty(shape, device=device, dtype=dtype).fill_(lam)

    elif shape and lam.shape != shape:
        lam = lam.expand(shape)

    # Sample u from U(0, 1)
    u = torch.rand(lam.shape, generator=generator, device=device, dtype=dtype)

    # Compute the inverse CDF to obtain samples
    samples = -torch.log(1 - u * (1 - torch.exp(-1 / lam))) * lam

    return samples


def default_width_sampler(
    batch_size,
    parameter_dim,
    weight: float = 0.5,
    lam: float = 0.05,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """
    Default width sampler for the hyperprior. Samples a mixture of truncated exponential and uniform distributions:
    p(x) = weight * exp(-lam * x) / (1 - exp(-lam)) + (1 - weight).

    Examples:
        >>> from scipy import stats
        >>> gen = torch.Generator().manual_seed(0)

        # when width is zero, it reduces to uniform distribution:
        >>> samples = default_width_sampler(10000, 1, weight=0, lam=1e-5, generator=gen).squeeze()
        >>> stats.kstest(samples, stats.uniform.cdf).pvalue > 0.05
        True

        # when width is one, it reduces to truncated exponential distribution:
        >>> samples = default_width_sampler(10000, 1, weight=1, lam=1e-5, generator=gen).squeeze()
        >>> stats.kstest(samples * 1e5, stats.expon.cdf).pvalue > 0.05
        True

        # mean of the distribution is close to the expected value:
        >>> lam = 0.2
        >>> expected_mean = (lam + 0.5) / 2
        >>> samples = default_width_sampler(10000, 1, weight=0.5, lam=lam, generator=gen).squeeze()
        >>> abs(samples.mean().item() - expected_mean) < 1e-2
        True

    """
    if isinstance(lam, Tensor):
        device = lam.device
        dtype = lam.dtype

    generator = generator or torch.Generator(device=device)
    width = torch.rand(
        batch_size, parameter_dim, device=device, dtype=dtype, generator=generator
    )
    indices = torch.rand_like(width) < weight
    if indices.any():
        width[indices] = truncated_exponential_sampler(
            lam, *width[indices].shape, device=device, dtype=dtype, generator=generator
        )
    return width


if __name__ == "__main__":
    import doctest

    doctest.testmod()
