# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import Tensor

from nflows.transforms import Transform
from nflows.distributions import Distribution

from panpe.simulator.distributions import BoxUniform

__all__ = [
    "ReparameterizedDistribution",
    "ParameterizedPrior",
    "UniformParameterizedPrior",
    "AffineBoundConditionedTransform",
]


class ReparameterizedDistribution(Distribution):
    """
    A wrapper class for a reparameterized distribution.
    """

    def __init__(self, distribution: Distribution, transform: Transform):
        super().__init__()
        self.dist = distribution
        self.transform = transform

    def sample_reparameterized(self, num_samples: int) -> Tensor:
        """
        Sample from the reparameterized distribution.
        """
        return self.dist.sample(num_samples)

    def log_prob_reparameterized(self, reparameterized_theta: Tensor) -> Tensor:
        """
        Compute the log probability of the reparameterized samples.
        """
        return self.dist.log_prob(reparameterized_theta)

    def restore_theta(
        self, reparameterized_theta: Tensor, phi: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Restore theta from the reparameterized parameters and the hyperprior parameters phi.
        Args:
            reparameterized_theta: reparameterized samples of shape (num_samples, num_params)
            phi: hyperprior parameters of shape (num_samples, num_hyperparams)

        Returns:
            theta of shape (num_samples, num_params) and logdetjac of shape (num_samples, )

        """
        return self.transform(reparameterized_theta, phi)

    def reparameterize_theta(self, theta: Tensor, phi: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameterize theta using the hyperprior parameters phi.

        Args:
            theta: samples of shape (num_samples, num_params)
            phi: hyperprior parameters of shape (num_samples, num_hyperparams)

        Returns:
            reparameterized theta of shape (num_samples, num_params) and logdetjac of shape (num_samples, )


        """
        return self.transform.inverse(theta, phi)

    def _log_prob(self, theta: Tensor, phi: Tensor) -> Tensor:
        """
        Compute the log probability of the samples.

        Args:
            theta: samples of shape (num_samples, num_params)
            phi: hyperprior parameters of shape (num_samples, num_hyperparams)

        Returns:
            log probability of the samples of shape (num_samples, )
        """
        assert phi is not None, "phi must be provided"
        reparameterized_theta, logdetjac = self.transform.inverse(theta, phi)
        return self.log_prob_reparameterized(reparameterized_theta) + logdetjac

    def _sample(self, num_samples: int, phi: Tensor = None) -> Tensor:
        """
        Sample from the distribution.
        """
        assert phi is not None, "phi must be provided"

        return self.sample_with_reparam(num_samples, phi)[0]

    def sample_with_reparam(
        self, num_samples: int, phi: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Sample from the distribution. Returns both theta and reparameterized theta.
        """
        assert phi is not None, "phi must be provided"

        if phi.ndim == 1:
            phi = phi.unsqueeze(0).expand(num_samples, -1)
        elif phi.ndim == 2 and phi.shape[0] != 1:
            assert (
                phi.shape[0] == num_samples
            ), "phi must have the same number of samples as num_samples"

        reparameterized_theta = self.sample_reparameterized(num_samples)
        theta = self.restore_theta(reparameterized_theta, phi)[0]
        return theta, reparameterized_theta


class ParameterizedPrior(ReparameterizedDistribution):
    """
    An abstract class for parameterized prior distributions for the PANPE model.
    """

    def get_fixed_theta_mask(self, phi: Tensor) -> Tensor:
        """
        Returns the mask of the fixed theta parameters.
        """
        raise NotImplementedError

    def get_rvolume(self) -> float:
        """
        Returns the volume of the reparameterized prior distribution, defined
        as the mathematical expectation of the probability density.
        """
        raise NotImplementedError

    def get_volume(self, phi: Tensor) -> Tensor:
        """
        Returns the volume of the prior distribution, defined
        as the mathematical expectation of the probability density.
        """
        raise NotImplementedError


class UniformParameterizedPrior(ParameterizedPrior):
    """
    A uniform prior distribution for the PANPE model.
    """

    def __init__(self, ndim: int):
        super().__init__(
            BoxUniform([-1.0] * ndim, [1.0] * ndim), AffineBoundConditionedTransform()
        )
        self.ndim = ndim

    def get_fixed_theta_mask(self, phi: Tensor) -> Tensor:
        """
        Returns the mask of the fixed theta parameters.
        """
        min_bounds, max_bounds = self.transform.phi_to_bounds(phi)
        fixed_theta_mask = (min_bounds == max_bounds).to(phi)
        return fixed_theta_mask

    def get_rvolume(self) -> float:
        """
        Returns the volume of the reparameterized distribution, defined
        as the mathematical expectation of the probability density.
        """
        return 2**self.ndim

    def get_volume(self, phi: Tensor) -> Tensor:
        """
        Returns the volume of the prior distribution, defined
        as the mathematical expectation of the probability density.
        """
        min_bounds, max_bounds = self.transform.phi_to_bounds(phi)
        return (max_bounds - min_bounds).prod(-1)


class AffineBoundConditionedTransform(Transform):
    """
    A bound conditioned affine transform. The forward transform
    maps [-1, 1] interval to [max_bounds, min_bounds] and
    is defined as
    output = (input + 1) / 2 * (max_bounds - min_bounds) + min_bounds,
    where min_bounds and max_bounds depend on the context (phi).
    """

    def forward(self, inputs, context=None):
        assert context is not None, "context (phi) must be provided"
        min_bounds, max_bounds = self.phi_to_bounds(context)
        detjac = (max_bounds - min_bounds) / 2
        outputs = (inputs + 1) * detjac + min_bounds
        detjac[min_bounds == max_bounds] = 1
        logdetjac = torch.log(detjac).sum(-1)
        return outputs, logdetjac

    def inverse(self, inputs, context=None):
        assert context is not None, "context (phi) must be provided"
        min_bounds, max_bounds = self.phi_to_bounds(context)
        fixed_indices = min_bounds == max_bounds
        detjac = (max_bounds - min_bounds) / 2
        detjac[fixed_indices] = 1
        outputs = (inputs - min_bounds) / detjac - 1
        logdetjac = -torch.log(detjac).sum(-1)
        outputs[fixed_indices.expand_as(outputs)] = 0.0
        return outputs, logdetjac

    @staticmethod
    def phi_to_bounds(phi: Tensor) -> tuple[Tensor, Tensor]:
        """
        Convert hyperprior parameters phi to min_bounds and max_bounds.
        """
        num_hyperpriors = phi.shape[-1]
        assert num_hyperpriors % 2 == 0
        min_bounds, max_bounds = (
            phi[..., : num_hyperpriors // 2],
            phi[..., num_hyperpriors // 2 :],
        )
        return min_bounds, max_bounds
