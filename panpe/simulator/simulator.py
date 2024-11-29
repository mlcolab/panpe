# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import Tensor, nn

from panpe.simulator.physical_models import PhysicalModel
from panpe.simulator.q_simulator import QSimulator
from panpe.simulator.measurement_noise_simulator import MeasurementNoiseSimulator
from panpe.simulator.hyperpriors import Hyperprior


class Simulator(nn.Module):
    def sample(self, batch_size: int) -> dict[str, Tensor]:
        raise NotImplementedError

    def log_prob(
        self, theta: Tensor, data: Tensor, q: Tensor, sigmas: Tensor, phi: Tensor
    ) -> Tensor:
        raise NotImplementedError


class ReflectometrySimulator(Simulator):
    def __init__(
        self,
        physical_model: PhysicalModel,
        hyperprior: Hyperprior,
        q_simulator: QSimulator,
        measurement_noise: MeasurementNoiseSimulator,
    ):
        super().__init__()
        self.physical_model = physical_model
        self.q_simulator = q_simulator
        self.measurement_noise = measurement_noise
        self.hyperprior = hyperprior

    @property
    def prior(self):
        return self.hyperprior.prior

    def sample(self, batch_size: int) -> dict[str, Tensor]:
        batch = {}
        batch["phi"] = self.hyperprior.sample(batch_size)
        batch["fixed_theta_mask"] = self.hyperprior.get_fixed_theta_mask(batch["phi"])
        batch["theta"], batch["reparameterized_theta"] = self.prior.sample_with_reparam(
            batch_size, batch["phi"]
        )

        batch["q"] = self.q_simulator.sample(batch_size, batch)
        batch["noiseless_data"] = self.physical_model.simulate_reflectivity(
            batch["q"], batch["theta"]
        )

        batch["sigmas"] = self.measurement_noise.sample_sigmas(batch["noiseless_data"])
        batch["data"] = self.measurement_noise.sample(
            batch["noiseless_data"], batch["sigmas"]
        )

        return batch

    def log_prob(
        self, theta: Tensor, data: Tensor, q: Tensor, sigmas: Tensor, phi: Tensor
    ) -> Tensor:
        return self.calc_unnormalized_posterior(theta, data, q, sigmas, phi)

    def calc_likelihood(
        self, theta: Tensor, data: Tensor, q: Tensor, sigmas: Tensor
    ) -> Tensor:
        return self.measurement_noise.log_prob(
            data, sigmas, self.physical_model.simulate_reflectivity(q, theta)
        )

    def calc_unnormalized_posterior(
        self, theta: Tensor, data: Tensor, q: Tensor, sigmas: Tensor, phi: Tensor
    ) -> Tensor:
        reparameterized_theta = self.prior.reparameterize_theta(theta, phi)[0]
        log_probs = self.prior.log_prob_reparameterized(reparameterized_theta)
        finite_indices = torch.isfinite(log_probs)

        if finite_indices.any():
            log_probs[finite_indices] += self.calc_likelihood(
                theta[finite_indices], data, q, sigmas
            )
        return log_probs
