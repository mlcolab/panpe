# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Union

from itertools import product

import torch
from torch import Tensor, nn

from panpe.inference.measured_data import MeasuredData
from panpe.simulator import ReflectometrySimulator
from panpe.nns import FlowWrapper

from panpe.likelihood_refinement.importance_sampling.snis import (
    ImportanceSampling,
    SNISBackend,
)
from panpe.likelihood_refinement.mcmc.torchemcee import run_mcmc

from panpe.inference.inference_processing import (
    InferenceProcessing,
    QScalingProcessing,
)

from panpe.visualization.plot_sampled_profiles import plot_sampled_profiles
from panpe.utils import tqdm


class InferenceResult(nn.Module):
    """
    Class for running PANPE and storing the results of the inference.
    """

    def __init__(
        self,
        model: "InferenceModel",
        data: MeasuredData,
        processing: InferenceProcessing,
        snis_backend: SNISBackend = None,
        sim_theta: Tensor = None,
    ):
        super().__init__()

        self.model = model
        self.data = data
        self.sim_theta = sim_theta

        self.processing = processing
        self.importance_sampling = ImportanceSampling(snis_backend)

    @torch.no_grad()
    def log_prob(self, theta: Tensor) -> Tensor:
        """
        Returns the log probability of the data given the parameters.
        """
        return self.model.simulator.log_prob(
            theta, self.data.data, self.data.q, self.data.sigmas, self.data.phi
        )

    def _data2dict(self, data: MeasuredData) -> dict[str, Tensor]:
        data = data.to_dict()
        data["fixed_theta_mask"] = self.model.simulator.hyperprior.get_fixed_theta_mask(
            data["phi"]
        )
        return data

    def get_prior_logdetjac(self):
        sample = self.model.simulator.prior.sample_reparameterized(1)
        sample, logdetjac = self.model.simulator.prior.restore_theta(
            sample, self.data.phi
        )
        sample, logdetjac = self.processing.postprocess(sample, logdetjac)
        return logdetjac

    @torch.no_grad()
    def sample_and_log_prob(
        self, num_samples: int, batch_size: int = None
    ) -> tuple[Tensor, Tensor]:
        """
        Samples from the PANPE model and returns the samples and their log probabilities.
        """
        context = self.processing.preprocess(self._data2dict(self.data))
        frozen_flow = self.model.flow.get_flow_with_frozen_context(context)

        samples, log_probs = [], []

        remaining_samples = num_samples

        if batch_size is None:
            batch_size = remaining_samples

        while remaining_samples > 0:
            batch_size = min(batch_size, remaining_samples)
            batch_samples, batch_log_probs = frozen_flow.sample_and_log_prob(batch_size)
            samples.append(batch_samples[0])
            log_probs.append(batch_log_probs[0])
            remaining_samples -= batch_size

        samples = torch.cat(samples, dim=0)
        log_probs = torch.cat(log_probs, dim=0)

        thetas, logdetjac = self.model.simulator.prior.restore_theta(
            samples, context["phi"]
        )
        log_probs += logdetjac
        thetas, log_probs = self.processing.postprocess(thetas, log_probs)

        return thetas, log_probs

    def get_snis_sample_eff_estimation(self) -> float:
        """
        Returns the sample efficiency estimation of the conventional IS based on PANPE-IS samples.
        """
        assert self.importance_sampling.size > 0, "No PANPE-IS samples available."

        return self.importance_sampling.streaming_snis.get_uniform_eff_approx(
            self.model.simulator.prior.get_rvolume() ** 2
            / self.model.simulator.prior.get_volume(self.data.phi)
        )

    def reset(self):
        """
        Resets the inference result; deletes all samples.
        """
        self.processing.reset()
        self.importance_sampling.reset()

    @torch.no_grad()
    def run_panpe_mcmc(self, num_iterations: int, num_chains: int = 2**14, **kwargs):
        """
        Runs MCMC initialized with samples from PANPE (PANPE-MCMC).
        """
        thetas, _ = self.sample_and_log_prob(num_chains)

        return run_mcmc(thetas, self.log_prob, num_iterations, **kwargs)

    def plot_sampled_profiles(
        self,
        num_samples: int = 1000,
        thetas: Tensor = None,
        use_weights: bool = True,
        show_prior: bool = False,
        **kwargs,
    ):
        """
        Plots the sampled SLD profiles and reflectometry curves along with the data.
        """
        if thetas is None:
            if use_weights:
                thetas, _ = self.importance_sampling.sample(num_samples)
            else:
                thetas = self.importance_sampling.params[-num_samples:]
                if not len(thetas):
                    raise ValueError("No samples available.")

        if show_prior:
            # Below is a hacky way to visualize priors via SLD profiles. There might be
            # an analytical solution, but as a quick fix, we just sample from
            # the prior and add the corners of the prior hypercube.

            prior_thetas = self.model.simulator.prior.sample(num_samples, self.data.phi)
            prior_thetas = torch.cat(
                [
                    prior_thetas,
                    _get_corner_thetas(
                        *self.model.simulator.hyperprior.get_total_theta_ranges(
                            self.data.phi
                        )
                    ),
                ],
                dim=0,
            )
        else:
            prior_thetas = None

        if "used_theta" not in kwargs and self.sim_theta is not None:
            kwargs["used_theta"] = self.sim_theta.to(self.data.q)

        plot_sampled_profiles(
            self.data,
            thetas.to(self.data.q),
            self.model.simulator.physical_model,
            prior_thetas=prior_thetas,
            **kwargs,
        )

    @torch.no_grad()
    def run_panpe_snis(
        self,
        target_neff: int = 500,
        max_num_samples: int = 2**20,
        batch_size: int = 2**15,
        verbose: bool = True,
    ) -> None:
        """
        Runs self-normalized importance sampling using PANPE as a proposal distribution (PANPE-IS).
        """

        n = 0
        neff = 0

        pbar = tqdm(
            total=target_neff, disable=not verbose, desc="PANPE-IS", unit=" Neff"
        )

        while self.importance_sampling.neff < target_neff:
            thetas, log_probs = self.sample_and_log_prob(batch_size)

            if not torch.isfinite(log_probs).all():
                raise ValueError(
                    "Log probabilities contain NaNs or Infs. "
                    "Note that SNIS is not applicable with parameter-conditioned posterior estimation. "
                    "Use PANPE-MCMC instead."
                )

            log_posteriors = self.log_prob(thetas)
            self.importance_sampling.update(log_posteriors, log_probs, thetas)

            pbar.update(round(self.importance_sampling.neff - neff))
            neff = self.importance_sampling.neff
            n += batch_size

            if n >= max_num_samples:
                if verbose:
                    print(f"Maximum number of samples reached. Neff = {neff:.1f}")
                break


class InferenceModel(nn.Module):
    """
    PANPE inference model with the simulator.
    """

    def __init__(self, simulator: ReflectometrySimulator, flow: FlowWrapper):
        super().__init__()
        self.simulator = simulator
        self.flow = flow
        self.eval()

    @staticmethod
    def from_config(config: Union[dict, str]) -> InferenceModel:
        """
        Initializes the inference model from a config.
        """
        from panpe.config_utils.init_from_config import init_inference_model_from_config

        return init_inference_model_from_config(config)

    def forward(
        self,
        data: MeasuredData,
        snis_backend: SNISBackend = None,
        other_processing: InferenceProcessing = None,
        **kwargs,
    ) -> InferenceResult:
        return self.init_inference_result(
            data, snis_backend=snis_backend, other_processing=other_processing, **kwargs
        )

    @torch.no_grad()
    def init_inference_result(
        self,
        data: MeasuredData,
        run_snis: bool = True,
        snis_backend: SNISBackend = None,
        other_processing: InferenceProcessing = None,
        sim_theta: Tensor = None,
        **kwargs,
    ) -> InferenceResult:
        self.eval()

        processing = QScalingProcessing(self.simulator)

        if other_processing is not None:
            processing += other_processing

        res = InferenceResult(
            self,
            data,
            processing=processing,
            snis_backend=snis_backend,
            sim_theta=sim_theta,
        )

        if run_snis:
            res.run_panpe_snis(**kwargs)
        return res

    @torch.no_grad()
    def sample_simulated_data(
        self, run_snis: bool = True, snis_backend: SNISBackend = None
    ):
        batch = self.simulator.sample(1)

        sim_theta = batch["theta"]

        data = MeasuredData(batch["q"], batch["data"], batch["sigmas"], batch["phi"])

        return self.init_inference_result(
            data, run_snis=run_snis, snis_backend=snis_backend, sim_theta=sim_theta
        )


def _get_corner_thetas(theta_min: Tensor, theta_max: Tensor) -> Tensor:
    combinations = list(
        product(*zip(theta_min.squeeze().tolist(), theta_max.squeeze().tolist()))
    )

    corner_thetas = torch.tensor(combinations).to(theta_min)
    return corner_thetas
