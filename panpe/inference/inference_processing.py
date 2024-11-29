# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
import warnings

from torch import Tensor

from panpe.simulator import ReflectometrySimulator


class InferenceProcessing:
    def preprocess(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        raise NotImplementedError

    def postprocess(self, thetas: Tensor, log_probs: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, InferenceProcessingPipeline):
            return InferenceProcessingPipeline(self, *other.processing_list)
        elif isinstance(other, InferenceProcessing):
            return InferenceProcessingPipeline(self, other)
        raise NotImplemented

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class InferenceProcessingPipeline(InferenceProcessing):
    def __init__(self, *processing: InferenceProcessing):
        self.processing_list = processing

    def preprocess(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        for transform in self.processing_list:
            data = transform.preprocess(data)
        return data

    def postprocess(self, thetas: Tensor, log_probs: Tensor) -> tuple[Tensor, Tensor]:
        for transform in reversed(self.processing_list):
            thetas, log_probs = transform.postprocess(thetas, log_probs)
        return thetas, log_probs

    def reset(self):
        for transform in self.processing_list:
            transform.reset()

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join([repr(p) for p in self.processing_list])})'


class QScalingProcessing(InferenceProcessing):
    def __init__(self, simulator: ReflectometrySimulator):
        self.simulator = simulator
        self.q_factor = 1.0

    def preprocess(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        q, phi = data["q"], data["phi"]
        self.q_factor = self.simulator.q_simulator.get_q_factor(q)
        q_scaled = q / self.q_factor

        data["q"] = q_scaled

        data = self.simulator.q_simulator.interpolate_data(data)
        q_scaled_phi = self.simulator.hyperprior.scale_phi_with_q_factor(
            phi, 1 / self.q_factor
        )
        phi_min, phi_max = self.simulator.hyperprior.total_phi_ranges

        if (q_scaled_phi < phi_min).any() or (q_scaled_phi > phi_max).any():
            raise ValueError("phi is out of range.")

        data["phi"] = q_scaled_phi

        return data

    def postprocess(self, thetas: Tensor, log_probs: Tensor) -> tuple[Tensor, Tensor]:
        thetas, logdetjac = self.simulator.physical_model.scale_theta_with_q_factor(
            thetas, self.q_factor
        )
        return thetas, log_probs + logdetjac

    def reset(self):
        self.q_factor = 1.0


class RShiftProcessing(InferenceProcessing):
    def __init__(self, simulator: ReflectometrySimulator):
        self.simulator = simulator
        if self.simulator.physical_model.log_bkg_idx is not None:
            warnings.warn(
                """RShiftTransform with log background is not implemented. 
In general, that would require adjusting background via b * r_shift
with the corresponding adjustment of phi. """
            )
            self.r_shift_idx = None
        else:
            self.r_shift_idx = self.simulator.physical_model.r_scale_idx
        self.r_shift = 1.0

    def preprocess(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.r_shift_idx is None:
            return data

        data["data"] = data["data"] / self.r_shift
        return data

    def postprocess(self, thetas: Tensor, log_probs: Tensor) -> tuple[Tensor, Tensor]:
        if self.r_shift_idx is not None:
            self.r_shift *= thetas[..., self.r_shift_idx].mean()
            thetas[..., self.r_shift_idx] *= self.r_shift
        return thetas, log_probs

    def reset(self):
        self.r_shift = 1.0


class QShiftProcessing(InferenceProcessing):
    def __init__(
        self,
        simulator: ReflectometrySimulator,
    ):
        self.simulator = simulator
        self.q_shift_idx = self.simulator.physical_model.q_offset_idx
        self.q_shift = 0.0

    def preprocess(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.q_shift_idx is None:
            return data

        data["q"] = data["q"] + self.q_shift
        return self.simulator.q_simulator.interpolate_data(data)

    def postprocess(self, thetas: Tensor, log_probs: Tensor) -> tuple[Tensor, Tensor]:
        if self.q_shift_idx is not None:
            new_q_shift = thetas[..., self.q_shift_idx].mean()
            thetas[..., self.q_shift_idx] += self.q_shift
            self.q_shift += new_q_shift
        return thetas, log_probs

    def reset(self):
        self.q_shift = 0.0
