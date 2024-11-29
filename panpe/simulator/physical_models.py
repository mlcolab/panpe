# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional, Union

from math import log

import torch
from torch import Tensor, nn

from panpe.simulator.reflectivity import simulate_reflectivity
from panpe.simulator.reflectivity.refl_utils import (
    get_param_labels,
    get_density_profiles,
)


class PhysicalModel(nn.Module):
    NAME: str

    @property
    def theta_dim(self) -> int:
        raise NotImplementedError

    @property
    def param_names(self) -> tuple[str]:
        raise NotImplementedError

    def simulate_reflectivity(self, q: Tensor, theta: Tensor) -> Tensor:
        raise NotImplementedError

    def get_sld_profiles(
        self, theta: Tensor, z: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def scale_theta_with_q_factor(
        self, theta: Tensor, q: float
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    @property
    def q_offset_idx(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def r_scale_idx(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def log_bkg_idx(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def sld_indices(self) -> tuple[int]:
        raise NotImplementedError

    def get_total_thickness(self, theta: Tensor) -> Tensor:
        raise NotImplementedError


class BasicLayerStructureModel(PhysicalModel):
    NAME = "basic-layer-structure"

    def __init__(
        self,
        num_layers: int,
        enable_q_misalignment: bool = False,
        enable_r_misalignment: bool = False,
        enable_background: bool = False,
        min_background: float = 1e-10,
        dq_q: float = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_dq_misalignment = enable_q_misalignment
        self.use_dr_misalignment = enable_r_misalignment
        self.use_background = enable_background
        self.min_background = min_background if not enable_background else 0.0
        self.dq_q = dq_q
        self._theta_dim = (
            3 * num_layers
            + 2
            + int(enable_background)
            + int(enable_q_misalignment)
            + int(enable_r_misalignment)
        )

    def _theta2refl_kwargs(self, theta: Tensor):
        assert theta.shape[-1] == self._theta_dim

        num_layers = self.num_layers

        params = dict(
            thickness=theta[..., :num_layers],
            roughness=theta[..., num_layers : 2 * num_layers + 1],
            sld=theta[..., 2 * num_layers + 1 : 3 * num_layers + 2],
        )

        add_idx = 3 * num_layers + 2

        if self.use_dq_misalignment:
            params["q_offset"] = theta[..., add_idx : add_idx + 1]
            add_idx += 1
        if self.use_dr_misalignment:
            params["r_scale"] = theta[..., add_idx : add_idx + 1]
            add_idx += 1
        if self.use_background:
            params["bkg"] = 10 ** theta[..., add_idx : add_idx + 1]
        elif self.min_background:
            params["bkg"] = self.min_background

        if self.dq_q is not None:
            params["dq_q"] = torch.tensor([[self.dq_q]]).to(theta)

        return params

    @property
    def q_offset_idx(self) -> Optional[int]:
        return 3 * self.num_layers + 2 if self.use_dq_misalignment else None

    @property
    def r_scale_idx(self) -> Optional[int]:
        if not self.use_dr_misalignment:
            return None
        idx = 3 * self.num_layers + 2
        if self.use_dq_misalignment:
            idx += 1
        return idx

    @property
    def log_bkg_idx(self) -> Optional[int]:
        if not self.use_background:
            return None
        idx = 3 * self.num_layers + 2
        if self.use_dq_misalignment:
            idx += 1
        if self.use_dr_misalignment:
            idx += 1
        return idx

    @property
    def sld_indices(self) -> tuple[int]:
        return tuple(range(2 * self.num_layers + 1, 3 * self.num_layers + 2))

    def get_sld_profiles(
        self, theta: Tensor, z: Tensor = None, **kwargs
    ) -> tuple[Tensor, Tensor]:
        params = self._theta2refl_kwargs(theta)
        return get_density_profiles(
            thicknesses=params["thickness"],
            roughnesses=params["roughness"],
            slds=params["sld"],
            z_axis=z,
            **kwargs,
        )

    @property
    def theta_dim(self) -> int:
        return self._theta_dim

    @property
    def param_names(self) -> tuple[str]:
        additional_labels = []
        if self.use_dq_misalignment:
            additional_labels.append(r"$\Delta q$ (Å$^{{-1}}$)")
        if self.use_dr_misalignment:
            additional_labels.append(r"$\Delta R$")
        if self.use_background:
            additional_labels.append(r"$log(bkg)$")

        return tuple(
            get_param_labels(self.num_layers, add_units=True) + additional_labels
        )

    def simulate_reflectivity(self, q: Tensor, theta: Tensor) -> Tensor:
        return simulate_reflectivity(q, **self._theta2refl_kwargs(theta))

    def scale_theta_with_q_factor(
        self, theta: Tensor, q_factor: float
    ) -> tuple[Tensor, Tensor]:
        theta = theta.clone()
        logdetjac = -torch.ones_like(theta[..., 0]) * log(q_factor)

        theta[..., : 2 * self.num_layers + 1] /= q_factor
        theta[..., 2 * self.num_layers + 1 : 3 * self.num_layers + 2] *= q_factor**2
        if self.use_dq_misalignment:
            theta[..., 3 * self.num_layers + 2] *= q_factor
            logdetjac *= 2

        return theta, logdetjac

    def get_total_thickness(self, theta: Tensor) -> Tensor:
        return theta[..., : self.num_layers].sum(dim=-1)
