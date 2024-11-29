# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Uniform

from panpe.simulator.reflectivity.refl_utils import get_max_allowed_roughness
from panpe.simulator.priors import (
    ParameterizedPrior,
    UniformParameterizedPrior,
)
from panpe.simulator.distributions import (
    Distribution,
    DefaultWidthSampler,
    MixtureWidthSampler,
    BoxUniform,
)
from panpe.nns.scalers import DataScaler, AffineScaler

__all__ = [
    "Hyperprior",
    "HyperpriorForLayeredStructures",
    "BasicHyperpriorForUniformPriors",
    "HyperpriorForUniformPriorsWithConstrainedRoughness",
]


class Hyperprior(nn.Module):
    """
    An abstract class for the hyperprior distribution for the PANPE model.
    """

    @property
    def phi_dim(self):
        """
        Returns the dimensionality of the hyperprior distribution.
        """
        raise NotImplementedError

    def sample(self, num: int) -> Tensor:
        """
        Samples phi from the hyperprior distribution.
        """
        raise NotImplementedError

    @property
    def prior(self) -> ParameterizedPrior:
        """
        Returns the prior distribution.
        """
        raise NotImplementedError

    @property
    def total_phi_ranges(self) -> tuple[Tensor, Tensor]:
        """
        Returns the total ranges of the phi parameters. The first tensor contains the minimum values and the second
        tensor contains the maximum values. Both tensors have shape (1, phi_dim).
        """
        raise NotImplementedError

    def get_total_theta_ranges(self, phi: Tensor) -> tuple[Tensor, Tensor]:
        """
        Returns the total ranges of the theta parameters. The first tensor contains the minimum values and the second
        tensor contains the maximum values. Both tensors have shape (1, theta_dim).
        """
        raise NotImplementedError

    def get_phi_scaler(self) -> DataScaler:
        """
        Returns a scaler for the phi parameters used in the neural network.
        """
        return AffineScaler(*self.total_phi_ranges)

    def scale_phi_with_q_factor(self, phi: Tensor, q_factor: float or Tensor) -> Tensor:
        """
        Scales the phi parameters with the q factor following scaling invariance in reflectometry.

        q_factor = max(q_new) / max(q_old)
        """
        raise NotImplementedError

    def get_fixed_theta_mask(self, phi: Tensor) -> Tensor:
        """
        Returns the mask of the fixed theta parameters.
        """
        return self.prior.get_fixed_theta_mask(phi)


class HyperpriorForLayeredStructures(Hyperprior):
    """
    An abstract class for the hyperprior distribution for reflectometry simulations with basic layered structures.
    """

    @classmethod
    def from_param_ranges(
        cls,
        num_layers: int,
        thickness_range: tuple[float, float],
        roughness_range: tuple[float, float],
        sld_range: tuple[float, float],
        q_offset_range: tuple[float, float] = None,
        r_scale_range: tuple[float, float] = None,
        log_bkg_range: tuple[float, float] = None,
        **kwargs,
    ):
        raise NotImplementedError


class BasicHyperpriorForUniformPriors(HyperpriorForLayeredStructures):
    def __init__(
        self,
        total_param_ranges: Tensor,
        q_scale_rates: Tensor,
        width_distribution: Distribution or str = None,
        max_widths: Tensor = None,
    ):
        super().__init__()
        self.theta_dim = total_param_ranges.shape[0]
        min_bounds, max_bounds = total_param_ranges.T[:, None]

        if max_widths is None:
            max_widths = torch.ones_like(min_bounds)

        self.register_buffer("min_bounds", min_bounds)
        self.register_buffer("max_bounds", max_bounds)
        self.register_buffer("q_scale_rates", torch.atleast_2d(q_scale_rates))
        self.register_buffer(
            "max_widths", torch.atleast_2d(torch.as_tensor(max_widths).to(min_bounds))
        )

        self.width_distribution = self._init_width_distribution(width_distribution)
        self._prior = UniformParameterizedPrior(self.theta_dim)

    def _init_width_distribution(self, width_distribution: Distribution or str = None):
        if isinstance(width_distribution, str):
            if width_distribution == "default":
                width_distribution = DefaultWidthSampler(self.theta_dim)
            elif width_distribution == "mixture":
                width_distribution = MixtureWidthSampler(self.theta_dim)
            elif width_distribution == "uniform":
                width_distribution = BoxUniform(
                    [0.0] * self.theta_dim, [1.0] * self.theta_dim
                )
        elif width_distribution is None:
            width_distribution = DefaultWidthSampler(self.theta_dim)

        assert isinstance(
            width_distribution, Distribution
        ), "width_distribution should be a Distribution."
        return width_distribution

    def scale_phi_with_q_factor(self, phi: Tensor, q_factor: float) -> Tensor:
        """
        Scales the phi parameters with the q factor following scaling invariance in reflectometry.

        q_factor = max(q_new) / max(q_old)
        """
        q_scale_vector = q_factor**self.q_scale_rates
        return phi * q_scale_vector

    @property
    def phi_dim(self):
        return self.theta_dim * 2

    @property
    def total_phi_ranges(self) -> tuple[Tensor, Tensor]:
        phi_min = torch.cat([self.min_bounds, self.min_bounds], -1)
        phi_max = torch.cat([self.max_bounds, self.max_bounds], -1)
        return phi_min, phi_max

    def get_total_theta_ranges(self, phi: Tensor) -> tuple[Tensor, Tensor]:
        min_bounds, max_bounds = phi[..., : self.theta_dim], phi[..., self.theta_dim :]
        return min_bounds, max_bounds

    @property
    def prior(self) -> ParameterizedPrior:
        return self._prior

    @classmethod
    def from_param_ranges(
        cls,
        num_layers: int,
        thickness_range: tuple[float, float],
        roughness_range: tuple[float, float],
        sld_range: tuple[float, float],
        q_offset_range: tuple[float, float] = None,
        r_scale_range: tuple[float, float] = None,
        log_bkg_range: tuple[float, float] = None,
        width_distribution: Distribution = None,
        max_widths: Tensor = None,
    ):
        total_param_ranges, q_scale_rates = _get_total_param_ranges(
            num_layers,
            thickness_range,
            roughness_range,
            sld_range,
            q_offset_range,
            r_scale_range,
            log_bkg_range,
        )
        return cls(
            total_param_ranges, q_scale_rates, width_distribution, max_widths=max_widths
        )

    def sample(self, num: int) -> Tensor:
        widths = (
            self.width_distribution.sample(num).to(self.min_bounds) * self.max_widths
        )
        centers = Uniform(widths / 2, 1 - widths / 2).sample()
        min_bounds = (centers - widths / 2) * (
            self.max_bounds - self.min_bounds
        ) + self.min_bounds
        max_bounds = (centers + widths / 2) * (
            self.max_bounds - self.min_bounds
        ) + self.min_bounds
        phi = torch.cat([min_bounds, max_bounds], -1)
        return phi


class HyperpriorForUniformPriorsWithConstrainedRoughness(
    BasicHyperpriorForUniformPriors
):
    """
    A hyperprior for the PANPE model with uniform priors on the parameters and additional constraints on the
    roughness parameters.

    Examples:
        >>> hyperprior = HyperpriorForUniformPriorsWithConstrainedRoughness.from_param_ranges(
        ...     2,
        ...     (0., 100.),
        ...     (0., 50.),
        ...     (0., 60.),
        ...     (0., 1.),
        ...     (-1., 1.),
        ...     max_total_thickness=10.,
        ... )
        >>> hyperprior.total_phi_ranges[0].shape
        torch.Size([1, 20])
        >>> sampled_phi = hyperprior.sample(32)
        >>> sampled_phi.shape
        torch.Size([32, 20])
        >>> ndim = sampled_phi.shape[-1] // 2
        >>> min_bounds, max_bounds = sampled_phi[..., :ndim], sampled_phi[..., ndim:]
        >>> max_roughnesses = max_bounds[:, hyperprior.roughness_mask]
        >>> max_thicknesses = max_bounds[:, hyperprior.thickness_mask]
        >>> max_allowed_roughnesses = get_max_allowed_roughness(
        ...     max_thicknesses, hyperprior.max_bounds[..., hyperprior.roughness_mask]
        ... )
        >>> torch.all(max_roughnesses <= max_allowed_roughnesses).item()
        True
        >>> torch.all(min_bounds >= hyperprior.min_bounds).item()
        True
        >>> torch.all(max_bounds < hyperprior.max_bounds).item()
        True
        >>> torch.all(min_bounds <= max_bounds).item()
        True

        >>> scaled_phi = hyperprior.scale_phi_with_q_factor(sampled_phi, 2.)
        >>> restored_phi = hyperprior.scale_phi_with_q_factor(scaled_phi, 0.5)
        >>> torch.allclose(sampled_phi, restored_phi)
        True
        >>> [round(v, 1) for v in (scaled_phi[0] / sampled_phi[0]).tolist()]
        [0.5, 0.5, 0.5, 0.5, 0.5, 4.0, 4.0, 4.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 4.0, 4.0, 4.0, 2.0, 1.0]

        >>> max_thicknesses = max_bounds[..., hyperprior.thickness_mask].sum(-1)
        >>> torch.all(max_thicknesses <= 10.).item()
        True
        >>> torch.all(max_bounds < hyperprior.max_bounds).item()
        True

    """

    def __init__(
        self,
        total_param_ranges: Tensor,
        q_scale_rates: Tensor,
        thickness_mask: Tensor,
        roughness_mask: Tensor,
        width_distribution: Distribution = None,
        max_total_thickness: float = None,
        coef: float = 1.0,
        max_widths: Tensor = None,
    ):
        super().__init__(
            total_param_ranges, q_scale_rates, width_distribution, max_widths=max_widths
        )
        self.register_buffer("thickness_mask", thickness_mask)
        self.register_buffer("roughness_mask", roughness_mask)
        self.coef = coef
        self.max_total_thickness = max_total_thickness

    @classmethod
    def from_param_ranges(
        cls,
        num_layers: int,
        thickness_range: tuple[float, float],
        roughness_range: tuple[float, float],
        sld_range: tuple[float, float],
        q_offset_range: tuple[float, float] = None,
        r_scale_range: tuple[float, float] = None,
        log_bkg_range: tuple[float, float] = None,
        width_distribution: Distribution = None,
        max_total_thickness: float = None,
        coef: float = 0.5,
        max_widths: Tensor = None,
    ):
        total_param_ranges, q_scale_rates = _get_total_param_ranges(
            num_layers,
            thickness_range,
            roughness_range,
            sld_range,
            q_offset_range,
            r_scale_range,
            log_bkg_range,
        )

        thickness_mask = torch.zeros(total_param_ranges.shape[0], dtype=torch.bool)
        roughness_mask = torch.zeros(total_param_ranges.shape[0], dtype=torch.bool)
        thickness_mask[:num_layers] = True
        roughness_mask[num_layers : num_layers * 2 + 1] = True

        return cls(
            total_param_ranges,
            q_scale_rates,
            thickness_mask,
            roughness_mask,
            width_distribution,
            max_total_thickness=max_total_thickness,
            coef=coef,
            max_widths=max_widths,
        )

    def sample(self, num: int) -> Tensor:
        phi = super().sample(num)

        ndim = phi.shape[-1] // 2
        min_bounds, max_bounds = phi[..., :ndim], phi[..., ndim:]

        if self.max_total_thickness is not None:
            # TODO: This is a bit of a hack. We should probably do something more elegant.
            total_thickness = max_bounds[:, self.thickness_mask].sum(-1)
            indices = total_thickness > self.max_total_thickness

            if indices.any():
                eps = 0.01  # to avoid numerical issues.
                rand_scale = torch.rand_like(total_thickness) * eps + 1 - eps
                scale_coef = self.max_total_thickness / total_thickness * rand_scale
                scale_coef[~indices] = 1.0
                min_bounds[:, self.thickness_mask] *= scale_coef[:, None]
                max_bounds[:, self.thickness_mask] *= scale_coef[:, None]

                min_bounds[:, self.thickness_mask] = torch.clamp_min(
                    min_bounds[:, self.thickness_mask],
                    self.min_bounds[:, self.thickness_mask],
                )

                max_bounds[:, self.thickness_mask] = torch.clamp_min(
                    max_bounds[:, self.thickness_mask],
                    self.min_bounds[:, self.thickness_mask],
                )

        total_max_thickness = max_bounds[:, self.thickness_mask]
        total_max_roughness = self.max_bounds[..., self.roughness_mask]
        total_min_roughness = self.min_bounds[..., self.roughness_mask]

        max_allowed_roughness = get_max_allowed_roughness(
            total_max_thickness, total_max_roughness, self.coef
        )
        scale_coef = (max_allowed_roughness - total_min_roughness) / (
            total_max_roughness - total_min_roughness
        )

        assert torch.all(scale_coef <= 1.0)

        min_bounds[:, self.roughness_mask] = (
            min_bounds[:, self.roughness_mask] - total_min_roughness
        ) * scale_coef + total_min_roughness

        max_bounds[:, self.roughness_mask] = (
            max_bounds[:, self.roughness_mask] - total_min_roughness
        ) * scale_coef + total_min_roughness

        phi = torch.cat([min_bounds, max_bounds], -1)

        return phi


def _get_total_param_ranges(
    num_layers: int,
    thickness_range: tuple[float, float],
    roughness_range: tuple[float, float],
    sld_range: tuple[float, float],
    q_offset_range: tuple[float, float] = None,
    r_scale_range: tuple[float, float] = None,
    log_bkg_range: tuple[float, float] = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> tuple[Tensor, Tensor]:
    """

    Returns the total ranges of the parameters and the q scale vector for the hyperprior.

    Examples:
        >>> total_param_ranges, q_scale = _get_total_param_ranges(
        ...     1,
        ...     (0., 100.),
        ...     (0., 50.),
        ...     (0., 60.),
        ...     (-1., 1.),
        ...     (0., 1.),
        ...     (-9., -4.),
        ... )
        >>> q_scale.tolist()
        [-1.0, -1.0, -1.0, 2.0, 2.0, 1.0, 0.0, 0.0, -1.0, -1.0, -1.0, 2.0, 2.0, 1.0, 0.0, 0.0]
        >>> total_param_ranges.tolist()
        [[0.0, 100.0], [0.0, 50.0], [0.0, 50.0], [0.0, 60.0], [0.0, 60.0], [-1.0, 1.0], [0.0, 1.0], [-9.0, -4.0]]
    """

    additional_param_ranges = []
    if q_offset_range is not None:
        additional_param_ranges.append(q_offset_range)
    if r_scale_range is not None:
        additional_param_ranges.append(r_scale_range)
    if log_bkg_range is not None:
        additional_param_ranges.append(log_bkg_range)

    total_param_ranges = torch.tensor(
        [
            *([thickness_range] * num_layers),
            *([roughness_range] * (num_layers + 1)),
            *([sld_range] * (num_layers + 1)),
            *additional_param_ranges,
        ],
        dtype=dtype,
        device=device,
    )

    q_scale = torch.zeros_like(total_param_ranges)

    q_scale[: num_layers * 2 + 1] = -1.0
    q_scale[num_layers * 2 + 1 : 3 * num_layers + 2] = 2.0
    if q_offset_range is not None:
        q_scale[3 * num_layers + 2] = 1.0

    q_scale = q_scale.T.flatten()

    return total_param_ranges, q_scale


if __name__ == "__main__":
    import doctest

    doctest.testmod()
