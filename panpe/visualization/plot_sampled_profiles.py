# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Union, Optional

import torch
from torch import Tensor

from matplotlib.colors import Colormap
import matplotlib.pyplot as plt

from panpe.visualization.palettes import get_spectral_cmap
from panpe.simulator import PhysicalModel
from panpe.inference.measured_data import MeasuredData


def plot_sampled_profiles(
    data: MeasuredData,
    thetas: Tensor,
    physical_model: PhysicalModel,
    prior_thetas: Optional[Tensor] = None,
    used_theta: Optional[Tensor] = None,
    q: Union[Tensor, tuple[float, float, int], tuple[float, float], float, None] = None,
    z: Union[Tensor, tuple[float, float, int], tuple[float, float], float, None] = None,
    figsize: tuple = (15, 5),
    data_color: str = "m",
    data_lw: float = 2,
    cmap: Colormap = None,
    samples_alpha: float = 0.2,
    prior_alpha: float = 0.1,
    prior_color: float = "gray",
    prior_lw: float = 0.5,
    show: bool = True,
    capsize: float = None,
    errorbar_kwargs: dict = None,
    data_label: str = None,
    legend_kw: dict = None,
    w_pad: float = 0.1,
    axs: tuple = None,
    norm_z_axis: bool = False,
    max_thickness: float = None,
    fontsize: int = 16,
    dpi: int = 300,
):
    if q is None:
        q = data.q
    else:
        q = _process_q_argument(q).to(thetas)

    if z is not None:
        z = _process_z_argument(z).to(thetas)

    sampled_curves = physical_model.simulate_reflectivity(q, thetas).cpu().numpy()

    q = q.squeeze().cpu().numpy()
    q_data = data.q.squeeze().cpu().numpy()
    measured_curve = data.data.squeeze().cpu().numpy()
    sigmas = data.sigmas.squeeze().cpu().numpy()

    z, sampled_profiles = physical_model.get_sld_profiles(thetas, z)

    if used_theta is not None:
        _, used_profile = physical_model.get_sld_profiles(used_theta, z)
        anchor_profile_for_colors = used_profile
        used_profile = used_profile.squeeze().cpu().numpy()
    else:
        used_profile = None
        anchor_profile_for_colors = sampled_profiles[:1]

    if prior_thetas is not None:
        _, prior_profiles = physical_model.get_sld_profiles(prior_thetas, z=z)
        min_profile = prior_profiles.min(dim=0).values.cpu().numpy()
        max_profile = prior_profiles.max(dim=0).values.cpu().numpy()
    else:
        min_profile = None
        max_profile = None

    color_distances = torch.linalg.norm(
        sampled_profiles - anchor_profile_for_colors, dim=-1
    )
    color_distances = torch.nan_to_num(color_distances, torch.nanmean(color_distances))
    color_distances = 1 - (color_distances - color_distances.min()) / (
        color_distances.max() - color_distances.min()
    )

    color_distances = color_distances.cpu().numpy()
    sampled_profiles = sampled_profiles.cpu().numpy()
    z = z.squeeze().cpu().numpy()

    if norm_z_axis:
        assert (
            max_thickness is not None
        ), "max_thickness must be provided if norm_z_axis is True"
        z = z / max_thickness

    cmap = cmap or get_spectral_cmap()

    with plt.rc_context({"font.size": fontsize}):

        if axs is None:
            fig, axs = plt.subplots(ncols=2, figsize=figsize, dpi=dpi)
        else:
            show = False

        plt.sca(axs[0])

        for curve, distance in zip(sampled_curves, color_distances):
            plt.semilogy(q, curve, alpha=samples_alpha, color=cmap(distance))

        plt.errorbar(
            x=q_data,
            y=measured_curve,
            yerr=sigmas,
            color=data_color,
            lw=data_lw,
            capsize=capsize,
            **(errorbar_kwargs or {}),
            label=data_label,
        )

        plt.grid()
        plt.xlabel(r"$q$ (Å$^{-1}$)")
        plt.ylabel(r"$R(q)$")

        plt.sca(axs[1])

        for profile, distance in zip(sampled_profiles, color_distances):
            plt.plot(z, profile, alpha=samples_alpha, color=cmap(distance))

        if used_profile is not None:
            plt.plot(z, used_profile, color=data_color, lw=data_lw)

        if min_profile is not None:
            plt.plot(z, min_profile, color=prior_color, lw=prior_lw, ls="--")
            plt.plot(z, max_profile, color=prior_color, lw=prior_lw, ls="--")
            plt.fill_between(
                z, min_profile, max_profile, color="gray", alpha=prior_alpha
            )

        plt.grid()
        plt.xlabel(r"$z$ (Å)")
        plt.ylabel(r"$\rho(z)$ ($10^{-6}$ Å$^{-2}$)", rotation=-90, labelpad=25)

        if data_label:
            plt.legend(**(legend_kw or {}))

        axs[1].yaxis.tick_right()
        axs[1].yaxis.set_label_position("right")

        plt.tight_layout(w_pad=w_pad)
        if show:
            plt.show()
        else:
            return axs


def _process_q_argument(
    q: Union[Tensor, tuple[float, float, int], tuple[float, float], float],
    default_size: int = 1000,
) -> Tensor:
    """
    Convert q argument to Tensor.
    """
    if isinstance(q, tuple):
        if len(q) == 2:
            q = torch.linspace(*q, default_size)
        elif len(q) == 3:
            q = torch.linspace(*q)
        else:
            raise ValueError("q tuple must have length 2 or 3.")
    elif isinstance(q, float):
        q = torch.linspace(0, q, default_size)
    else:  # Tensor or Tensor-like
        q = torch.atleast_2d(torch.as_tensor(q))

    return q


def _process_z_argument(
    z: Union[Tensor, tuple[float, float, int], tuple[float, float]],
    default_size: int = 1000,
):
    """
    Convert z argument to Tensor.
    """
    if z is None:
        return z
    if isinstance(z, tuple):
        if len(z) == 2:
            z = torch.linspace(*z, default_size)
        elif len(z) == 3:
            z = torch.linspace(*z)
        else:
            raise ValueError("z tuple must have length 2 or 3.")
    else:  # Tensor or Tensor-like
        z = torch.as_tensor(z)

    return z
