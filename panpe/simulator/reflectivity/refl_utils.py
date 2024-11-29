# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from math import sqrt, pi

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor

__all__ = [
    "get_reversed_params",
    "get_density_profiles",
    "get_param_labels",
    "get_max_allowed_roughness",
    "get_d_rhos",
    "get_slds_from_d_rhos",
    "angle_to_q",
    "q_to_angle",
    "energy_to_wavelength",
    "wavelength_to_energy",
]


def get_density_profiles(
    thicknesses: Tensor,
    roughnesses: Tensor,
    slds: Tensor,
    z_axis: Tensor = None,
    num: int = 1000,
    min_z_padding: float = 0.2,
    max_z_padding: float = 1.1,
):
    assert torch.all(roughnesses >= 0), "Negative roughness occurred"
    assert torch.all(thicknesses >= 0), "Negative thickness occurred"

    sample_num, layer_num = thicknesses.shape

    if slds.shape[1] == layer_num + 2:
        ambient, slds = slds[:, 0], slds[:, 1:]
    else:
        ambient = 0

    d_rhos = get_d_rhos(slds)

    zs = torch.cumsum(
        torch.cat([torch.zeros(sample_num, 1).to(thicknesses), thicknesses], dim=-1),
        dim=-1,
    )

    if z_axis is None:
        z_axis = torch.linspace(
            -zs.max() * min_z_padding,
            zs.max() * max_z_padding,
            num,
            device=thicknesses.device,
        )[None]
    elif len(z_axis.shape) == 1:
        z_axis = z_axis[None]

    sigmas = roughnesses * sqrt(2)

    profiles = ambient + _get_erf(
        z_axis[:, None], zs[..., None], sigmas[..., None], d_rhos[..., None]
    ).sum(1)

    # d_profiles = _get_gauss(z_axis[:, None], zs[..., None], sigmas[..., None], d_rhos[..., None]).sum(1)

    z_axis = z_axis[0]

    return z_axis, profiles


def get_reversed_params(thicknesses: Tensor, roughnesses: Tensor, slds: Tensor):
    reversed_slds = torch.cumsum(
        torch.flip(
            torch.diff(
                torch.cat([torch.zeros(slds.shape[0], 1).to(slds), slds], dim=-1),
                dim=-1,
            ),
            (-1,),
        ),
        dim=-1,
    )
    reversed_thicknesses = torch.flip(thicknesses, [-1])
    reversed_roughnesses = torch.flip(roughnesses, [-1])
    reversed_params = torch.cat(
        [reversed_thicknesses, reversed_roughnesses, reversed_slds], -1
    )

    return reversed_params


def get_d_rhos(slds: Tensor) -> Tensor:
    d_rhos = torch.cat([slds[:, 0][:, None], torch.diff(slds, dim=-1)], -1)
    return d_rhos


def get_slds_from_d_rhos(d_rhos: Tensor) -> Tensor:
    slds = torch.cumsum(d_rhos, dim=-1)
    return slds


def get_param_labels(
    num_layers: int,
    *,
    thickness_name: str = "d",
    roughness_name: str = r"\sigma",
    sld_name: str = r"\rho",
    substrate_name: str = "sub",
    add_units: bool = True,
    thickness_units: str = "Å",
    roughness_units: str = "Å",
    sld_units: str = "$10^{{-6}}$ Å$^{{-2}}$",
    label_template: str = r"$%s_{%s}$",
    unit_template: str = r"(%s)",
) -> list[str]:
    def _label_str(i: int, label_name: str, units: str) -> str:
        subscript = str(i + 1) if i < num_layers else substrate_name
        label = label_template % (label_name, subscript)
        if add_units:
            label = label + " " + unit_template % units
        return label

    thickness_labels = [
        _label_str(i, thickness_name, thickness_units) for i in range(num_layers)
    ]
    roughness_labels = [
        _label_str(i, roughness_name, roughness_units) for i in range(num_layers + 1)
    ]
    sld_labels = [_label_str(i, sld_name, sld_units) for i in range(num_layers + 1)]

    return thickness_labels + roughness_labels + sld_labels


def get_max_allowed_roughness(
    thicknesses: Tensor, total_max_roughness: Tensor, coef: float = 1.0
) -> Tensor:
    """
    Calculate max allowed roughness for each layer based on thicknesses and total max roughness.

    Args:
        thicknesses: tensor of shape (batch_size, layers_num)
        total_max_roughness: tensor of shape (layer_num + 1, )
        coef: a share of thicknesses to be used as max roughness

    Returns:
        max_roughness: tensor of shape (batch_size, layers_num + 1)


    Examples:
        >>> sampled_thicknesses = torch.tensor([[1., 2., 3.], [10., 55., 5.]])
        >>> total_maximum_roughness = torch.tensor([4., 10., 10., 10.])
        >>> get_max_allowed_roughness(sampled_thicknesses, total_maximum_roughness, coef=1.)
        tensor([[ 1.,  1.,  2.,  3.],
                [ 4., 10.,  5.,  5.]])

        >>> get_max_allowed_roughness(sampled_thicknesses, total_maximum_roughness, coef=0.5)
        tensor([[0.5000, 0.5000, 1.0000, 1.5000],
                [4.0000, 5.0000, 2.5000, 2.5000]])

    """
    batch_size, layers_num = thicknesses.shape
    max_roughness = torch.atleast_2d(total_max_roughness).clone()

    if max_roughness.shape[0] != batch_size:
        max_roughness = max_roughness.repeat(batch_size, 1)

    assert max_roughness.shape == (
        batch_size,
        layers_num + 1,
    ), "Wrong shape of max_roughness"

    boundary = thicknesses * coef

    max_roughness[:, :-1] = torch.minimum(boundary, max_roughness[:, :-1])
    max_roughness[:, 1:] = torch.minimum(boundary, max_roughness[:, 1:])
    return max_roughness


def _get_erf(z, z0, sigma, amp):
    return (torch.erf((z - z0) / sigma) + 1) * amp / 2


def _get_gauss(z, z0, sigma, amp):
    return amp / (sigma * sqrt(2 * pi)) * torch.exp(-((z - z0) ** 2) / 2 / sigma**2)


def angle_to_q(scattering_angle: ndarray or float, wavelength: float):
    """Conversion from full scattering angle (degrees) to scattering vector (inverse angstroms)"""
    return 4 * np.pi / wavelength * np.sin(scattering_angle / 2 * np.pi / 180)


def q_to_angle(q: ndarray or float, wavelength: float):
    """Conversion from scattering vector (inverse angstroms) to full scattering angle (degrees)"""
    return 2 * np.arcsin(q * wavelength / (4 * np.pi)) / np.pi * 180


def energy_to_wavelength(energy: float):
    """Conversion from photon energy (eV) to photon wavelength (angstroms)"""
    return 1.2398 / energy * 1e4


def wavelength_to_energy(wavelength: float):
    """Conversion from photon wavelength (angstroms) to photon energy (eV)"""
    return 1.2398 / wavelength * 1e4


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
