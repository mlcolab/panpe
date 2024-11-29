# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor

from panpe.simulator.reflectivity.abeles import abeles_compiled, abeles
from panpe.simulator.reflectivity.memory_eff import abeles_memory_eff
from panpe.simulator.reflectivity.numpy_implementations import (
    kinematical_approximation_np,
    abeles_np,
)
from panpe.simulator.reflectivity.smearing import abeles_constant_smearing
from panpe.simulator.reflectivity.kinematical import kinematical_approximation

__all__ = [
    "simulate_reflectivity",
    "abeles",
    "abeles_compiled",
    "abeles_memory_eff",
    "abeles_np",
    "kinematical_approximation",
    "kinematical_approximation_np",
    "abeles_constant_smearing",
]


def simulate_reflectivity(
    q: Tensor,
    thickness: Tensor,
    roughness: Tensor,
    sld: Tensor,
    dq_q: Tensor = None,
    gauss_num: int = 51,
    xrr_dq: bool = False,
    abeles_func=None,
    q_offset: Tensor = 0.0,
    bkg: Tensor = 0.0,
    r_scale: Tensor = 1.0,
):
    abeles_func = abeles_func or abeles
    q = torch.atleast_2d(q) + q_offset
    q = torch.clamp(q, min=0.0)

    if dq_q is None:
        reflectivity_curves = abeles_func(q, thickness, roughness, sld)
    else:
        reflectivity_curves = abeles_constant_smearing(
            q,
            thickness,
            roughness,
            sld,
            dq=dq_q,
            gauss_num=gauss_num,
            xrr_dq=xrr_dq,
            abeles_func=abeles_func,
        )

    reflectivity_curves = reflectivity_curves * r_scale + bkg

    return reflectivity_curves
