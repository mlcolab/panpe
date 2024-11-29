# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np

import torch
from torch import Tensor, nn

from panpe.simulator.reflectivity.refl_utils import angle_to_q
from panpe.simulator.physical_models import PhysicalModel


class QSimulator(nn.Module):
    """
    Abstract class for q value simulators.
    """

    def __init__(self, physical_model: PhysicalModel = None):
        super().__init__()
        self.physical_model = physical_model

    def sample(self, batch_size: int, context: dict = None) -> Tensor:
        raise NotImplementedError

    def get_q_factor(self, q: Tensor) -> float:
        raise NotImplementedError

    def interpolate_data(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        raise NotImplementedError


class FixedQSimulator(QSimulator):
    """
    Returns fixed q values.
    """

    def __init__(self, q: Tensor, physical_model: PhysicalModel):
        super().__init__(physical_model)
        self.register_buffer("q", q)

    def sample(self, batch_size: int, context: dict = None) -> Tensor:
        return self.q.clone()[None].expand(batch_size, self.q.shape[0])

    def get_q_factor(self, q: Tensor) -> float:
        return q.max().item() / self.q.max().item()

    def interpolate_data(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        q, sigmas, curve = data["q"], data["sigmas"], data["data"]

        assert q.shape == curve.shape
        assert q.shape[0] == curve.shape[0] == 1

        if (q.shape[-1] == self.q.shape[-1]) and torch.allclose(
            q.squeeze(), self.q.squeeze()
        ):
            return data

        q_np = q.squeeze().cpu().numpy()
        data_np = curve.squeeze().cpu().numpy()

        assert q_np.shape == data_np.shape
        assert len(q_np.shape) == 1

        log_data = np.log10(data_np + 1e-10)
        log_sigmas = np.log10(sigmas.squeeze().cpu().numpy() + 1e-10)
        interp_log_data = np.interp(self.q.squeeze().cpu().numpy(), q_np, log_data)
        interp_log_sigmas = np.interp(self.q.squeeze().cpu().numpy(), q_np, log_sigmas)
        interp_data = 10**interp_log_data - 1e-10
        interp_sigmas = 10**interp_log_sigmas - 1e-10

        data["data"] = torch.from_numpy(interp_data).to(curve)[None]
        data["sigmas"] = torch.from_numpy(interp_sigmas).to(sigmas)[None]
        data["q"] = torch.atleast_2d(self.q)

        return data


class FixedEquidistantQSimulator(FixedQSimulator):
    """
    Returns fixed equidistant q values.
    """

    def __init__(
        self,
        q_range: tuple[float, float, int],
        remove_zero: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        physical_model: PhysicalModel = None,
    ):
        q = torch.linspace(*q_range, device=device, dtype=dtype)
        if remove_zero:
            q = q[1:]
        super().__init__(q=q, physical_model=physical_model)


class FixedEquidistantAngleQSimulator(FixedQSimulator):
    """
    Returns fixed q values corresponding to equidistant angles.
    """

    def __init__(
        self,
        angle_range: tuple[float, float, int] = (0.0, 0.2, 257),
        wavelength: float = 1.0,
        remove_zero: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        physical_model: PhysicalModel = None,
    ):
        q = (
            torch.from_numpy(angle_to_q(np.linspace(*angle_range), wavelength))
            .to(device)
            .to(dtype)
        )

        if remove_zero:
            q = q[1:]
        super().__init__(q=q, physical_model=physical_model)


class RandomQSimulator(QSimulator):
    def __init__(
        self,
        q_range: tuple[float, float],
        physical_model: PhysicalModel = None,
        max_q_num: int = 128,
        drop_range: tuple[float, float] = (0.0, 0.8),
        device="cpu",
        dtype: torch.dtype = torch.float32,
        q_factor_mode: str = "max",  # Literal['one', 'max']
    ):
        super().__init__(physical_model=physical_model)
        self.register_buffer(
            "q_range", torch.tensor(q_range, device=device, dtype=dtype)
        )
        self.max_q_num = max_q_num
        self.drop_range = drop_range
        self.q_factor_mode = q_factor_mode

    def sample(self, batch_size: int, context: dict = None) -> Tensor:
        q = torch.rand(
            batch_size,
            self.max_q_num,
            device=self.q_range.device,
            dtype=self.q_range.dtype,
        )
        q = q * (self.q_range[1] - self.q_range[0]) + self.q_range[0]

        if self.training:
            drop_mask = generate_drop_mask(
                batch_size, self.max_q_num, self.drop_range, device=self.q_range.device
            )
            context["drop_mask"] = drop_mask
        else:
            q = torch.sort(q, dim=-1).values
        return q

    def get_q_factor(self, q: Tensor) -> float:
        if self.q_factor_mode == "one":
            return 1.0
        elif self.q_factor_mode == "max":
            return q.max().item() / (self.q_range[1].item() - 1e-6)
        else:
            raise ValueError(f"Unknown q_factor_mode: {self.q_factor_mode}")

    def interpolate_data(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        q = data["q"].squeeze(0)
        indices_within_range = (q >= self.q_range[0]) & (q <= self.q_range[1])
        if not torch.all(indices_within_range):
            data = dict(data)
            for k in ("q", "sigmas", "data"):
                data[k] = data[k][:, indices_within_range]
        return data


class RandomEquidistantQSimulator(RandomQSimulator):
    def __init__(
        self,
        q_range: tuple[float, float],
        physical_model: PhysicalModel = None,
        max_q_num: int = 64,
        min_q_num: int = 20,
        drop_range: tuple[float, float] = (0.0, 0.3),
        device="cpu",
        dtype: torch.dtype = torch.float32,
        q_factor_mode: str = "max",  # Literal['one', 'max']
        replace_prob: float = 0.1,
        max_padding: float = 0.2,
        min_points_per_period: int = 4,
    ):
        super().__init__(
            q_range, physical_model, max_q_num, drop_range, device, dtype, q_factor_mode
        )
        self.replace_prob = replace_prob
        self.max_padding = max_padding
        self.min_points_per_period = min_points_per_period
        self.min_q_num = min_q_num

    def sample(self, batch_size: int, context: dict = None) -> Tensor:
        total_thicknesses = self.physical_model.get_total_thickness(context["theta"])
        ns_min = (
            total_thicknesses
            * (self.q_range[1] - self.q_range[0])
            / 2
            / np.pi
            * self.min_points_per_period
        )

        ns_min = torch.clamp(ns_min, min=self.min_q_num)

        # ns ~ U(ns_min, self.max_q_num)
        ns = torch.rand(
            batch_size, device=self.q_range.device, dtype=self.q_range.dtype
        )
        ns = ns * (self.max_q_num - ns_min) + ns_min

        qs, drop_mask = _generate_equidistant_q(
            batch_size, self.q_range, ns, self.max_q_num, self.max_padding
        )

        if self.drop_range and self.drop_range[1] > 0:

            drop_mask = drop_mask & generate_drop_mask(
                batch_size, self.max_q_num, self.drop_range, device=self.q_range.device
            )

        if self.replace_prob:
            random_qs = torch.rand(
                batch_size,
                self.max_q_num,
                device=self.q_range.device,
                dtype=self.q_range.dtype,
            )
            random_qs = (
                random_qs * (self.q_range[1] - self.q_range[0]) + self.q_range[0]
            )

            # random mask is a mask that is True with probability replace_prob, but taking into account drop_mask
            fraction_dropped = (~drop_mask).sum(-1).float() / self.max_q_num
            random_mask = torch.rand_like(random_qs) < self.replace_prob / (
                1 - fraction_dropped[:, None]
            )

            qs[random_mask] = random_qs[random_mask]

        if not self.training:
            qs = qs[drop_mask]
            qs = torch.sort(qs, dim=-1).values
        else:
            qs[~drop_mask] = 0
            context["drop_mask"] = drop_mask
        return qs


def _generate_equidistant_q(
    batch_size: int,
    q_range: Tensor,
    q_nums: Tensor,
    max_q_num: int,
    max_padding: float = 0.2,
) -> tuple[Tensor, Tensor]:
    # first q points q0 ~ U(q_min, (q_max - q_min) * max_padding + q_min)
    q0 = (
        torch.rand(batch_size, 1, device=q_range.device, dtype=q_range.dtype)
        * (q_range[1] - q_range[0])
        * max_padding
        + q_range[0]
    )

    dq = (q_range[1] - q0) / q_nums[:, None]

    qs = (
        torch.arange(max_q_num, device=q_range.device, dtype=q_range.dtype)[None] * dq
        + q0
    )

    drop_mask = torch.ones_like(qs, dtype=torch.bool)

    drop_mask[qs > q_range[1]] = False

    return qs, drop_mask


def shuffle(x: Tensor, dim: int = -1):
    """
    Shuffle the elements of x independently along specified dimension.
    """
    random_indices = torch.rand(x.size(), device=x.device).argsort(dim=dim)

    shuffled_x = torch.gather(x, dim, random_indices)

    return shuffled_x


def generate_drop_mask(
    batch_size: int, n: int, drop_range: tuple[float, float], device="cpu"
):
    """
    Generate a random mask for the indices to drop for each batch.
    """
    counts = torch.randint(
        int(round(drop_range[0] * n)),
        int(round(drop_range[1] * n)),
        (batch_size,),
        device=device,
    )

    mask = shuffle(torch.arange(n, device=device) > counts.unsqueeze(1))

    # assert torch.all(mask.sum(-1) == counts)

    return mask
