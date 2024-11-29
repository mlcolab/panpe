# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

"""
Module with PyTorch implementation of MCMC samplers.
It is based on the emcee package (https://github.com/dfm/emcee) and is parallelized on GPU.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import torch
from torch import Tensor

from panpe.utils import trange


__all__ = [
    "MCMCSampler",
    "MCMCBackend",
    "MCMCState",
    "run_mcmc",
    "DEMove",
    "StretchMove",
]


class MCMCState:
    """A state of MCMC chains."""

    __slots__ = ("coords", "log_prob")

    def __init__(self, coords: Tensor, log_prob: Tensor):
        """Create a new state.

        Args:
               coords: The coordinates of the walkers. Tensor of shape (num_walkers, ndim).
               log_prob: The log probability of the walkers. Tensor of shape (num_walkers,).
        """
        self.coords = coords
        self.log_prob = log_prob

    @property
    def device(self):
        return self.coords.device

    @property
    def dtype(self):
        return self.coords.dtype

    def __repr__(self):
        return "State({0}, log_prob={1})".format(self.coords, self.log_prob)


class MCMCBackend:
    """
    A backend for storing states of MCMC chains.

    Args:
        num_walkers: The number of walkers in the chain.
        device: The device to store the states on.
        thin_by: The number of steps to thin the chain by.

    """

    def __init__(
        self, num_walkers: int, device: torch.device = "cpu", thin_by: int = 1
    ):
        self.coords = []
        self.log_probs = []
        self.num_walkers = num_walkers
        self.accepted = torch.zeros(num_walkers).to(device)
        self.device = device
        self.thin_by = thin_by
        self._iteration = 0

    @property
    def accepted_fractions(self) -> Tensor:
        return (
            self.accepted / self.iteration if self.iteration else self.accepted
        ).clone()

    @property
    def iteration(self) -> int:
        return self._iteration

    def save_state(self, state: MCMCState, accepted: Tensor):
        self.accepted += accepted.to(self.device)

        if self._iteration % self.thin_by == 0:
            self._save_state(state)

        self._iteration += 1

    def _save_state(self, state: MCMCState):
        self.coords.append(state.coords.to(self.device).clone())
        self.log_probs.append(state.log_prob.to(self.device).clone())

    def get_chain(self, flat: bool = False) -> Tensor:
        if flat:
            return torch.cat(self.coords, dim=0)
        else:
            return torch.stack(self.coords, dim=0)

    @property
    def chain_size(self):
        return sum(c.shape[0] for c in self.coords)

    def __repr__(self):
        return (
            f"Backend(iterations={self.iteration}, num_walkers={self.num_walkers}, chain_size={self.chain_size},"
            f"accepted_fractions={self.accepted_fractions})"
        )


class MCMCSampler:
    """
    A class for running MCMC chains.

    Args:
        ndim: The number of dimensions of the target distribution.
        num_walkers: The number of walkers in the chain.
        log_prob_fn: A vectorized function that computes log probabilities of the target distribution.
        backend: A backend for storing the states of the chain.
        thin_by: The number of steps to thin the chain by.
    """

    def __init__(
        self,
        ndim: int,
        num_walkers: int,
        log_prob_fn,
        backend: MCMCBackend = None,
        thin_by: int = 1,
    ):
        self.ndim, self.num_walkers = ndim, num_walkers
        self.backend = backend or MCMCBackend(self.num_walkers, thin_by=thin_by)
        self.log_prob_fn = log_prob_fn

    def run(
        self, init_coords: Tensor, num_steps: int, *, burn_in: int = 0, **kwargs
    ) -> MCMCState:
        """
        Run MCMC.

        Args:
            init_coords: The initial coordinates of the walkers. Tensor of shape (num_walkers, ndim).
            num_steps: Number of steps to run the chain for.
            burn_in: Number of steps to discard at the beginning of the chain.

        Returns:
            The final state of the chains.

        """

        backend, state = run_mcmc(
            init_coords,
            self.log_prob_fn,
            num_steps,
            backend=self.backend,
            burn_in=burn_in,
            **kwargs,
        )
        return state

    def get_chain(self):
        return self.backend.get_chain()

    @property
    def accepted_fractions(self) -> Tensor:
        return self.backend.accepted_fractions

    def __repr__(self):
        return f"MCMCSampler(backend={repr(self.backend)})"


class MCMCMove:
    """
    A class for proposing new states of the MCMC chain.
    """

    def step(self, state: MCMCState, log_prob_fun) -> tuple[MCMCState, Tensor]:
        """
        Propose a new state of the chain.

        Args:
            state: The current state of the chain.
            log_prob_fun: The vectorized log probability function of the target distribution.

        Returns:
            A tuple of the new state and a boolean tensor indicating whether the proposal was accepted.

        """
        raise NotImplementedError


def run_mcmc(
    init_coords: Tensor,
    log_prob_fn,
    num_steps: int,
    backend: MCMCBackend = None,
    *,
    moves: MCMCMove or tuple[tuple[MCMCMove, float], ...] = None,
    burn_in: int = 0,
    thin_by: int = 1,
    disable_tqdm: bool = False,
) -> tuple[MCMCBackend, MCMCState]:
    """
    Run MCMC chains.

    Args:
        init_coords: The initial coordinates of the walkers. Tensor of shape (num_walkers, ndim).
        log_prob_fn: The log probability function of the target distribution.
        num_steps: The number of steps to run the chain for.
        backend: The backend to store the states of the chain.
        moves: A tuple of tuples containing MCMC moves and their probabilities.
         If None, the default is a single DEMove with weight 1.
        burn_in: The number of steps to discard at the beginning of the chain.
        thin_by: The number of steps to thin the chain by. The default is 1. Ignored if backend is not None.
        disable_tqdm: Whether to disable the progress bar.

    Returns:
        A tuple of the backend and the final state of the chain.

    """

    if moves is None:
        moves, weights = (DEMove(),), (1.0,)
    elif isinstance(moves, MCMCMove):
        moves, weights = (moves,), (1.0,)
    else:
        moves, weights = zip(*moves)

    weights = np.array(weights) / np.sum(weights)

    state = MCMCState(init_coords, log_prob_fn(init_coords))

    num_walkers, ndim = init_coords.shape

    backend = backend or MCMCBackend(num_walkers, thin_by=thin_by)

    pbar = trange(num_steps, disable=disable_tqdm)

    for step_idx in pbar:
        move = np.random.choice(moves, p=weights)
        state, accepted = move.step(state, log_prob_fn)

        if step_idx >= burn_in:
            backend.save_state(state, accepted)

    return backend, state


class RedBlueMove(MCMCMove):
    """
    A class for red-blue MCMC moves based on emcee package.
    """

    def step(self, state: MCMCState, log_prob_fun) -> tuple[MCMCState, Tensor]:
        """
        Propose a new state of the population.
        Args:
            state: The current state of the population. The state is modified in place.
            log_prob_fun: The vectorized log probability function of the target distribution.

        Returns:
            A tuple of the updated state and a boolean tensor indicating whether the proposal was accepted.

        """
        nwalkers, ndim = state.coords.shape
        device, dtype = state.device, state.dtype

        accepted = torch.zeros((nwalkers,), dtype=torch.bool, device=device)

        all_indices = torch.arange(nwalkers, device=device)

        split_num_indices = shuffle(all_indices % 2)

        for split_num in range(2):
            updated_indices = split_num_indices == split_num
            updated_walkers, source_walkers = (
                state.coords[updated_indices],
                state.coords[~updated_indices],
            )
            proposed_coords, factors = self._get_proposal(
                updated_walkers, source_walkers
            )
            new_log_probs = log_prob_fun(proposed_coords)

            sampled_rands = torch.log(
                torch.rand(factors.shape[0], device=device, dtype=dtype)
            )

            lnpdiff = (
                factors + new_log_probs - state.log_prob[all_indices[updated_indices]]
            )

            accepted[updated_indices] = lnpdiff > sampled_rands

            new_state = MCMCState(proposed_coords, log_prob=new_log_probs)
            state = _update_state(state, new_state, accepted, updated_indices)

        return state, accepted

    def _get_proposal(self, updated_walkers: Tensor, source_walkers: Tensor):
        raise NotImplementedError


class StretchMove(RedBlueMove):
    """
    A stretch move.
    """

    def __init__(self, a: float = 2.0):
        self.a = a

    def _get_proposal(self, s, c):
        return stretch_move(s, c, self.a)


class DEMove(RedBlueMove):
    """
    A differential evolution MCMC move.
    """

    def __init__(self, sigma: float = 1e-5, g0: float = None):
        self.sigma = sigma
        self.g0 = g0

    def _get_proposal(self, s, c):
        return de_move(s, c, self.sigma, self.g0)


def _update_state(old_state, new_state, accepted, subset):
    m1 = subset & accepted
    m2 = accepted[subset]
    old_state.coords[m1] = new_state.coords[m2]
    old_state.log_prob[m1] = new_state.log_prob[m2]
    return old_state


def stretch_move(s: Tensor, c: Tensor, a: float):
    """
    A stretch move.
    """

    ns, nc, ndim = s.shape[0], c.shape[0], c.shape[1]

    zz = ((a - 1.0) * torch.rand(ns, device=s.device, dtype=s.dtype) + 1) ** 2.0 / a
    factors = (ndim - 1.0) * torch.log(zz)
    rint = torch.randint(nc, size=(ns,), device=s.device)

    return c[rint] - (c[rint] - s) * zz[:, None], factors


def de_move(
    updated_walkers: Tensor,
    source_walkers: Tensor,
    sigma: float = 1e-5,
    g0: float = None,
):
    """
    Differential evolution MCMC move.
    """

    u_num, ndim = updated_walkers.shape
    s_num = source_walkers.shape[0]

    if g0 is None:
        g0 = 2.38 / sqrt(2 * ndim)

    # sample pairs of walkers from the c population that exclude pairs of same walkers

    # Get the lower triangle indices
    rows, cols = torch.tril_indices(s_num, s_num, -1, device=updated_walkers.device)

    # Combine rows-cols and cols-rows pairs
    pairs = torch.stack([torch.cat([rows, cols]), torch.cat([cols, rows])], dim=1)

    # Sample from the pairs
    indices = torch.randint(0, s_num * (s_num - 1), (u_num,), dtype=torch.long)
    pairs = pairs[indices]

    # Get the differences between the sampled pairs of source walkers
    diffs = torch.diff(source_walkers[pairs], dim=1).squeeze(dim=1)  # (ns, ndim)

    # Sample a gamma value for each walker following Nelson et al. (2013) https://doi.org/10.1088/0067-0049/210/1/11
    gamma = g0 * (1 + sigma * torch.randn(u_num, 1).to(updated_walkers))  # (ns, 1)

    q = updated_walkers + gamma * diffs

    return q, torch.zeros_like(updated_walkers[..., 0])


def shuffle(t: Tensor) -> Tensor:
    """
    Shuffle a tensor along the first dimension.
    Args:
        t: tensor to shuffle.

    Returns:
        The shuffled tensor.
    """
    idx = torch.randperm(t.shape[0], device=t.device)
    return t[idx].view(t.size())
