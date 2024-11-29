# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import Tensor

from math import log, sqrt


class StreamingSNIS:
    """
    Streaming efficient self-normalized importance sampling.
    """

    def __init__(
        self,
        bins: tuple = None,
        calc_moments: bool = False,
    ):
        self.marginal_dists = (
            [torch.zeros_like(bin_ax[:-1]) for bin_ax in bins] if bins else None
        )

        self._calc_moments = calc_moments
        self.means = None
        self.vars = None

        self.bins = bins

        self._log_sum_w = None
        self._log_sum_w2 = None
        self._log_mean_p = None

        self.n = 0
        self.total_n = 0

    def update(self, log_p: Tensor, log_q: Tensor, params: Tensor = None) -> None:
        """
        Update the statistics with new samples.

        Args:
            log_p: log probability of the unnormalized target distribution. Tensor of shape (num_samples, )
            log_q: log probability of the proposal distribution. Tensor of shape (num_samples, )
            params: sampled parameters. Tensor of shape (num_samples, ndim), optional.
        """
        self.total_n += log_p.shape[0]

        finite_indices = torch.isfinite(log_p)

        if not finite_indices.any():
            return

        log_p, log_q = log_p[finite_indices], log_q[finite_indices]

        if params is not None:
            params = params[finite_indices]

        new_n = log_p.shape[0]

        log_weights = log_p - log_q

        r1, r2, new_log_sum_w = self._update_sum_w(log_weights)

        log_weights -= new_log_sum_w

        weights = torch.exp(log_weights)

        # already normalized:
        # assert abs(weights.sum().item() - 1.) < 1e-5

        self._update_log_p(log_p, log_weights, r1, r2)

        if params is None:
            self.n += new_n
            return

        if self._calc_moments:
            self._update_moments(params, weights, r1, r2)

        if self.bins is not None:
            self._update_marginal_dist(params, weights, r1, r2)

        self.n += new_n

    __call__ = update

    @property
    def acceptance_ratio(self):
        return self.n / self.total_n if self.total_n else 1.0

    @property
    def stds(self):
        if self.means is None:
            return
        return torch.sqrt(self.vars - self.means**2)

    @property
    def log_evidence(self) -> float:
        return self._log_sum_w.item() - log(self.n)

    @property
    def log_mean_p(self):
        return self._log_mean_p - self.log_evidence

    def get_uniform_eff_approx(self, volume: float = 1.0):
        return (-log(volume) - self.log_mean_p).exp().item()

    @property
    def scaled_uniform_eff(self):
        return self.get_uniform_eff_approx()

    @property
    def neff(self) -> int:
        if self._log_sum_w is None:
            return 0
        return torch.exp(2 * self._log_sum_w - self._log_sum_w2).item()

    @property
    def log_evidence_std(self):
        e = self.eff
        return sqrt((1 - e) / self.n / e)

    @property
    def eff(self):
        return self.neff / self.n

    def __repr__(self):
        if self.n == 0:
            return f"StreamingCalculations(n=0)"
        values = f"n={self.n}, eff={self.eff:.2e}, logZ={self._log_sum_w:.2f}+-{self.log_evidence_std:.2f}"
        return f"StreamingCalculations({values})"

    def _update_moments(self, params, weights, r1, r2):
        new_means = (params * weights[..., None]).sum(0)
        new_vars = (params**2 * weights[..., None]).sum(0)

        if self.means is None:
            self.means = new_means
            self.vars = new_vars
        else:
            self.means = self.means * r1 + new_means * r2
            self.vars = self.vars * r1 + new_vars * r2

    def _update_marginal_dist(self, params, weights, r1, r2):
        for param, bin_ax, marginal in zip(params.T, self.bins, self.marginal_dists):
            new_hist = _get_weighted_hist(param, weights, bin_ax)
            marginal[...] = marginal * r1 + new_hist * r2

    def _update_sum_w(self, log_weights: Tensor):
        new_log_sum_w = torch.logsumexp(log_weights, 0)
        new_log_sum_w2 = torch.logsumexp(2 * log_weights, 0)

        if self._log_sum_w is None:
            r1 = r2 = 1
            self._log_sum_w, self._log_sum_w2 = new_log_sum_w, new_log_sum_w2
        else:
            r1, r2 = _give_coefs(self._log_sum_w, new_log_sum_w)

            torch.logaddexp(self._log_sum_w, new_log_sum_w, out=self._log_sum_w)
            torch.logaddexp(self._log_sum_w2, new_log_sum_w2, out=self._log_sum_w2)

        return r1, r2, new_log_sum_w

    def _update_log_p(self, log_p, log_weights, r1, r2):
        new_log_p = torch.logsumexp(log_p + log_weights, 0).squeeze()

        if self._log_mean_p is None or r1 == 0:
            self._log_mean_p = new_log_p
        elif r2 != 0:

            self._log_mean_p = torch.logsumexp(
                torch.stack(
                    [self._log_mean_p.to(new_log_p) + log(r1), new_log_p + log(r2)]
                ),
                0,
            ).squeeze()


def _give_coefs(old, new):
    r = torch.exp(old - new).item()
    r1 = r / (1 + r)
    r2 = 1 / (1 + r)
    return r1, r2


def _get_weighted_hist(param: Tensor, weights: Tensor, bins: Tensor) -> Tensor:
    indices = (param >= bins[0]) & (param <= bins[-1])

    num_outsize = param.shape[0] - indices.sum().item()

    if num_outsize:
        param, weights = param[indices], weights[indices]

    p, indices = torch.sort(param.float())

    split_sections = torch.diff(torch.searchsorted(p, bins.to(p))).tolist()

    split_sections[0] += p.shape[0] - sum(
        split_sections
    )  # should check for corner cases

    hist = torch.stack(
        list(map(torch.sum, torch.split(weights[indices].float(), split_sections)))
    )

    return hist


def logsubexp(tensor, other):
    """
    Analog of torch.logaddexp(tensor, other) for subtraction
    """
    a = torch.max(tensor, other)
    return a + ((tensor - a).exp() - (other - a).exp()).log()
