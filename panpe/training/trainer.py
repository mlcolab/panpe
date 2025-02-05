# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Iterable, Any
from collections import defaultdict

import numpy as np

import torch
from torch import Tensor
from torch.nn import Module

from panpe.utils import trange
from panpe.training.loggers import Logger, Loggers
from panpe.simulator import Simulator
from panpe.training.utils import is_divisor

__all__ = [
    "Trainer",
    "TrainerCallback",
    "PeriodicTrainerCallback",
]


class Trainer(Module):
    TOTAL_LOSS_KEY: str = "total_loss"

    def __init__(
        self,
        model: Module,
        simulator: Simulator,
        lr: float,
        batch_size: int,
        logger: Logger or tuple[Logger, ...] or Loggers = None,
        optim_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        optim_kwargs: dict = None,
        **kwargs,
    ):

        super().__init__()

        self.model = model
        self.simulator = simulator
        self.batch_size = batch_size

        self.optim = self.configure_optimizer(optim_cls, lr=lr, **(optim_kwargs or {}))
        self.losses = defaultdict(list)
        self.logger = _init_logger(logger)

        self.callback_params = {}
        self.lrs = []

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.init()

    def init(self):
        pass

    def log(self, name: str, data):
        self.logger.log(name, data)

    def run_training(
        self,
        num_batches: int,
        callbacks: tuple["TrainerCallback", ...] or "TrainerCallback" = (),
        disable_tqdm: bool = False,
        update_tqdm_freq: int = 10,
        grad_accumulation_steps: int = 1,
    ):

        if not grad_accumulation_steps:
            return

        if isinstance(callbacks, TrainerCallback):
            callbacks = (callbacks,)

        callbacks = _StackedTrainerCallbacks(list(callbacks))

        pbar = trange(num_batches, disable=disable_tqdm)

        callbacks.start_training(self)

        for batch_num in pbar:
            self.model.train()
            self.optim.zero_grad()
            total_loss, avr_loss_dict = 0, defaultdict(list)

            for _ in range(grad_accumulation_steps):

                training_batch = self.simulate_training_batch()
                loss_dict = self.get_loss_dict(training_batch)
                loss: Tensor = sum(loss_dict.values()) / grad_accumulation_steps
                total_loss += loss.item()
                _update_loss_dict(avr_loss_dict, loss_dict)

                if not torch.isfinite(loss).item():
                    raise ValueError("Loss is not finite!")

                loss.backward()

            self.optim.step()
            avr_loss_dict = {k: np.mean(v) for k, v in avr_loss_dict.items()}
            self._update_losses(avr_loss_dict, total_loss)

            if not disable_tqdm:
                self._update_tqdm(pbar, batch_num, update_tqdm_freq)

            break_epoch = callbacks.end_batch(self, batch_num)

            if break_epoch:
                break

        callbacks.end_training(self)

    def _update_tqdm(self, pbar, batch_num: int, update_tqdm_freq: int):
        if is_divisor(batch_num, update_tqdm_freq):
            last_loss = np.mean(self.losses[self.TOTAL_LOSS_KEY][-10:])
            pbar.set_description(f"Loss = {last_loss:.2e}")

    def simulate_training_batch(self) -> tuple[Tensor, dict[str, Tensor]]:
        batch_data = self.simulator.sample(self.batch_size)
        return self._process_simulated_data(batch_data)

    def _process_simulated_data(
        self, batch_data: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        reparameterized_theta = batch_data.pop("reparameterized_theta")

        return reparameterized_theta, batch_data

    def get_loss_dict(self, processed_batch_data) -> dict[str, Tensor]:
        reparameterized_theta, context = processed_batch_data

        loss = -self.model.log_prob(
            inputs=reparameterized_theta, context=context
        ).mean()

        return {"loss": loss}

    def _update_losses(self, loss_dict: dict, loss: float) -> None:
        _update_loss_dict(self.losses, loss_dict)
        self.losses[self.TOTAL_LOSS_KEY].append(loss)
        self.lrs.append(self.get_lr())

    def configure_optimizer(
        self, optim_cls, lr: float, **kwargs
    ) -> torch.optim.Optimizer:
        optim = optim_cls(self.model.parameters(), lr, **kwargs)
        return optim

    @property
    def lr(self) -> float:
        return self.get_lr()

    def get_lr(self, param_group: int = 0) -> float:
        return self.optim.param_groups[param_group]["lr"]

    def set_lr(self, lr: float, param_group: int = 0) -> None:
        self.optim.param_groups[param_group]["lr"] = lr


class TrainerCallback:
    def start_training(self, trainer: Trainer) -> None:
        pass

    def end_training(self, trainer: Trainer) -> None:
        pass

    def end_batch(self, trainer: Trainer, batch_num: int) -> bool or None:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class PeriodicTrainerCallback(TrainerCallback):
    def __init__(self, step: int = 1, last_epoch: int = -1):
        self.step = step
        self.last_epoch = last_epoch

    def end_batch(self, trainer: Trainer, batch_num: int) -> bool or None:
        if is_divisor(batch_num, self.step) and (
            self.last_epoch == -1 or batch_num < self.last_epoch
        ):
            return self._end_batch(trainer, batch_num)

    def _end_batch(self, trainer: Trainer, batch_num: int) -> bool or None:
        pass


class _StackedTrainerCallbacks(TrainerCallback):
    def __init__(self, callbacks: Iterable[TrainerCallback]):
        self.callbacks = tuple(callbacks)

    def start_training(self, trainer: Trainer) -> None:
        for c in self.callbacks:
            c.start_training(trainer)

    def end_training(self, trainer: Trainer) -> None:
        for c in self.callbacks:
            c.end_training(trainer)

    def end_batch(self, trainer: Trainer, batch_num: int) -> bool or None:
        break_epoch = False
        for c in self.callbacks:
            break_epoch += bool(c.end_batch(trainer, batch_num))
        return break_epoch

    def __repr__(self):
        callbacks = ", ".join(repr(c) for c in self.callbacks)
        return f"StackedTrainerCallbacks({callbacks})"


def _init_logger(logger: Logger or tuple[Logger, ...] or Loggers = None):
    if not logger:
        return Logger()
    if isinstance(logger, Logger):
        return logger
    return Loggers(*logger)


def _update_loss_dict(loss_dict: dict, new_values: dict):
    for k, v in new_values.items():
        loss_dict[k].append(v.item())
