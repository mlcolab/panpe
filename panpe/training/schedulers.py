# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.optim import lr_scheduler

import numpy as np

from panpe.training.trainer import Trainer, TrainerCallback, PeriodicTrainerCallback

__all__ = [
    "ScheduleBatchSize",
    "ScheduleLR",
    "StepLR",
    "CyclicLR",
    "LogCyclicLR",
    "ReduceLrOnPlateau",
    "WarmupLR",
]


class ScheduleBatchSize(PeriodicTrainerCallback):
    def __init__(
        self, step: int, gamma: int = 2, last_epoch: int = -1, mode: str = "add"
    ):
        super().__init__(step, last_epoch)

        assert mode in ("add", "multiply")

        self.gamma = gamma
        self.mode = mode

    def _end_batch(self, trainer: Trainer, batch_num: int) -> None:
        if self.mode == "add":
            trainer.batch_size += self.gamma
        elif self.mode == "multiply":
            trainer.batch_size *= self.gamma


class ScheduleLR(TrainerCallback):
    def __init__(self, lr_scheduler_cls, **kwargs):
        self.lr_scheduler_cls = lr_scheduler_cls
        self.kwargs = kwargs
        self.lr_scheduler = None

    def start_training(self, trainer: Trainer) -> None:
        self.lr_scheduler = self.lr_scheduler_cls(trainer.optim, **self.kwargs)
        trainer.callback_params["lrs"] = []

    def end_batch(self, trainer: Trainer, batch_num: int) -> None:
        self.lr_scheduler.step()


class StepLR(ScheduleLR):
    def __init__(self, step_size: int, gamma: float, last_epoch: int = -1, **kwargs):
        super().__init__(
            lr_scheduler.StepLR,
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch,
            **kwargs,
        )

    def start_training(self, trainer: Trainer) -> None:
        trainer.optim.param_groups[0]["initial_lr"] = trainer.get_lr()
        super().start_training(trainer)


class CyclicLR(ScheduleLR):
    def __init__(
        self,
        base_lr,
        max_lr,
        step_size_up: int = 2000,
        cycle_momentum: bool = False,
        gamma: float = 1.0,
        mode: str = "triangular",
        **kwargs,
    ):
        super().__init__(
            lr_scheduler.CyclicLR,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            cycle_momentum=cycle_momentum,
            gamma=gamma,
            mode=mode,
            **kwargs,
        )


class LogCyclicLR(TrainerCallback):
    def __init__(
        self,
        base_lr,
        max_lr,
        period: int = 2000,
        gamma: float = None,
        log: bool = True,
        param_groups: tuple = (0,),
        start_period: int = 25,
    ):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.period = period
        self.gamma = gamma
        self.param_groups = param_groups
        self.log = log
        self.start_period = start_period
        self._axis = None
        self._period = None

    def get_lr(self, batch_num: int):
        return self._get_lr(batch_num)

    def _get_lr(self, batch_num):
        num_period, t = batch_num // self.period, batch_num % self.period

        if self._period != num_period:
            self._period = num_period
            if self.gamma and (num_period >= self.start_period):
                amp = (self.max_lr - self.base_lr) * (
                    self.gamma ** (num_period - self.start_period)
                )
                max_lr = self.base_lr + amp
            else:
                max_lr = self.max_lr

            if self.log:
                self._axis = np.logspace(
                    np.log10(self.base_lr), np.log10(max_lr), self.period // 2
                )
            else:
                self._axis = np.linspace(self.base_lr, max_lr, self.period // 2)
        if t < self.period // 2:
            lr = self._axis[t]
        else:
            lr = self._axis[self.period - t - 1]
        return lr

    def end_batch(self, trainer: Trainer, batch_num: int):
        lr = self.get_lr(batch_num)
        for param_group in self.param_groups:
            trainer.set_lr(lr, param_group)


class WarmupLR(TrainerCallback):
    def __init__(
        self,
        start_lr: float,
        end_lr: float,
        warmup_steps: int,
        param_groups: tuple = (0,),
        logscale: bool = True,
    ):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.warmup_steps = warmup_steps
        self.param_groups = param_groups
        self.logscale = logscale

    def end_batch(self, trainer: Trainer, batch_num: int) -> bool or None:
        if batch_num < self.warmup_steps:
            if self.logscale:
                lr = 10 ** (
                    np.log10(self.start_lr)
                    + (np.log10(self.end_lr) - np.log10(self.start_lr))
                    * batch_num
                    / self.warmup_steps
                )
            else:
                lr = (
                    self.start_lr
                    + (self.end_lr - self.start_lr) * batch_num / self.warmup_steps
                )
            for param_group in self.param_groups:
                trainer.set_lr(lr, param_group)


class ReduceLrOnPlateau(TrainerCallback):
    def __init__(
        self,
        factor: float = 0.5,
        patience: int = 100,
        loss_key: str = "total_loss",
        start: int = 0,
        min_lr: float = 1e-6,
        param_groups: tuple = (0,),
    ):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.start = start
        self.loss_key = loss_key
        self.param_groups = param_groups
        self._last_update = 0

    def end_batch(self, trainer: Trainer, batch_num: int) -> None:
        if batch_num < self.start:
            return
        if batch_num - self._last_update < self.patience:
            return

        loss = trainer.losses[self.loss_key]

        if len(loss) < self.patience:
            return

        previous_loss = np.mean(loss[-self.patience : -self.patience // 2])
        current_loss = np.mean(loss[-self.patience // 2 :])

        if current_loss > previous_loss:
            self._last_update = batch_num
            for param_group in self.param_groups:
                new_lr = max(self.min_lr, trainer.get_lr(param_group) * self.factor)
                trainer.set_lr(new_lr, param_group)
