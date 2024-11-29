# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

import numpy as np

from panpe.training.trainer import (
    TrainerCallback,
    Trainer,
)
from panpe.training.utils import is_divisor

__all__ = [
    "SaveBestModel",
    "LogLosses",
    "SaveIntermediateModels",
]


class SaveBestModel(TrainerCallback):
    def __init__(
        self, path: str, freq: int = 50, average: int = 10, save_twice: bool = False
    ):
        self.path = path
        self.average = average
        self._best_loss = np.inf
        self.freq = freq
        self.save_twice = save_twice

        if self.save_twice:
            self.path2 = path.split(".")[0] + "_copy.pt"

    def end_batch(self, trainer: Trainer, batch_num: int) -> None:
        if is_divisor(batch_num, self.freq):

            loss = np.mean(trainer.losses["total_loss"][-self.average :])

            if loss < self._best_loss:
                self._best_loss = loss
                self.save(trainer, batch_num)

    def save(self, trainer: Trainer, batch_num: int):
        prev_save = trainer.callback_params.pop("saved_iteration", 0)
        trainer.callback_params["saved_iteration"] = batch_num
        save_dict = {
            "model": trainer.model.state_dict(),
            "lrs": trainer.lrs,
            "losses": trainer.losses,
            "prev_save": prev_save,
            "batch_num": batch_num,
            "best_loss": self._best_loss,
        }
        torch.save(save_dict, self.path)
        if self.save_twice:
            torch.save(save_dict, self.path2)


class SaveIntermediateModels(TrainerCallback):
    def __init__(self, folder: str, name: str, num_iterations: tuple):
        self.folder = folder
        self.name = name
        self.num_iterations = num_iterations

    def _get_path(self, num: int):
        return f"{self.folder}/{self.name}_{num}.pt"

    def end_batch(self, trainer: Trainer, batch_num: int) -> None:
        if batch_num in self.num_iterations:
            self.save(trainer, batch_num)

    def save(self, trainer: Trainer, batch_num: int):
        save_dict = {
            "model": trainer.model.state_dict(),
            "lrs": trainer.lrs,
            "losses": trainer.losses,
            "batch_num": batch_num,
        }
        torch.save(save_dict, self._get_path(batch_num))


class LogLosses(TrainerCallback):
    def end_batch(self, trainer: Trainer, batch_num: int) -> None:
        try:
            trainer.log("train/total_loss", trainer.losses[trainer.TOTAL_LOSS_KEY][-1])
        except IndexError:
            pass
