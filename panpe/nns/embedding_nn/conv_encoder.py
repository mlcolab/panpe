# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from pathlib import Path

from torch import nn, load

from panpe.nns.utils import activation_by_name
from panpe.paths import SAVED_MODELS_DIR

__all__ = [
    "ConvEncoder",
]

logger = logging.getLogger(__name__)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        hidden_dims: tuple = (32, 64, 128, 256, 512),
        latent_dim: int = 64,
        avpool: int = 1,
        use_batch_norm: bool = True,
        activation: str = "lrelu",
        in_dim: int = 1,
    ):
        super().__init__()

        modules = []

        activation = activation_by_name(activation)

        for h_dim in hidden_dims:
            layers = [
                nn.Conv1d(
                    in_dim, out_channels=h_dim, kernel_size=3, stride=2, padding=1
                ),
                activation,
            ]
            if use_batch_norm:
                layers.insert(1, nn.BatchNorm1d(h_dim))
            modules.append(nn.Sequential(*layers))
            in_dim = h_dim

        self.core = nn.Sequential(*modules)
        self.avpool = nn.AdaptiveAvgPool1d(avpool)
        self.fc = nn.Linear(hidden_dims[-1] * avpool, latent_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.core(x)
        x = self.avpool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

    def load_weights(self, path: str or Path = None, strict: bool = False):
        if not path:
            return

        if isinstance(path, str):
            if not path.endswith(".pt"):
                path = path + ".pt"
            path = SAVED_MODELS_DIR / path

        if not path.is_file():
            logger.error(f"File {str(path)} is not found.")
            return
        try:
            state_dict = load(path)
            self.load_state_dict(state_dict, strict=strict)
        except Exception as err:
            logger.exception(err)
