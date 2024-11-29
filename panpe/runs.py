# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import click

from panpe.config_utils import run_training_from_config


@click.command()
@click.argument("config_name", type=str)
def run_train(config_name: str):
    run_training_from_config(config_name)
