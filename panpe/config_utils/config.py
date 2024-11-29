# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
import yaml
import warnings

from panpe.paths import CONFIG_DIR


def load_config(config_name: str or Path) -> dict:
    path = find_config_path(config_name)

    with open(path, "r") as f:
        config = yaml.safe_load(f)
    config["config_path"] = path
    return config


def find_config_path(config_name: str or Path) -> str:
    if isinstance(config_name, Path):
        return str(config_name.absolute())
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"

    path = CONFIG_DIR / config_name
    if path.is_file():
        return str(path.absolute())
    paths = list(CONFIG_DIR.rglob(config_name))
    if not paths:
        raise FileNotFoundError(
            f"Could not find config {config_name} in {str(CONFIG_DIR)}."
        )
    path = str(paths[0].absolute())

    if len(paths) > 1:
        warnings.warn(
            f"Found {len(paths)} configs with the same name {config_name}. "
            f"Use the first one: {path}."
        )

    return path
