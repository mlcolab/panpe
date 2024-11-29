# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings

from typing import Union, Optional

from pathlib import Path
from functools import wraps

import torch

from panpe import nns as nns_module
from panpe import simulator as simulator_module
from panpe import training as training_module
from panpe import inference as inference_module

from panpe.paths import ROOT_DIR
from panpe.utils import set_seed
from panpe.config_utils.config import load_config

__all__ = [
    "run_training_from_config",
    "get_paths_from_config",
    "init_callbacks_from_config",
    "init_flow_from_config",
    "init_simulator_from_config",
    "init_trainer_from_config",
    "init_inference_model_from_config",
]


def load_config_if_needed(func):
    """
    A decorator for functions that take a config as the first argument. If the first argument is a string, it is
    interpreted as a config name and the corresponding config is loaded.
    """

    @wraps(func)
    def wrapper(config: Union[dict, str], *args, **kwargs):
        if isinstance(config, str):
            config = load_config(config)
        return func(config, *args, **kwargs)

    return wrapper


@load_config_if_needed
def init_simulator_from_config(config: Union[dict, str]):
    """
    Initialize a simulator from a config.
    """
    sim_config = config["simulator"]
    num_layers = sim_config["num_layers"]
    hyperprior_cls = getattr(simulator_module, sim_config["hyperprior"]["cls"])
    hyperprior = hyperprior_cls.from_param_ranges(
        **sim_config["hyperprior"]["kwargs"], num_layers=num_layers
    )
    physical_model = _instance_from_conf(
        sim_config["physical_model"], module=simulator_module, num_layers=num_layers
    )
    q_simulator = _instance_from_conf(
        sim_config["q_simulator"],
        physical_model=physical_model,
        module=simulator_module,
    )
    measurement_noise = _instance_from_conf(
        sim_config["measurement_noise"], module=simulator_module
    )

    simulator = simulator_module.ReflectometrySimulator(
        physical_model=physical_model,
        hyperprior=hyperprior,
        q_simulator=q_simulator,
        measurement_noise=measurement_noise,
    )

    device = config["general"]["device"]
    simulator.to(device)

    return simulator


@load_config_if_needed
def init_flow_from_config(
    config: Union[dict, str],
    folder_paths: Optional[dict] = None,
    simulator: Optional[simulator_module.Simulator] = None,
):
    """
    Initialize a flow from a config.
    """
    if folder_paths is None:
        folder_paths = get_paths_from_config(config)
    if simulator is None:
        simulator = init_simulator_from_config(config)

    device = config["general"]["device"]

    nn_config = config["nn"]
    embedding_net_conf = nn_config["embedding_net"]

    data_scaler = nns_module.ScalerDict(
        data=nns_module.ScalarLogAffineScaler(),
        sigmas=nns_module.ScalarLogAffineScaler(),
        phi=simulator.hyperprior.get_phi_scaler(),
    )

    theta_dim = simulator.physical_model.theta_dim
    phi_dim = simulator.hyperprior.phi_dim

    embedding_net = _instance_from_conf(
        embedding_net_conf,
        module=nns_module,
        data_scaler=data_scaler,
        phi_dim=phi_dim,
    )
    embedding_net = _load_pretrained_model(
        embedding_net, 
        embedding_net_conf.get("pretrained_name", None), 
        folder_paths["saved_models"], 
        device
    )

    flow_config = nn_config["flow"]
    pretrained_name = flow_config.pop("pretrained_name", None)
    transform_config = flow_config.pop("transform_net")

    flow = nns_module.get_rq_nsf_c_flow(
        features=theta_dim,
        embedding_net=embedding_net,
        transform_net_fn=nns_module.get_residual_transform_net_fn(**transform_config),
        **flow_config,
    )

    model_weights_loaded: bool = False

    if pretrained_name:
        _load_pretrained_model(flow, pretrained_name, folder_paths["saved_models"], device)
        model_weights_loaded = True
    try:
        # try to load from the saved_models dir with the same name as the config
        _load_pretrained_model(
            flow, folder_paths["model"], folder_paths["saved_models"], device
        )
        model_weights_loaded = True
    except (FileNotFoundError, RuntimeError, KeyError) as err:
        warnings.warn(f"Could not load model weights: {err}")

    if not model_weights_loaded:
        warnings.warn("No model weights loaded. Using random initialization.")

    device = config["general"]["device"]
    flow.to(device)

    return flow


@load_config_if_needed
def init_trainer_from_config(config: Union[dict, str]) -> training_module.Trainer:
    """
    Initialize a trainer from a config.
    """
    folder_paths = get_paths_from_config(config, mkdir=True)

    simulator = init_simulator_from_config(config)
    flow = init_flow_from_config(config, folder_paths, simulator)

    train_conf = config["training"]

    optim_cls = getattr(torch.optim, train_conf["optimizer"])

    trainer_kwargs = train_conf.get("init_kwargs", {})

    trainer = training_module.Trainer(
        flow,
        simulator,
        train_conf["lr"],
        train_conf["batch_size"],
        optim_cls=optim_cls,
        **trainer_kwargs,
    )
    device = config["general"]["device"]
    trainer.to(device)

    return trainer


@load_config_if_needed
def init_inference_model_from_config(
    config: Union[dict, str],
) -> inference_module.InferenceModel:
    """
    Initialize an inference model from a config.
    """
    folder_paths = get_paths_from_config(config, mkdir=True)

    simulator = init_simulator_from_config(config)
    flow = init_flow_from_config(config, folder_paths, simulator)
    inference_model = inference_module.InferenceModel(simulator, flow)

    device = config["general"]["device"]
    inference_model.to(device)

    return inference_model


@load_config_if_needed
def run_training_from_config(config: Union[dict, str]):
    """
    Run training from a config.
    """

    if "seed" in config["general"]:
        set_seed(config["general"]["seed"])

    folder_paths = get_paths_from_config(config, mkdir=True)

    trainer = init_trainer_from_config(config)
    callbacks = init_callbacks_from_config(config)

    trainer.run_training(
        config["training"]["num_iterations"],
        callbacks,
        disable_tqdm=False,
        update_tqdm_freq=config["training"]["update_tqdm_freq"],
        grad_accumulation_steps=config["training"].get("grad_accumulation_steps", 1),
    )

    torch.save(
        {
            "paths": folder_paths,
            "losses": trainer.losses,
            "params": config,
        },
        folder_paths["losses"],
    )

    return trainer


def _instance_from_conf(conf, module, **kwargs):
    if not conf:
        return
    cls_name = conf["cls"]
    if not cls_name:
        return

    cls = getattr(module, cls_name, None)

    if not cls:
        raise ValueError(f"Unknown class {cls_name}")

    conf_args = conf.get("args", [])
    conf_kwargs = conf.get("kwargs", {})
    conf_kwargs.update(kwargs)
    return cls(*conf_args, **conf_kwargs)


def get_paths_from_config(config: dict, mkdir: bool = False) -> dict:
    """
    Get paths from a config.
    """
    root_dir = Path(config["general"]["root_dir"] or ROOT_DIR)
    name = config["general"]["name"]

    assert root_dir.is_dir()

    saved_models_dir = root_dir / "saved_models"
    saved_losses_dir = root_dir / "saved_losses"

    if mkdir:
        saved_models_dir.mkdir(exist_ok=True)
        saved_losses_dir.mkdir(exist_ok=True)

    model_path = str((saved_models_dir / f"model_{name}.pt").absolute())

    losses_path = saved_losses_dir / f"{name}_losses.pt"

    return {
        "name": name,
        "model": model_path,
        "losses": losses_path,
        "root": root_dir,
        "saved_models": saved_models_dir,
        "saved_losses": saved_losses_dir,
    }


def init_callbacks_from_config(
    config: dict, folder_paths: Optional[dict] = None
) -> tuple["TrainerCallback", ...]:
    """
    Initialize training callbacks from a config.
    """
    callbacks = []

    folder_paths = folder_paths or get_paths_from_config(config)

    train_conf = config["training"]
    callback_conf = dict(train_conf["callbacks"])

    save_conf = callback_conf.pop("save_best_model", None)
    save_intermediate_conf = callback_conf.pop("save_intermediate_models", None)

    if save_conf:
        save_model = training_module.SaveBestModel(
            folder_paths["model"], freq=save_conf["freq"]
        )
        callbacks.append(save_model)

    if save_intermediate_conf:
        callbacks.append(
            training_module.SaveIntermediateModels(
                folder=folder_paths["saved_models"],
                name=folder_paths["name"],
                num_iterations=tuple(save_intermediate_conf["num_iterations"]),
            )
        )

    for conf in callback_conf.values():
        callback = _instance_from_conf(conf, training_module)

        if callback:
            callbacks.append(callback)

    return tuple(callbacks)


def _load_pretrained_model(
    model, model_name: str, saved_models_dir: Path, device: str, verbose: bool = True
):
    if not model_name:
        return model
    if "." not in model_name:
        model_name = model_name + ".pt"
    model_path = saved_models_dir / model_name

    if not model_path.is_file():
        model_path = saved_models_dir / f"model_{model_name}"
    if not model_path.is_file():
        raise FileNotFoundError(f"File {str(model_path)} does not exist.")
    try:
        pretrained = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as err:
        raise RuntimeError(f"Could not load model from {str(model_path)}. Error: {err}") from err
    if "model" in pretrained:
        pretrained = pretrained["model"]
    try:
        model.load_state_dict(pretrained)
        if verbose:
            print(f"Loaded model {model_name}")
    except Exception as err:
        raise RuntimeError(
            f"Could not update state dict from {str(model_path)}. Error: {err}"
        ) from err

    return model
