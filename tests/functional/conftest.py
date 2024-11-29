import pytest
from panpe import InferenceModel, ExpDataset, ROOT_DIR
from panpe.config_utils.config import load_config


@pytest.fixture(scope="session")
def config():
    """Session-scoped fixture for the config."""
    return load_config("panpe-2layers-xrr")


@pytest.fixture(scope="session")
def inference_model(config):
    """Session-scoped fixture for the inference model."""
    return InferenceModel.from_config(config)


@pytest.fixture(scope="session")
def inference_kwargs(config):
    """Session-scoped fixture for the inference kwargs for testing purposes.
    It is not used in the actual inference pipeline, but it accelerates the testing process.
    """
    kwargs = dict(target_neff=500 if config["general"]["device"] == "cuda" else 100)
    return kwargs


@pytest.fixture(scope="session")
def exp_dataset(config):
    """Session-scoped fixture for the experimental dataset."""
    device = config["general"]["device"]
    return ExpDataset(ROOT_DIR / "data/xrr_data.h5", device=device)
