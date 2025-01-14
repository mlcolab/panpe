# panpe-reflectometry
## Deep learning for Bayesian reflectometry analysis

Code for the manuscript 

**Fast and Reliable Probabilistic Reflectometry Inversion with Prior-Amortized Neural Posterior Estimation** 

by Vladimir Starostin<sup>1</sup>,
Maximilian Dax<sup>2, 3, 4</sup>,
Alexander Gerlach<sup>5</sup>, 
Alexander Hinderhofer<sup>5</sup>, 
Álvaro Tejero-Cantero<sup>1</sup>, 
and Frank Schreiber<sup>5</sup>.


- *[1] University of Tübingen, Tübingen, Germany*
- *[2] Max Planck Institute for Intelligent Systems Tübingen, Germany*
- *[3] ETH Zurich, Zurich, Switzerland*
- *[4] ELLIS Institute Tübingen, Tübingen, Germany*
- *[5] Institute of Applied Physics, University of Tübingen, Tübingen, Germany.*


Accepted for publication in Science Advances.

### Installation

To install the package, clone the repository and run the following command in the root directory:

```bash
pip install .
```

The following dependencies are required and will be installed automatically:

- numpy>=1.26.0
- scipy>=1.11.3
- torch>=2.1.0
- nflows>=0.14
- tqdm
- PyYAML
- click
- pytest
- matplotlib
- h5py

### Testing installation

To test the installation, run the following command in the root directory:

```bash
pytest
```

This will execute all tests, including doctests in the `panpe` package and tests in the `tests/functional` directory.


### Inference

To run inference, please refer to the [notebooks](notebooks) directory for examples of how to use the package for inference.

```python
# imports
from panpe import InferenceModel, ExpDataset, ROOT_DIR

# load the model
model = InferenceModel.from_config("panpe-2layers-xrr")

# load the experimental dataset (three datasets are concatenated together)
exp_dset = ExpDataset(ROOT_DIR / "data/xrr_data.h5", device="cpu")

# run inference
res = model(exp_dset[200])

# plot the results
res.plot_sampled_profiles(show_prior=True)
```


### Training from scratch

The `panpe-2layers-xrr` model for XRR data is provided with the package. 
To train an additional model, run the following command:

```bash
python -m panpe.train <config_name>
```

where `<config_name>` is a name of a configuration file, such as `panpe-2layers-xrr`.
Configuration files are stored in the `configs` directory.

