# panpe-reflectometry
## Deep learning for Bayesian reflectometry analysis

Code for the manuscript 

**Fast and Reliable Probabilistic Reflectometry Inversion with Prior-Amortized Neural Posterior Estimation** 

by Vladimir Starostin<sup>a</sup>,
Maximilian Dax<sup>b</sup>,
Alexander Gerlach<sup>a</sup>, 
Alexander Hinderhofer<sup>a</sup>, 
Álvaro Tejero-Cantero<sup>a</sup>, 
and Frank Schreiber<sup>a</sup>.


*[a] University of Tübingen, 72076 Tübingen, Germany*  
*[b] Max Planck Institute for Intelligent Systems, 72076 Tübingen, Germany*

(under review)

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


### Training from scratch

The `panpe-2layers-xrr` model for XRR data is provided with the package. 
To train an additional model, run the following command:

```bash
python -m panpe.train <config_name>
```

where `<config_name>` is a name of a configuration file, such as `panpe-2layers-xrr`.
Configuration files are stored in the `configs` directory.

