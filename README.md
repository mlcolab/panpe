
# panpe
## Deep learning for Bayesian reflectometry analysis

Code for the manuscript **"Fast and Reliable Probabilistic Reflectometry Inversion with Prior-Amortized Neural Posterior Estimation"** by
Vladimir Starostin<sup>a</sup>,
Maximilian Dax<sup>b</sup>,
Alexander Gerlach<sup>a</sup>, 
Alexander Hinderhofer<sup>a</sup>, 
Álvaro Tejero-Cantero<sup>a</sup>, 
and Frank Schreiber<sup>a</sup> (submitted).


*[a] University of Tübingen, 72076 Tübingen, Germany*  
*[b] Max Planck Institute for Intelligent Systems, 72076 Tübingen, Germany*

### Installation

To install the package, clone the repository and run the following command in the root directory:

```bash
pip install .
```

Dependencies are listed in the `requirements.txt` file.

### Training

The `panpe-2layers-xrr` model for XRR data is provided with the package. 
To train an additional model, run the following command:

```bash
python -m panpe.train <config_name>
```

where `<config_name>` is a name of a configuration file, such as `panpe-2layers-xrr`.
Configuration files are stored in the `configs` directory.

