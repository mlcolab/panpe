# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

def get_version():
    version_info = (0, 1, 0)
    return ".".join(map(str, version_info))

def get_description():
    return "Prior-Amortized Neural Posterior Estimation for Bayesian Reflectometry Analysis"

def get_classifiers():
    return [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ]

__version__ = get_version()
description = get_description()
classifiers = get_classifiers()

__license__ = "MIT"
__author__ = "Vladimir Starostin"
__email__ = "vladimir.starostin@uni-tuebingen.de"
