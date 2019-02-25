# pce
This repository includes an implementation of the Polynomial Chaos Expansion method.

## Brief description
More comprehensive tools on the same subject are available (e.g. Chaospy), this repository is born because I (the author) need to implements stuff to really understand stuff.

At the moment, one can use this module to study the uncertainty propagation of a model with uncertain inputs. The following aspects are implemented:

* each uncertain variable can be associated to a uniform or normal distribution
* evaluation of the coefficient with spectral projection method
* global sensitivity analysis with Sobol' indices

## Requirements

The current version is developed using python 3.7.2 but it is not strictly necessary to use the very latest python version.

Other requirements (I tend to use always the latest version of the following libraries):

* numpy
* scipy
* matplotlib
* joblib
