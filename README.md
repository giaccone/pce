# pce
This repository includes an implementation of the Polynomial Chaos Expansion method.

## Brief description
More comprehensive tools on the same subject are available (e.g. Chaospy), this repository is born during a self-learning activity of the authors.

At the moment, one can use this module to study the uncertainty propagation of a model with uncertain inputs. The following aspects are implemented:

* each uncertain variable can be associated to a uniform or normal distribution
* evaluation of the coefficient with spectral projection method
* global sensitivity analysis with Sobol' indices

## How can I use it? How can I cite this module?
If you use this module you can consider to cite the following paper [direct link](https://www.mdpi.com/2079-9292/9/9/1471).

*Giaccone, L.; Lazzeroni, P.; Repetto, M. Uncertainty Quantification in Energy Management Procedures. Electronics 2020, 9, 1471. https://doi.org/10.3390/electronics9091471*

In this paper the `pce` module has been used successfully to estimate uncertainties. You can also find all codes associated to the paper here [https://github.com/giaccone/cogen_eval](https://github.com/giaccone/cogen_eval).

## Requirements

The current version is tested with python 3.7.7 and 3.8.3 but it is not strictly necessary to use the very latest Python version.

Other requirements (I tend to use always the latest version of the following libraries):

* numpy
* scipy
* matplotlib
* joblib

# Installation

This project is deployed through the Python Package Index, therefore, it can be easily obtained bu running the following command:

```bash
pip install pce
```