# 1. pce
This repository includes an implementation of the Polynomial Chaos Expansion method.

## 1.1 Brief description
More comprehensive tools on the same subject are available (e.g. Chaospy), this repository is born during a self-learning activity of the authors.

At the moment, one can use this module to study the uncertainty propagation of a model with uncertain inputs. The following aspects are implemented:

* each uncertain variable can be associated to a uniform or normal distribution
* evaluation of the coefficient with spectral projection method
* global sensitivity analysis with Sobol' indices

## 1.2 How can I use it? How can I cite this module?
If you use this module you can consider to cite the following papers:

[1] Luca Giaccone, *"Uncertainty quantification in the assessment of human exposure to pulsed or multi-frequency fields"*, Physics in Medicine & Biology, [10.1088/1361-6560/acc924](https://doi.org/10.1088/1361-6560/acc924)

[2] Giaccone, L.; Lazzeroni, P.; Repetto, M. Uncertainty Quantification in Energy Management Procedures. Electronics 2020, 9, 1471. [https://doi.org/10.3390/electronics9091471](https://doi.org/10.3390/electronics9091471)

In [1] the `pce` is used to estimate the uncertainty associated to methods for the assessment of pulsed magnetic or electric fields. You can also find all codes associated to the paper here [https://github.com/giaccone/wpm_uncertainty](https://github.com/giaccone/wpm_uncertainty). In [2] the `pce` module has been used successfully to estimate uncertainties. You can also find all codes associated to the paper here [https://github.com/giaccone/cogen_eval](https://github.com/giaccone/cogen_eval).

## 1.3 Requirements

The project is developed using Python 3. The installer requires a Python version `>= 3.6`.

Other requirements (I tend to use always the latest version of the following libraries):

* numpy
* scipy
* matplotlib
* joblib

# 2. Installation

This project is deployed through the Python Package Index, therefore, it can be easily obtained by running the following command:

```bash
pip install pce
```