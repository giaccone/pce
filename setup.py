# this file is created according to the official documentation:
# https://packaging.python.org/tutorials/packaging-projects/
# ----------------------------------------------------------

import setuptools

# read the README
with open("README.md", 'r') as fh:
    long_description = fh.read()

# create setup
setuptools.setup(
    name = "pce",
    version = "0.1.1",
    author = "Luca Giaccone",
    author_email = "luca.giaccone@polito.it",
    description = "Polynomial Chaos Expansion method",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/giaccone/pce",
    packages=setuptools.find_packages(),
    install_requires=[
          'numpy',
          'scipy',
          'joblib',
          'matplotlib',
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
