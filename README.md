# bayesian-stats
Sandbox repo for Bayesian statistics and modeling

## External libraries
- [PyMC](https://www.pymc.io/welcome.html): PyMC is a probabilistic programming library for Python that allows users to build Bayesian models with a simple Python API and fit them using Markov chain Monte Carlo (MCMC) methods.
- [ArviZ](https://python.arviz.org/en/stable/): ArviZ is a Python package for exploratory analysis of Bayesian models.
- [NumPyro](https://num.pyro.ai/en/stable/): NumPyro is a lightweight probabilistic programming library that provides a NumPy backend for Pyro.
- [xarray](https://xarray.dev/): Xarray introduces labels in the form of dimensions, coordinates, and attributes on top of raw NumPy-like arrays.

## Create conda environment
```bash
# Update base environment
conda update -y -n base -c defaults conda

# Create bayesian_stats env
conda create -y -n bayesian_stats \
    -c conda-forge \
    arviz \
    black \
    ipykernel \
    matplotlib \
    numpy \
    numpyro \
    pandas \
    pymc \
    python=3.10 \
    python-dotenv \
    python-graphviz \
    scipy \
    statsmodels

# Activate bayesian_stats env
conda activate bayesian_stats

# Install python package
cd ./bayesian-stats
python -m pip install -e .
```