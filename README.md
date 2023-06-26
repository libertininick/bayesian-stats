# bayesian-stats
Sandbox repository for my journey learning Probabilistic Programming & Bayesian statistics

## PPL libraries
- [NumPyro](https://num.pyro.ai/en/stable/): NumPyro is a lightweight probabilistic programming library that provides a NumPy backend for Pyro.
- [ArviZ](https://python.arviz.org/en/stable/): ArviZ is a Python package for exploratory analysis of Bayesian models.

## Create conda environment & install
```bash
# Clone repository
git clone git@github.com:libertininick/bayesian-stats.git

# Navigate to local repo directory
cd bayesian-stats

# Update base conda environment
conda update -y -n base -c defaults conda

# Install conda-lock
# conda-lock is used to generate fully reproducible conda environments via a lock file
conda install --channel=conda-forge --name=base conda-lock

# Create conda environment from `conda-lock.yml`
conda-lock install --name bayesian_stats conda-lock.yml

# (optionally) for use with Jupyter install `ipykernel` and `ipympl`
conda install -c conda-forge -n bayesian_stats ipykernel ipympl

# Activate conda environment
conda activate bayesian_stats

# Install `bayesian-stats` w/ poetry
poetry install --with dev
```