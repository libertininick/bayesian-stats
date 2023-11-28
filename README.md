# bayesian-stats
This is a sandbox repository for documenting my journey learning probabilistic programming & Bayesian statistics.

I've chosen to build this library on top of [PyTorch](https://pytorch.org/) and [Pyro](https://pyro.ai/) given my experience
with torch from my deep learning adventures.


## Installation

To create a conda environment and install `bayesian-stats` with [poetry](https://python-poetry.org/docs/):

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

# Activate conda environment
conda activate bayesian_stats

# Install `bayesian-stats` w/ poetry
poetry install --with dev,jupyter
```

## Running Linting and Type Checking

To format code to adhere to our style and run type checking run the following:

```bash
ruff check . --fix
mypy src/
```

## Running Tests

To run tests and test coverage, run the following:

```bash
coverage erase \
    && coverage run -m pytest \
    && coverage report
```

To skip slow tests
```bash
coverage run -m pytest -m "not slow"
```

## Authors
- [@libertininick](https://github.com/libertininick)