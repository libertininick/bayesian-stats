# bayesian-stats
Sandbox repository for my journey learning Probabilistic Programming & Bayesian statistics

## PPL libraries
- [NumPyro](https://num.pyro.ai/en/stable/): NumPyro is a lightweight probabilistic programming library that provides a NumPy backend for Pyro.
- [ArviZ](https://python.arviz.org/en/stable/): ArviZ is a Python package for exploratory analysis of Bayesian models.

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
    ipympl \
    matplotlib \
    numpy \
    numpyro \
    pandas \
    python=3.11 \
    python-dotenv \
    python-graphviz \
    scipy

# Activate bayesian_stats env
conda activate bayesian_stats

# Install python package
cd ./bayesian-stats
python -m pip install -e .
```