# bayesian-stats
This is a sandbox repository for documenting my journey learning probabilistic programming & Bayesian statistics.

I've chosen to build this library on top of [PyTorch](https://pytorch.org/) and [Pyro](https://pyro.ai/) given my experience
with torch from my deep learning adventures.

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

# Activate conda environment
conda activate bayesian_stats

# Install `bayesian-stats` w/ poetry
poetry install --with dev,jupyter
```