# bayesian-stats
Sandbox repo for Bayesian statistics and modeling

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
    python=3.10 \
    pymc \
    python-dotenv \
    scipy

# Activate bayesian_stats env
conda activate bayesian_stats

# Install python package
cd ./bayesian-stats
python -m pip install -e .
```