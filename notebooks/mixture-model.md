---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python [conda env:bayesian_stats]
    language: python
    name: conda-env-bayesian_stats-py
---

```python
%load_ext autoreload
%autoreload 2
import os
from pathlib import Path
from typing import Tuple, Union

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import statsmodels.api as sm
from dotenv import load_dotenv
from numpy import ndarray
from scipy import stats

from bayesian_stats import (
    get_auto_corr,
    get_effective_sample_size,
    get_gelman_rubin_diagnostic,
    get_highest_density_interval,
    get_invgamma_params,
    one_hot_encode,
)

# Load environment variables from .env
load_dotenv()

# Set plotting preferences
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
```

## Load data

```python
DATA_DIR = Path(os.environ.get("DATA_DIR"))
data = pd.read_csv(DATA_DIR.joinpath("mixture.csv"), header=None, names=["y"])
missing = data.isnull().values.any(-1)
data = data.loc[~missing, :]
data.shape
```

```python
fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(data, bins=30)
```

## Hierarchical Mixture Model

```python
help(pm.Mixture)
```

### Model specification

```python
with pm.Model() as mm_model:
    # Priors for unknown means of mixture distributions
    mu_a = pm.Normal("mu_a", mu=-1, sigma=10)
    mu_b = pm.Normal("mu_b", mu=1, sigma=10)
    
    # Prior for unknown common std of mixture distributions
    sigma = pm.InverseGamma("sigma", **get_invgamma_params(variance_prior=1**2, effective_sample_size=1))**0.5
    
    # Component distributions
    components = [
        pm.Normal.dist(mu=mu_a, sigma=sigma),
        pm.Normal.dist(mu=mu_b, sigma=sigma),
    ]
    
    # Prior for mixture wts
    wts = pm.Dirichlet("wts", a=np.array([1.0, 1.0]))
    
    # Likelihood of latent mixture class
    like = pm.Mixture('like', w=wts, comp_dists=components, observed=data.y)
    
mm_model
```

```python
pm.model_to_graphviz(mm_model)
```

### Model fitting

```python
RANDOM_SEED = 58 
rng = np.random.default_rng(RANDOM_SEED)

# Number of chains
chains = 5

# Number of samples per chain
draws = 5000

with mm_model:
    # draw posterior samples
    trace_mm_model = pm.sample(draws=draws, chains=chains)
    
    # Add posterior predictive distribution
    trace_mm_model = pm.sample_posterior_predictive(trace_mm_model, extend_inferencedata=True, random_seed=rng)
```

```python
az.summary(trace_mm_model)
```

```python
az.plot_trace(trace_mm_model, combined=True)
```

```python
data.y
```

```python
fig, ax = plt.subplots(figsize=(10,5))
_, bins, _ = ax.hist(
    np.array(trace_mm_model.posterior_predictive.like[...,0]).reshape(-1),
    bins=100,
    alpha=0.5
)
_ = ax.hist(
    np.array(trace_mm_model.posterior_predictive.like[...,197]).reshape(-1),
    bins=bins,
    alpha=0.5
)
```

```python

```
