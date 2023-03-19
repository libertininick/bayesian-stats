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

print(pm.__version__)
```

## Make data

```python
mu = np.array([1, -1])
cov = np.array(
    [[1.0, 0.5],
     [0.5, 2.0],
    ]
)
n_samples = 300
data = np.random.multivariate_normal(mu, cov, n_samples)
data.shape
```

```python
fig, ax = plt.subplots(figsize=(5,5))
_ = ax.scatter(*data.T)
```

```python
np.corrcoef(data.T)
```

## Multivariate Normal Model


### Model specification

```python
with pm.Model() as model:
    # Priors for unknown means of multivariate
    mu = pm.Normal("mu", mu=0, sigma=10, shape=2)

    # Priors for covariance
    sd_dist = pm.Exponential.dist(1.0, shape=2)
    chol, corr, stds = pm.LKJCholeskyCov(
        'chol_cov', 
        n=2, 
        eta=1,
        sd_dist=sd_dist,
        compute_corr=True
    )

    # Likelihood of latent mixture class
    like = pm.MvNormal('like', mu=mu, chol=chol, observed=data)
    
model
```

```python
pm.model_to_graphviz(model)
```

### Model fitting

```python
RANDOM_SEED = 58 
rng = np.random.default_rng(RANDOM_SEED)

# Number of chains
chains = 5

# Number of samples per chain
draws = 5000

with model:
    # draw posterior samples
    trace = pm.sample(draws=draws, chains=chains)
```

```python
az.summary(trace)
```

```python

```

```python

```
