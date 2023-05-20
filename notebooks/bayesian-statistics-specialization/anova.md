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

# Plant Growth data


## Load data

```python
DATA_DIR = Path(os.environ.get("DATA_DIR"))
data = pd.read_csv(DATA_DIR.joinpath("plant-growth.csv"))
missing = data.isnull().values.any(-1)
data = data.loc[~missing, :]
print(data.shape)
data.head()
```

```python
data.group.value_counts()
```

```python
# One-hot-encode group variables
levels = np.unique(data.group)
x = data.group.values.reshape(-1,1)[..., None] == levels.reshape(1,-1)[None]
x = x.all(1).astype(float)

x.shape
```

## Default Reference OLS Model

```python
ols_model = sm.OLS(data.weight.values, x)
results = ols_model.fit()
results.summary()
```

## Bayesian Cell Means Model


### Priors

```python
betas_prior_params = dict(
    mu=0,
    sigma=int(1e6**0.5),
)
print(betas_prior_params)

variance_prior_params = get_invgamma_params(variance_prior=1, effective_sample_size=5)
print(variance_prior_params)
```

### Model specification

```python
with pm.Model() as lm:
    # Priors for unknown model parameters
    betas = pm.Normal("betas", **betas_prior_params, shape=3)
    sigma = pm.InverseGamma("variance", **variance_prior_params)**0.5

    # Expected value of outcome
    mu = x @ betas

    # Likelihood (sampling distribution) of observations
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data.weight)
    
lm
```

```python
pm.model_to_graphviz(lm)
```

### Model fitting

```python
RANDOM_SEED = 58 
rng = np.random.default_rng(RANDOM_SEED)

# Number of chains
chains = 5

# Number of samples per chain
draws = 5000

with lm:
    # draw posterior samples
    trace_posterior = pm.sample(draws=draws, chains=chains)
    trace_posterior = pm.sample_posterior_predictive(trace_posterior, extend_inferencedata=True, random_seed=rng)
```

```python
az.summary(trace_posterior)
```

```python
get_highest_density_interval(
    trace_posterior.posterior.betas.values[...,0],
    confidence_level=0.9,
    axis=0
)
```

```python
az.plot_trace(trace_posterior, combined=True)
```

```python
az.plot_forest(trace_posterior, var_names=["betas"], combined=True, hdi_prob=0.95, r_hat=True)
```

```python

fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(trace_posterior.posterior_predictive.y[0, :, 0], bins=30)
_ = ax.axvline(x=data.log_infant[0], color="red")
```

## Posterior Analysis


### Posterior Mean Estimates

```python
pme = np.mean(trace_posterior.posterior.betas, (0,1)).values
pme
```

```python
y_hat = x @ pme
```

```python
residuals = data.weight - y_hat
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(y_hat, residuals)
```

### P(mean of treatment-2 is 10% larger than control mean)

```python
(trace_posterior.posterior.betas[..., 2] > 1.1*trace_posterior.posterior.betas[..., 0]).mean().values
```

### 95% interval of highest posterior density (HPD) for μ2 − μ0

```python
get_highest_density_interval(
    (trace_posterior.posterior.betas[..., 2] - trace_posterior.posterior.betas[..., 0]).values,
    confidence_level=0.95,
)
```

## Multi-variance model

```python
betas_prior_params = dict(
    mu=0,
    sigma=int(1e6**0.5),
)
print(betas_prior_params)

variance_prior_params = get_invgamma_params(variance_prior=1, effective_sample_size=5)
print(variance_prior_params)

with pm.Model() as mvm:
    # Priors for unknown model parameters
    betas = pm.Normal("betas", **betas_prior_params, shape=3)
    sigmas = pm.InverseGamma("variances", **variance_prior_params, shape=3)**0.5

    # Likelihood (sampling distribution) of observations
    for i in range(x.shape[1]):
        yi = pm.Normal(f"y{i}", mu=betas[i], sigma=sigmas[i], observed=data.weight[x[:, i] == 1])
    
mvm
```

```python
pm.model_to_graphviz(mvm)
```

```python
RANDOM_SEED = 58 
rng = np.random.default_rng(RANDOM_SEED)

# Number of chains
chains = 5

# Number of samples per chain
draws = 5000

with mvm:
    # draw posterior samples
    trace_posterior = pm.sample(draws=draws, chains=chains)
    trace_posterior = pm.sample_posterior_predictive(trace_posterior, extend_inferencedata=True, random_seed=rng)
```

```python
az.summary(trace_posterior)
```

```python

```
