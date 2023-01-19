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

## Load data

```python
DATA_DIR = Path(os.environ.get("DATA_DIR"))
data = pd.read_csv(DATA_DIR.joinpath("urine.csv"))
missing = data.isnull().values.any(-1)
data = data.loc[~missing, :]
print(data.shape)
data.head()
```

## Normalize predictors

```python
x = data.loc[:,['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']]
means, stds = x.mean(0), x.std(0)
x_norm = (x - means) / stds
```

```python
means
```

## Logit Model

```python
logit_model = sm.Logit(data.r.values, x_norm)
results = logit_model.fit()
results.summary()
```

## Bayesian Logit Model

```python
# Laplace prior
x = np.linspace(-5,5,1000)
fig, ax = plt.subplots(figsize=(10,5))
for s in [0.25, 0.5, 1, 2, 4]:
    ax.plot(x, stats.laplace(loc=0, scale=1/s**0.5).pdf(x), label=s)
_ = ax.legend()
```

### Priors

```python
intercept_prior = dict(
    mu=0,
    sigma=1/25,
)
print(intercept_prior)

betas_prior_params = dict(
    mu=0,
    b=1 / 4**0.5,
)
print(betas_prior_params)
```

### Model specification

```python
with pm.Model() as logit_model:
    # Priors for unknown model parameters
    intercept = pm.Normal("intercept", **intercept_prior)
    betas = pm.Laplace("betas", **betas_prior_params, shape=6)

    # Expected value of outcome
    logit_p = intercept + x_norm.values @ betas

    # Likelihood (sampling distribution) of observations
    y = pm.Bernoulli("y", logit_p=logit_p, observed=data.r)
    
logit_model
```

```python
pm.model_to_graphviz(logit_model)
```

### Model fitting

```python
RANDOM_SEED = 58 
rng = np.random.default_rng(RANDOM_SEED)

# Number of chains
chains = 5

# Number of samples per chain
draws = 5000

with logit_model:
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

```python

```
