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
data = pd.read_csv(DATA_DIR.joinpath("bad-health.csv"))
missing = data.isnull().values.any(-1)
data = data.loc[~missing, :]
print(data.shape)
data.head()
```

## Priors for model coefficients

```python
noninformative_prior_params = dict(
    mu=0,
    sigma=1e3,
)
skeptical_prior_params = dict(
    mu=0,
    sigma=0.2
)

```

```python
fig, ax = plt.subplots(figsize=(10,5))
x = np.linspace(-5, 5, 1000)
noninformative_prior_dist = stats.norm(loc=noninformative_prior_params["mu"], scale=noninformative_prior_params["sigma"])
skeptical_prior_dist = stats.norm(loc=skeptical_prior_params["mu"], scale=skeptical_prior_params["sigma"])

_ = ax.plot(x, noninformative_prior_dist.pdf(x), label="noninformative")
_ = ax.plot(x, skeptical_prior_dist.pdf(x), label="skeptical")
```

## Model specification


### noninformative

```python
with pm.Model() as ninfo_model:
    # Priors for unknown model parameters
    betas = pm.Normal("betas", **noninformative_prior_params, shape=4)

    # Expected value of outcome
    log_lambda = (
        betas[0] 
        + data.badh.values * betas[1] 
        + data.age.values * betas[2] 
        + data.badh.values * data.age.values * betas[3]
    )

    # Likelihood (sampling distribution) of observations
    y = pm.Poisson("y", mu=np.exp(log_lambda), observed=data.numvisit.values)
    
ninfo_model
```

### skeptical

```python
with pm.Model() as skeptical_model:
    # Priors for unknown model parameters
    betas = pm.Normal("betas", **skeptical_prior_params, shape=4)

    # Expected value of outcome
    log_lambda = (
        betas[0] 
        + data.badh.values * betas[1] 
        + data.age.values * betas[2] 
        + data.badh.values * data.age.values * betas[3]
    )

    # Likelihood (sampling distribution) of observations
    y = pm.Poisson("y", mu=np.exp(log_lambda), observed=data.numvisit.values)
    
skeptical_model
```

## Model fitting

```python
RANDOM_SEED = 58 
rng = np.random.default_rng(RANDOM_SEED)

# Number of chains
chains = 5

# Number of samples per chain
draws = 5000

with ninfo_model:
    # draw posterior samples
    trace_ninfo = pm.sample(draws=draws, chains=chains)
    
with skeptical_model:
    # draw posterior samples
    trace_skeptical = pm.sample(draws=draws, chains=chains)
```

```python
az.summary(trace_ninfo).set_axis(["intercept", "bad health", "age", "age * bad health"])
```

```python
az.summary(trace_skeptical).set_axis(["intercept", "bad health", "age", "age * bad health"])
```

## Posterior density of parameters

Under the skeptical prior, our posterior distribution for b_badh has significantly dropped to between about 0.6 and 1.1. Although the strong prior influenced our inference on the magnitude of the bad health effect on visits, it did not change the fact that the coefficient is significantly above 0. In other words: even under the skeptical prior, bad health is associated with more visits, with posterior probability near 1.


### Bad health coeff

```python
ninfo_kde = sm.nonparametric.KDEUnivariate(np.array(trace_ninfo.posterior.betas[..., 1]).reshape(-1))
ninfo_kde.fit()
```

```python
skeptical_kde = sm.nonparametric.KDEUnivariate(np.array(trace_skeptical.posterior.betas[..., 1]).reshape(-1))
skeptical_kde.fit()
```

```python
fig, ax = plt.subplots(figsize=(10,5))
ax.set_xlim(left=-2, right=3)
ax.axvline(x=0, color="black", linestyle="--")
ax.plot(
    np.linspace(-3, 3, 1000), 
    skeptical_prior_dist.pdf(x), 
    linestyle="--", alpha=0.5, 
    color="orange", 
    label="skeptical prior"
)
ax.plot(ninfo_kde.support, ninfo_kde.density, label="noniformative")
ax.plot(skeptical_kde.support, skeptical_kde.density, label="skeptical")
_ = ax.legend()
```

```python

```
