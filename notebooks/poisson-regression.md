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

# Poisson likelihood


## Load data

```python
DATA_DIR = Path(os.environ.get("DATA_DIR"))
data = pd.read_csv(DATA_DIR.joinpath("bad-health.csv"))
missing = data.isnull().values.any(-1)
data = data.loc[~missing, :]
print(data.shape)
data.head()
```

## Default Reference Possion Model

```python
x = sm.add_constant(data[["badh", "age"]])
x["badh_age"] = x.badh * x.age
poisson_model = sm.Poisson(data.numvisit.values, x)
results = poisson_model.fit()
results.summary()
```

### Priors

```python
prior_params = dict(
    mu=0,
    sigma=int(1e3),
)
print(prior_params)
```

```python
data.badh.values
```

### Model specification

```python
with pm.Model() as poisson_model:
    # Priors for unknown model parameters
    betas = pm.Normal("betas", **prior_params, shape=4)

    # Expected value of outcome
    log_lambda = (
        betas[0] 
        + data.badh.values * betas[1] 
        + data.age.values * betas[2] 
        + data.badh.values * data.age.values * betas[3]
    )

    # Likelihood (sampling distribution) of observations
    y = pm.Poisson("y", mu=np.exp(log_lambda), observed=data.numvisit.values)
    
poisson_model
```

```python
pm.model_to_graphviz(poisson_model)
```

### Model fitting

```python
RANDOM_SEED = 58 
rng = np.random.default_rng(RANDOM_SEED)

# Number of chains
chains = 5

# Number of samples per chain
draws = 5000

with poisson_model:
    # draw posterior samples
    trace_posterior = pm.sample(draws=draws, chains=chains)
    trace_posterior = pm.sample_posterior_predictive(trace_posterior, extend_inferencedata=True, random_seed=rng)
```

```python
az.summary(trace_posterior)
```

```python
az.plot_trace(trace_posterior, combined=True)
```

```python
az.plot_rank(trace_posterior, kind="bars")
```

```python
axs = az.plot_forest(trace_posterior, var_names=["betas"], combined=True, hdi_prob=0.95, r_hat=True)
axs[0].axvline(x=0, color="red", linestyle="--")
```

## Posterior predictive distributions
Everything we've done so far, could have easily been computed using the Default Reference Analysi; now, let's do something where it really pays off to have an Bayesian Analysis with posterior samples:
- Let's say, we have two people age 35. 
- One person is in good health and the other person is in poor health.
- What is the posterior probability that the individual with poor health will have more doctor visits?

Here, we need to create Monte Carlo samples for the actual responses themselves.

```python
posterior_betas  = np.array(trace_posterior.posterior.betas).reshape(-1,4)
n_samples = posterior_betas.shape[0]

# (intercept, bad health, age, bad health * age)
person_1 = np.array([1, 0, 35, 0 * 35])
person_2 = np.array([1, 1, 35, 1 * 35])

# Posterior mean of each person's distribution
posterior_lambdas_1 = np.exp(posterior_betas @ person_1)
posterior_lambdas_2 = np.exp(posterior_betas @ person_2)

# Posterior distribution of number of visits
posterior_numvists_1 = stats.poisson(mu=posterior_lambdas_1).rvs(n_samples)
posterior_numvists_2 = stats.poisson(mu=posterior_lambdas_2).rvs(n_samples)
```

```python
# posterior probability that the individual with poor health will have more doctor visits
(posterior_numvists_2 > posterior_numvists_1).mean()
```

```python
fig, axs = plt.subplots(nrows=2, figsize=(10, 10))
bins = np.linspace(0, max(posterior_numvists_1.max(), posterior_numvists_2.max()), 100)
_ = axs[0].hist(posterior_numvists_1, bins=bins)
_ = axs[1].hist(posterior_numvists_2, bins=bins)
```

```python
np.exp(1.5 + 0.8*-0.3 + 1.2*1.0)
```

```python
stats.poisson(mu=2*15).cdf(21)
```

# Caller data

```python
DATA_DIR = Path(os.environ.get("DATA_DIR"))
data = pd.read_csv(DATA_DIR.joinpath("callers.csv"))
missing = data.isnull().values.any(-1)
data = data.loc[~missing, :]
data["call_rate"] = data.calls / data.days_active
print(data.shape)
data.head()
```

```python
fig, axs = plt.subplots(ncols=2, figsize=(10, 10), sharey=True)
_ = axs[0].boxplot(data.query("isgroup2 == 0").call_rate)
_ = axs[1].boxplot(data.query("isgroup2 == 1").call_rate)
_ = axs[0].set_title("Group1")
_ = axs[1].set_title("Group2")
```

### Model specification

```python
prior_params = dict(
    mu=0,
    sigma=10,
)
print(prior_params)

with pm.Model() as poisson_model:
    # Priors for unknown model parameters
    betas = pm.Normal("betas", **prior_params, shape=3)

    # Expected value of outcome
    log_lambda = (
        betas[0] 
        + data.isgroup2.values * betas[1] 
        + data.age.values * betas[2] 
    )

    # Likelihood (sampling distribution) of observations
    y = pm.Poisson("y", mu=np.exp(log_lambda) * data.days_active, observed=data.calls.values)
    
poisson_model
```

```python
RANDOM_SEED = 58 
rng = np.random.default_rng(RANDOM_SEED)

# Number of chains
chains = 5

# Number of samples per chain
draws = 5000

with poisson_model:
    # draw posterior samples
    trace_posterior = pm.sample(draws=draws, chains=chains)
```

```python
az.summary(trace_posterior)
```

```python
np.mean(trace_posterior.posterior.betas[..., 1] > 0)
```

```python

```
