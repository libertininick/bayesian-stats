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

# Leinhardt data


## Load data

```python
DATA_DIR = Path(os.environ.get("DATA_DIR"))
data = pd.read_csv(DATA_DIR.joinpath("leinhardt.csv"))
missing = data.isnull().values.any(-1)
data = data.loc[~missing, :]
data["log_income"] = np.log(data.income)
data["log_infant"] = np.log(data.infant)
data.shape
```

```python
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(data.log_income, data.log_infant)
ax.set_xlabel("Log(Income)")
ax.set_ylabel("Log(Infant Mortality)")
```

## Model


### Priors

```python
betas_prior_params = dict(
    mu=0,
    sigma=int(1e6**0.5),
)
print(betas_prior_params)

sigma_prior_params = get_invgamma_params(sigma_prior=10, effective_sample_size=5)
print(sigma_prior_params)
```

### Model specification

```python
with pm.Model() as lm:
    # Priors for unknown model parameters
    betas = pm.Normal("betas", **betas_prior_params, shape=2)
    sigma = pm.InverseGamma("sigma", **sigma_prior_params)

    # Expected value of outcome
    mu = betas[0] + betas[1] * data.log_income

    # Likelihood (sampling distribution) of observations
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data.log_infant)
    
lm
```

```python
pm.model_to_graphviz(lm)
```

### prior predictive

```python
RANDOM_SEED = 58
rng = np.random.default_rng(RANDOM_SEED)
with lm:
    trace_prior = pm.sample_prior_predictive(samples=1000, random_seed=rng)
```

```python
fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(trace_prior.prior_predictive.y[0, :, 0], bins=30)
_ = ax.axvline(x=data.log_infant[0], color="red")
```

### Model fitting

```python
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

<!-- #region heading_collapsed=true -->
### Posterior samples
<!-- #endregion -->

```python hidden=true
betas = np.array(idata.posterior.betas)
print(betas.shape)
sigma = np.array(idata.posterior.sigma)
print(sigma.shape)
```

## Convergence Diagnostics


### Trace plots

```python
fig, axs = plt.subplots(nrows=3, figsize=(10,15))
for i in range(chains):
    axs[0].plot(betas[i, :, 0], alpha=0.5, label=f"Chain {i}")
    axs[1].plot(betas[i, :, 1], alpha=0.5, label=f"Chain {i}")
    axs[2].plot(sigma[i, :], alpha=0.5, label=f"Chain {i}")
axs[0].set_title("Beta 0 trace plot")
axs[1].set_title("Beta 1 trace plot")
axs[2].set_title("Sigma trace plot")
```

<!-- #region heading_collapsed=true -->
### Effective sample sizes
<!-- #endregion -->

```python hidden=true
betas_ess = np.array([
    [
        get_effective_sample_size(chain, lags=100) for chain in betas[..., i]
    ]
    for i in range(betas.shape[-1])
])
betas_ess.mean(-1)
```

## Posterior Analysis


### Posterior Mean Estimates

```python
pme = betas.mean((0,1))
pme
```

```python
y_hat = pme[0] + pme[1]*data.log_income
```

```python

```

```python
residuals = data.log_infant - y_hat
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(y_hat, residuals)
```

## T-distribution likelihood


### Priors

```python
betas_prior_params = dict(
    mu=0,
    sigma=int(1e6**0.5),
)
print(betas_prior_params)

sigma_prior_params = get_invgamma_params(sigma_prior=10, effective_sample_size=5)
print(sigma_prior_params)

df_prior_params = {'lam': 0.25}
print(df_prior_params)
```

```python
fig, ax = plt.subplots(figsize=(10,5))

x = np.linspace(0, 30, 100)
for l in np.linspace(0.25, 1, 5):
    ax.plot(x, stats.expon(scale=1/l).pdf(x), label=f"{l:.2f}")
ax.legend()
```

### Model specification

```python
with pm.Model() as lm1:
    # Priors for unknown model parameters
    betas = pm.Normal("betas", **betas_prior_params, shape=2)
    sigma = pm.InverseGamma("sigma", **sigma_prior_params)
    df = pm.Exponential("df", **df_prior_params)

    # Expected value of outcome
    mu = betas[0] + betas[1] * data.log_income
    nu = 2 + df
    

    # Likelihood (sampling distribution) of observations
    y = pm.StudentT("y", mu=mu, sigma=sigma, nu=nu, observed=data.log_infant)
    
lm1
```

```python
pm.model_to_graphviz(lm1)
```

### Model fitting

```python
# Number of chains
chains = 5

# Number of samples per chain
draws = 5000

with lm1:
    # draw posterior samples
    idata = pm.sample(draws=draws, chains=chains)
```

```python
az.summary(idata)
```

### Posterior distribution of DF

```python
fig, ax = plt.subplots(figsize=(10,5))
_, bins, _ = ax.hist(np.array(idata.posterior.df[0]), bins=50, density=True)
_ = ax.plot(bins, stats.expon(scale=1/0.25).pdf(bins))
_ = ax.set_xticklabels(ax.get_xticklabels)
```

```python
az.convert_to_inference_data(idata)
```

# Anscombe data

```python
data = pd.read_csv(DATA_DIR.joinpath("anscombe.csv"))
missing = data.isnull().values.any(-1)
data = data.loc[~missing, :]
```

```python
data.head()
```

```python
fig, axs = plt.subplots(nrows=3, figsize=(5, 15))
axs[0].scatter(data.income, data.education)
axs[1].scatter(data.young, data.education)
axs[2].scatter(data.urban, data.education)
```

## Default Reference odinary least squares regression

```python
X = sm.add_constant(data[['income', 'young', 'urban']])
mod = sm.OLS(data.education, X)
res = mod.fit()
res.summary()
```

## Bayesian regression

```python
betas_prior_params = dict(
    mu=0,
    sigma=int(1e6**0.5),
)
print(betas_prior_params)

sigma_prior_params = get_invgamma_params(sigma_prior=1500*0.5, effective_sample_size=2)
print(sigma_prior_params)
```

```python
with pm.Model() as lm2:
    # Priors
    betas = pm.Normal("betas", **betas_prior_params, shape=3+1)
    sigma = pm.InverseGamma("sigma", **sigma_prior_params, shape=1)
    
    # Expected value of outcome
    mu = betas[0] + data[['income', 'young', 'urban']].values @ betas[1:]

    # Likelihood (sampling distribution) of observations
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data.education)
    
lm2
```

```python
pm.model_to_graphviz(lm2)
```

```python
# Number of chains
chains = 5

# Number of samples per chain
draws = 5000

with lm2:
    # draw posterior samples
    idata = pm.sample(draws=draws, chains=chains)
    
az.summary(idata)
```

```python
betas = np.array(idata.posterior.betas)
sigma = np.array(idata.posterior.sigma)
```

```python
np.mean(betas[0,:,1] > 0)
```

```python
sigma.shape
```

```python
for i in range(betas.shape[-1]):
    print(f"beta[{i}]: {get_gelman_rubin_diagnostic(betas[..., i]):>8.4}")


print(f"sigma  : {get_gelman_rubin_diagnostic(sigma[...,0]):>8.4}")
```

```python
pme = betas.mean((0,1))
pme
```

```python
y_hat = (pme[0] + data[['income', 'young', 'urban']] @ pme[1:, None]).squeeze()
```

```python
data.education.shape
```

```python
residuals = data.education - y_hat
residuals.shape
```

```python
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(y_hat, residuals)
```

```python

```
