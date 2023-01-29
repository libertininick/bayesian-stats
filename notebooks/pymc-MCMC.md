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
import os

from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc
from numpy import ndarray
from scipy import stats

from bayesian_stats import (
    get_auto_corr,
    get_effective_sample_size,
    get_gelman_rubin_diagnostic,
    get_invgamma_params,
)
```

# Example


- Suppose we have a model to represent the distribution of  year-over-year % change in total personnel for companies from a given industry:
	- `y | mu, sigma ~ N(loc=mu, scale=sigma)`
	- This model for `y` ( the distribution of percentage change for companies in the industry) can be modeled as a normal distribution with unknown:
        - mean, `mu`, which represents the (average) growth rate of the industry
        - standard deviation, `sigma`, which represents the  deviation in the average growth rate of the industry
- Our prior belief for the average growth rate for the industry is `mu ~ t(loc=0, scale=0, df=1)`
- Our prior belief for the standard deviation growth rate for the industry is `sigma ~ InverseGamma(alpha = sample_size / 2, beta = sigma_0 * sample_size / 2 )`
    - Prior effective sample size: `sample_size = 5`
    - Prior point estimate: `sigma_0 = 1.0`
- Now assume on the first day of the month, we get the year-over-year % change in total personnel for 10 companies in the given industry
- We want to find the posterior distribution of mu given those 10 samples:
	- `P(mu) = StudentT(loc=0, scale=0, df=1)`
    - `P(sigma) = InverseGamma(loc=0, scale=0, df=1)`
	- `P(mu_1 | sigma_0, y1, ..., y10) ∝ PRODUCT(Normal(mu_1, sigma_0).pdf(yi), for yi in Y[1:10]) * P(mu_1) * P(sigma_0)`
    - `P(sigma_1 | mu_1, y1, ..., y10) ∝ PRODUCT(Normal(mu_1, sigma_0).pdf(yi), for yi in Y[1:10]) * P(mu_1) * P(sigma_1)`
		- If we assume all the companies are i.i.d. then the likelihood is the product of all 10 likelihoods of the company


## Observed data

```python
y = np.array([1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9])
```

# Priors

```python
# Mu prior params
mu_priors = dict(mu=0, sigma=1, nu=1.0)
print(mu_priors)

# Sigma prior params
sigma_priors = get_invgamma_params(variance_prior=1, effective_sample_size=2)
print(sigma_priors)
```

## MCMC Model

```python
# Number of chains
chains = 5
# Number of samples per chain
draws = 5000

# Define model and draw samples
with pymc.Model() as model:
    
    # Priors for unknown model parameters
    mu = pymc.StudentT("mu", **mu_priors)
    sigma = pymc.InverseGamma("sigma", **sigma_priors)
    
    # Likelihood (sampling distribution) of observations
    y_obs = pymc.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
    
    # draw posterior samples
    idata = pymc.sample(draws=draws, chains=chains)
```

```python
pymc.model_to_graphviz(model)
```

## Posterior analysis

```python
mu = np.array(idata.posterior.mu)
sigma2 = np.array(idata.posterior.sigma)**2
for x in (mu, sigma2):
    print(x.mean(), x.std(), np.quantile(x, q=[0.025, 0.25, 0.5, 0.75, 0.975]))

```

```python
gelman_rubin_diagnostics = dict(
    mu=get_gelman_rubin_diagnostic(mu),
    sigma2=get_gelman_rubin_diagnostic(sigma2)
)
gelman_rubin_diagnostics
```

```python
effective_sample_sizes = dict(
    mu=get_effective_sample_size(mu[0], 100),
    sigma2=get_effective_sample_size(sigma2[0], 100)
)
effective_sample_sizes
```

```python
auto_corr = get_auto_corr(mu[0], lags=np.arange(1,100))

fig, ax = plt.subplots(figsize=(10,5))
ax.bar(x=np.arange(1,100), height=auto_corr)
ax.set_xlabel("Lag")
ax.set_ylabel("Auto Correlation")
```

```python
fig, ax = plt.subplots(figsize=(10,5))
bins = np.linspace(-1, 3, 100)
for i, chain in enumerate(mu):
    ax.hist(chain, bins=bins, density=True, alpha=0.5, label=f"chain {i + 1}")
_ = ax.legend()
```
