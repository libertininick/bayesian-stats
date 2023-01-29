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

## Simulate data

```python
# Draw a expected number of cookies produced for each of 5 locations
n_locs = 5
true_means = stats.norm(loc=8.5, scale=1.5).rvs(n_locs)
true_means.round(2)
```

```python
# Draw 30 cookies from each location:
n_cookies = 30
true_chips = dict()
for i, mu in enumerate(true_means, 1):
    true_chips[i] = stats.poisson(mu=mu).rvs(n_cookies)
```

```python
data = pd.melt(pd.DataFrame(true_chips))
data.columns = ["location", "chips"]
```

```python
fig, axs = plt.subplots(ncols=n_locs, figsize=(20,5), sharey=True)
for i in range(1, n_locs + 1):
    axs[i - 1].set_xlabel(f"Location {i}")
    axs[i - 1].axhline(y=8.5, linestyle="--")
    axs[i - 1].boxplot(data.query("location == @i").chips)
```

## Hierarchical Possion Model


### Priors for `alpha` and `beta`
- The different location means `lambda 1-5` come from a gamma distribution with hyperparameters `alpha`, `beta`.
- Alpha and beta control the distribution for these means between locations.
- The mean of this gamma distribution (`alpha / beta`) will represent the overall mean of number of chips for all cookies.
- The variance of this gamma distribution controls the variability between locations in the mean number of chips. 
    - `std = SQRT(alpha / beta**2)`
    - If this variance is high, the mean number of chips will vary widely from location to location. 
    - If it is small, the mean number of chips will be nearly the same from location to location.
- Solving for `alpha, beta` from `mean, std`:
    - `alpha = mean * beta`
    - `beta = mean / std**2`

```python
# prior guess for global mean
mu_0 = 8.5

# Prior sample size
sample_size = 30

# Shape 
alpha = sample_size / 2 

# Scale 
beta = mu_0 * sample_size / 2 

global_mean_params = dict(
    alpha=alpha,
    beta=beta,
)

fig, ax = plt.subplots(figsize=(10,5))
x = np.linspace(1, 16, 1000)
ax.plot(x, stats.invgamma(a=alpha, scale=beta).pdf(x))
```

```python
# prior guess for global std between locations
mu_0 = 5

# Prior sample size
sample_size = 15

# Shape 
alpha = sample_size / 2 

# Scale 
beta = mu_0 * sample_size / 2 

global_std_params = dict(
    alpha=alpha,
    beta=beta,
)

fig, ax = plt.subplots(figsize=(10,5))
x = np.linspace(1, 16, 1000)
ax.plot(x, stats.invgamma(a=alpha, scale=beta).pdf(x))
```

```python
print(global_mean_params)
print(global_std_params)
```

### Model specification

```python
with pm.Model() as h_model:
    # Priors for unknown model parameters
    global_mean = pm.InverseGamma("global_mean", **global_mean_params)
    global_std = pm.InverseGamma("global_std", **global_std_params)
    beta = global_mean / global_std**2
    alpha = global_mean * beta
    

    # Expected value of outcomes
    lambdas = pm.Gamma("lambdas", alpha=alpha, beta=beta, shape=5)

    # Likelihood (sampling distribution) of observations
    for i in range(5):
        y = pm.Poisson(f"y_{i+1}", mu=lambdas[i], observed=data.query("location == @i + 1").chips.values)
    
h_model
```

```python
pm.model_to_graphviz(h_model)
```

### Model fitting

```python
RANDOM_SEED = 58 
rng = np.random.default_rng(RANDOM_SEED)

# Number of chains
chains = 5

# Number of samples per chain
draws = 5000

with h_model:
    # draw posterior samples
    trace_h_model = pm.sample(draws=draws, chains=chains)
```

```python
az.summary(trace_h_model)
```

```python
az.plot_trace(trace_h_model, combined=True)
```

```python
axs = az.plot_forest(trace_h_model, var_names=["lambdas"], combined=True, hdi_prob=0.95, r_hat=True)
axs[0].axvline(x=8.5, color="red", linestyle="--")
```

## Posterior predictive distributions

```python
posterior_g_mean = trace_h_model.posterior.global_mean
posterior_g_std = trace_h_model.posterior.global_std
posterior_alpha = np.array(posterior_g_mean**2 / posterior_g_std**2).reshape(-1)
posterior_beta = np.array(posterior_g_mean / posterior_g_std**2).reshape(-1)

# Posterior distribution for lamba
n_samples = len(posterior_beta)
lambda_pred_dist = stats.gamma(a=posterior_alpha, scale=1/posterior_beta).rvs(n_samples)

# Posterior distribution for # chips
n_chips_dist = stats.poisson(mu=lambda_pred_dist).rvs(n_samples)
```

```python
(lambda_pred_dist > 15).mean()
```

```python
fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(lambda_pred_dist, bins=100)
```

```python
fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(n_chips_dist, bins=30)
```

```python
# posterior probability that the next cookie at location 1 will have frewer than 7 chips
lambda_1_posterior = np.array(trace_h_model.posterior.lambdas[0]).reshape(-1)
n_chips_at_1_dist = stats.poisson(mu=lambda_1_posterior).rvs(n_samples)

(n_chips_at_1_dist < 7).mean()
```

# Company Growth


## Load data

```python
DATA_DIR = Path(os.environ.get("DATA_DIR"))
data = pd.read_csv(DATA_DIR.joinpath("company-growth.csv"))
missing = data.isnull().values.any(-1)
data = data.loc[~missing, :]
data.shape
```

```python
data.groupby("grp").mean()
```

```python
data.groupby("grp").mean().std()
```

### Model specification

```python
with pm.Model() as h_model:
    # Priors for unknown model parameters
    mean_mu = pm.Normal("mean_mu", mu=0, sigma=1e3)
    mean_variance = pm.InverseGamma("mean_variance", alpha=1/2, beta=1*3/2)
    
    # Industry specific means from a common distribution
    industry_means = pm.Normal("industry_means", mu=mean_mu, sigma=mean_variance**0.5, shape=5)
    
    # common variance across all observations
    global_variance = pm.InverseGamma("global_std", alpha=2/2, beta=2*1/2)

    # Likelihood (sampling distribution) of observations
    for i in range(5):
        y = pm.Normal(
            f"y_{i+1}",
            mu=industry_means[i],
            sigma=global_variance**0.5,
            observed=data.query("grp == @i + 1").y.values)
    
h_model
```

```python
pm.model_to_graphviz(h_model)
```

### Model fitting

```python
RANDOM_SEED = 58 
rng = np.random.default_rng(RANDOM_SEED)

# Number of chains
chains = 5

# Number of samples per chain
draws = 5000

with h_model:
    # draw posterior samples
    trace_h_model = pm.sample(draws=draws, chains=chains)
```

```python
az.summary(trace_h_model)
```

```python
az.summary(trace_h_model)["mean"][1:6].std()
```

```python

```
