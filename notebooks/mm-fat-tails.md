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
import scipy
import statsmodels.api as sm
from dotenv import load_dotenv
from numpy import ndarray
from scipy import stats
from statsmodels.nonparametric.kde import KDEUnivariate

from bayesian_stats import (
    get_highest_density_interval,
    get_invgamma_params,
    get_rolling_windows,
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

# Data

```python
DATA_DIR = Path(os.environ.get("DATA_DIR"))
data = pd.read_csv(DATA_DIR.joinpath("sp_500.csv"))
dates = pd.to_datetime(data.Date).values
prices = data.Close.values

print(data.shape)
print(data.head(5))
```

## Log returns

```python
log_returns = np.log(prices[1:] / prices[:-1])
n = len(log_returns)
idxs = np.arange(n)

# Adjust dates and prices by 1
dates = dates[1:]
prices= prices[1:]
```

## Rolling windows

```python
# Volatility Window := 3 months
vol_window_len = 3 * 21

# Forecast Window
forecast_len = 21

vol_windows = get_rolling_windows(idxs, vol_window_len)
forecast_windows = get_rolling_windows(idxs, forecast_len)
```

### Rolling volatility (3-months)

```python
rolling_vol = np.mean(log_returns[vol_windows]**2, -1)**0.5
```

## Generate target and align

```python
cumulative_ret = np.cumsum(log_returns[-len(rolling_vol):])
forecast_ret = cumulative_ret[forecast_len:] - cumulative_ret[:-forecast_len]
normalized_forecast_ret = forecast_ret / (rolling_vol[:-forecast_len] * forecast_len**0.5)

d_aligned = dict(
    date=dates[-len(rolling_vol):-forecast_len],
    volatility=rolling_vol[:-forecast_len],
    target_raw=forecast_ret,
    target_normalize=normalized_forecast_ret,
)

d_aligned = pd.DataFrame(d_aligned).set_index('date')
d_aligned.head()
```

## Downsample

```python
d_sample = d_aligned.sample(frac=0.2, random_state=np.random.RandomState(123))
```

## MLE t-distribution

```python
dist_norm_MLE_raw = stats.norm(*stats.norm.fit(d_sample.target_raw))

MLE_fit_norm = stats.norm.fit(d_sample.target_normalize)
dist_norm_MLE = stats.norm(*MLE_fit_norm)
print(MLE_fit_norm)

MLE_fit_t = stats.t.fit(d_sample.target_normalize)
dist_t_MLE = stats.t(*MLE_fit_t)
print(MLE_fit_t)
```

## Calibration

```python
obs = np.sort(d_sample.target_normalize.values)
n = len(obs)
ideal_cdc = np.linspace(0, 1, n + 2)[1:-1]
```

```python
norm_raw_cdc = dist_norm_MLE_raw.cdf(np.sort(d_sample.target_raw.values))
norm_cdc = dist_norm_MLE.cdf(obs)
t_cdc = dist_t_MLE.cdf(obs)
```

```python

fig, axs = plt.subplots(ncols=3, figsize=(15,5))

for i, (lhs, rhs) in enumerate([(0, 1), (0, 0.1), (0.9, 1.0)]):
    
    axs[i].set_xlabel("Observed probability")
    axs[i].set_ylabel("Predicted probability")
    axs[i].plot(ideal_cdc, ideal_cdc, color="black", linestyle="--")
    axs[i].plot(ideal_cdc, norm_raw_cdc, color="pink")
    axs[i].plot(ideal_cdc, norm_cdc, color="orange")
    axs[i].plot(ideal_cdc, t_cdc, color="red")

    axs[i].set_xlim(lhs, rhs)
    axs[i].set_ylim(lhs, rhs)
```

# Model


### Model specification

```python
with pm.Model() as mm_model:
    n_components = 2
    
    # Prior for unknown mean of mixture distributions
    mu = pm.Normal("mu", mu=0, sigma=10)

    # Prior for unknown stds of mixture distributions
    variances = [
        pm.InverseGamma(
            f"var{i}", 
            **get_invgamma_params(variance_prior=(i+0.5)**2, effective_sample_size=1),
        )
        for i in range(n_components)
    ]
    
    
    # Component distributions
    components = [
        pm.Normal.dist(mu=mu, sigma=variances[i]**0.5)
        for i in range(n_components)
    ]
    
    # Prior for mixture wts
    wts = pm.Dirichlet("wts", a=np.ones(n_components))
    
    # Likelihood of latent mixture class
    like = pm.Mixture('like', w=wts, comp_dists=components, observed=d_sample.target_normalize)
    
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
chains = 3

# Number of samples per chain
draws = 3000

with mm_model:
    # draw posterior samples
    trace = pm.sample(draws=draws, chains=chains)
    
#     # draw posterior perdictive samples
#     trace = pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=rng)
```

```python
summary = az.summary(trace.posterior)
summary
```

```python
from dataclasses import dataclass
from numpy.typing import NDArray

@dataclass
class MixturePool:
    mu: NDArray[np.float_]
    variances: NDArray[np.float_]
    wts: NDArray[np.float_]
        
    @property
    def size(self) -> int:
        return len(self.mu)
        
    def sample(self, n: int):
        sample_idxs = np.random.choice(np.arange(self.size), n)
        
        # Get mixture index
        cumwts = np.cumsum(self.wts[sample_idxs], axis=-1)
        mixture_idx = ((np.random.rand(n, 1) - cumwts) > 0).sum(-1, keepdims=True)
        
        means = self.mu[sample_idxs]
        stds = self.variances[sample_idxs]**0.5
        stds = np.take_along_axis(stds, mixture_idx, axis=-1).squeeze()
        return stats.norm.rvs(loc=means, scale=stds)
```

```python
mpool = MixturePool(
    mu=trace.posterior.mu.values.flatten(),
    variances=np.stack(
        [
            trace.posterior[f"var{i}"].values.flatten()
            for i in range(n_components)
        ],
        axis=-1
    ),
    wts = trace.posterior.wts.values.reshape(-1, n_components),
)
```

```python
posterior_samples = np.sort(mpool.sample(10 * len(obs)))
mm_cdc = np.searchsorted(posterior_samples, obs, side='left') / len(posterior_samples)
```

```python

```

```python

fig, axs = plt.subplots(ncols=3, figsize=(15,5))

for i, (lhs, rhs) in enumerate([(0, 1), (0, 0.1), (0.9, 1.0)]):
    
    axs[i].set_xlabel("Observed probability")
    axs[i].set_ylabel("Predicted probability")
    axs[i].plot(ideal_cdc, ideal_cdc, color="black", linestyle="--")
    axs[i].plot(ideal_cdc, norm_raw_cdc, color="pink")
    axs[i].plot(ideal_cdc, norm_cdc, color="orange")
    axs[i].plot(ideal_cdc, t_cdc, color="red")
    axs[i].plot(ideal_cdc, mm_cdc, color="blue")

    axs[i].set_xlim(lhs, rhs)
    axs[i].set_ylim(lhs, rhs)
```

```python

```
