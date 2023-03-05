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
    get_probability_non_zero,
    get_rolling_windows,
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

```python
n = 100
x = np.random.randn(n, 120)

x[:,:100] = np.random.randn(100)
x = np.cumsum(x, -1)

fig, ax = plt.subplots(figsize=(10,5))
for i in range(n):
    ax.plot(x[i], color="black", alpha=0.05)
    
ax.axvline(x=99, color="red", linestyle="--")
ax.set_xticks([])
ax.set_yticks([])
# ax.spines[['left']].set_visible(False)
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

# Longer Term Context Window := 10 years
ctx_window_len = 10 * 252

# Forecast Window
forecast_len = 21

vol_windows = get_rolling_windows(idxs, vol_window_len)
ctx_windows = get_rolling_windows(idxs, ctx_window_len)
forecast_windows = get_rolling_windows(idxs, forecast_len)
```

### Rolling volatility (3-months)

```python
rolling_vol = np.mean(log_returns[vol_windows]**2, -1)**0.5
```

### Normalized price (10-yr windows)

```python
# Take log of prices
z = np.log(prices)

# 10yr rolling mean and std of 3mo Vol
rolling_z = z[ctx_windows]
z_mean = rolling_z.mean(-1)
z_std = rolling_z.std(-1)

# Normalize price by rolling mean and std
price_norm = (rolling_z[:, -1] - z_mean) / z_std
print(np.quantile(price_norm, q=[0.05, 0.25, 0.5, 0.75, 0.95]))

fig, axs = plt.subplots(nrows=2, figsize=(10,10))
_ = axs[0].plot(dates[-len(price_norm):], price_norm)
_ = axs[1].hist(
    price_norm,
    bins=51,
    edgecolor="black",
    alpha=0.5,
)
```

### Normalized 3-month volatility (10-yr windows)

```python
# Take log of volatility
z = np.log(rolling_vol)

# 10yr rolling mean and std of 3mo Vol
rolling_z = z[ctx_windows[:-(vol_window_len - 1)]]
z_mean = rolling_z.mean(-1)
z_std = rolling_z.std(-1)

# Normalize volatility by rolling mean and std
volatility_norm = (rolling_z[:, -1] - z_mean) / z_std
print(np.quantile(volatility_norm, q=[0.05, 0.25, 0.5, 0.75, 0.95]))

fig, axs = plt.subplots(nrows=2, figsize=(10,10))
_ = axs[0].plot(dates[-len(volatility_norm):], volatility_norm)
_ = axs[1].hist(
    volatility_norm,
    bins=51,
    edgecolor="black",
    alpha=0.5,
)
```

## Price - Volatatiltiy Environments

```python
# Find min data length among rolling variables
n = min(len(price_norm), len(volatility_norm))

# Align series
price_norm = price_norm[-n:]
volatility_norm = volatility_norm[-n:]
```

```python
n_env_bins = 3

env_labels = []
for var in [price_norm, volatility_norm]:
    bins = np.quantile(var, q=np.arange(0, 1, 1/n_env_bins)) 
    env_labels.append(np.digitize(var, bins) - 1)

lbl = {0: "L", 2: "H"}
env_labels = np.array([
    f"P{lbl[p]}_V{lbl[v]}" if p != 1 and v != 1 else "baseline"
    for p, v in np.stack(env_labels, axis=-1)
])

envs, cts = np.unique(env_labels, return_counts=True)

for env, ct in zip(envs, cts):
    print(f"{env:<15} {ct/len(env_labels):>8.1%}")
```

## Generate target and align

```python
cumulative_ret = np.cumsum(log_returns[-n:])
next_month_ret = cumulative_ret[forecast_len:] - cumulative_ret[:-forecast_len]

d_aligned = dict(
    date=dates[-n:-forecast_len],
    price_norm=price_norm[:-forecast_len],
    volatility_norm=volatility_norm[:-forecast_len],
    environment=env_labels[:-forecast_len],
    target=next_month_ret,
)

d_aligned = pd.DataFrame(d_aligned).set_index('date')
d_aligned.head()
```

## Downsample and split out columns

```python
d_sample = d_aligned.sample(frac=0.2)

# One-hot-encode environments
environments = ["baseline", 'PL_VH', 'PL_VL', 'PH_VL', 'PH_VH']
env_ohe = (
    one_hot_encode(d_sample.environment, var_name="env")
    .loc[:, [f"env_{env}" for env in environments[1:]]]
)

# Normalize target
tgt_mean = d_sample.target.mean()
tgt_std = d_sample.target.std()
d_sample["target"] = (d_sample.target - tgt_mean) / tgt_std

d_sample.shape
```

### Distribuition of returns

```python
MLE_fit_norm = stats.norm.fit(d_sample.target)
print(MLE_fit_norm)
MLE_fit_t = stats.t.fit(d_sample.target)
print(MLE_fit_t)

dist_norm = stats.norm(*MLE_fit_norm)
dist_t = stats.t(*MLE_fit_t)

fig, ax = plt.subplots(figsize=(10,5))
_, bins, _ = ax.hist(
    d_sample.target,
    bins=101,
    density=True,
    alpha=0.5,
)
ax.plot(
    bins,
    dist_norm.pdf(bins),
    color="black",
    linewidth=1,
    linestyle="--",
    label="MLE Normal(mu=0.0, sigma=1.0)"
)
ax.plot(
    bins,
    dist_t.pdf(bins),
    color="black",
    linewidth=1,
    linestyle="-",
    label="MLE t(mu=0.07, sigma=0.73, df=4.34)"
)
# ax.axvline(x=0, color="black", linewidth=0.5)

ax.set_xlabel("Normalized Return")
ax.set_yticks([])
ax.spines[['left']].set_visible(False)
ax.set_title(
    "Distribution of observed 1 month (normalized) returns\nS&P 500 1938 - 2022",
    loc="left"
)
_ = ax.legend()

```

### Summary of Environments

```python
env_summary = dict()
for grp_lbl, grp in d_sample.groupby("environment"):
    df, mu, sigma = stats.t.fit(grp.target)
    env_summary[grp_lbl] = dict(
        count=len(grp),
        mu=mu,
        sigma=sigma,
        df=df,
    )
env_summary = pd.DataFrame(env_summary).T
env_summary
```

## Model


## Priors

```python
# Non-informative baslines 
x = np.linspace(-5, 5, 1001)

fig, axs = plt.subplots(ncols=3, figsize=(15,5), sharey=True)

# Priors for baseline parameters
# Mean
axs[0].plot(
    x,
    stats.norm(loc=0, scale=10).pdf(x),
    color="black",

)
axs[0].set_xlabel("mean (mu) parameter")
axs[0].set_ylabel("Prior density")
axs[0].set_title(
    "Prior distribution for baseline return distribution parameters\n~ StudenT(mu, sigma, df)",
    loc="left"
)

# Standard Deviation
axs[1].plot(
    np.exp(x)**0.25,
    stats.norm(loc=0, scale=10).pdf(x),
    color="black",

)
axs[1].set_xlabel("standard deviation (sigma) parameter")


# Degrees of freedom
axs[2].plot(
    np.exp(x)**0.5,
    stats.norm(loc=0, scale=10).pdf(x),
    color="black",

)
axs[2].set_xlabel("degress of freedom (df) parameter")
# axs[0].plot(, d)
```

### Model specification

```python
with pm.Model() as model:
    
    # Priors
    baseline_params = pm.Normal("baseline_params", mu=0, sigma=10, shape=3)
    env_coeffs = pm.Laplace("environment_coeffs", mu=0, b=1/2, shape=(4, 3))

    # Apply environment adjustments to baseline parameters
    env_indicators = env_ohe.values
    param_adjments = env_indicators @ env_coeffs
    params = param_adjments + baseline_params
    
    # Apply deterministic transforms to parameters
    mu = params[:, 0]
    sigma = np.exp(params[:, 1])**0.25
    df = np.exp(params[:, 2])**0.5
    
    # Likelihood of observed returns
    returns = pm.StudentT(
        "returns",
        mu=mu,
        sigma=sigma,
        nu=df,
        observed=d_sample.target,
    )
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
chains = 3

# Number of samples per chain
draws = 3000

with model:
    # draw posterior samples
    trace = pm.sample(draws=draws, chains=chains)
    
    # draw posterior perdictive samples
    trace = pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=rng)
```

```python
summary = az.summary(trace.posterior)
summary
```

```python
probability_non_zero = np.zeros((4,3))
for i in range(4):
    for j in range(3):
        probability_non_zero[i,j] = get_probability_non_zero(
            trace.posterior.environment_coeffs[...,i,j].values
        )
```

```python
pd.DataFrame(
    probability_non_zero,
    index=environments[1:],
    columns=["mu", "sigma", "df"]
)
             
```

```python
baselines = summary["mean"][:3].values
env_coeffs = summary["mean"][-12:].values.reshape(-1,3)
env_dist_params = baselines[None, ...] + np.vstack((np.zeros((1,3)), env_coeffs))

env_dist_params[:,1] = np.exp(env_dist_params[:,1])**0.25
env_dist_params[:,2] = np.exp(env_dist_params[:,2])**0.5

env_dist_params = pd.DataFrame(
    env_dist_params,
    index=environments,
    columns=["mu", "sigma", "df"]
)

env_dist_params
```

```python
labels = [
    'baseline', 
    'Price Low & Vol High',
    'Price Low & Vol Low',
    'Price High & Vol Low',
    'Price High & Vol High',
]

colors = [
    "black",
    "red",
    "purple",
    "green",
    "orange",
]

fig, ax = plt.subplots(figsize=(12,6))
_, bins, _ = ax.hist(d_sample.target, bins=101, density=True, alpha=0.5)
for i in range(len(env_dist_params)):
    ax.plot(
        bins,
        stats.t(
            env_dist_params.iloc[i,2],
            loc=env_dist_params.iloc[i,0],
            scale=env_dist_params.iloc[i,1],
        ).pdf(bins),
        color=colors[i],
        label=labels[i],
        linestyle="--" if i == 0 else '-',
    )

    
ax.set_xlabel("Normalized Return")
ax.set_yticks([])
ax.spines[['left']].set_visible(False)
ax.set_title(
    "Distribution of observed 1 month (normalized) returns\nS&P 500 1938 - 2022",
    loc="left"
)
_ = ax.legend(loc="upper left")
```

```python

```
