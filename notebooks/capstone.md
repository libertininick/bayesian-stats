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
    get_rolling_windows,
)

from bayesian_stats.find_declines import find_market_declines, in_market_decline

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

print(data.shape)
print(data.head(5))
```

## Market environment: Rising vs. Declining

```python
dates = pd.to_datetime(data.Date).values
prices = data.Close.values

market_declines = find_market_declines(dates=dates, prices=prices)


n_declines = len(market_declines)
n_decline_days = sum(md.length for md in market_declines)

print(f"{n_declines} declines")
print(f"{n_decline_days/len(prices):.0%} of time in decline")
print(f"Size: {np.quantile([md.size for md in market_declines], q=[0, 0.25, 0.5, 0.75, 1]).round(2)}")
print(f"Length: {np.quantile([md.length for md in market_declines], q=[0, 0.25, 0.5, 0.75, 1]).round(0)}")
```

```python
fig, ax = plt.subplots(figsize=(15, 7))

ax.set_title("S&P 500 Index\n1928-01-04 to 2022-10-21")
ax.set_yscale("log")
ax.set_xlabel("Date")
ax.set_ylabel("Price (log scale)")
ax.plot(dates, prices, color="black")

for i, md in enumerate(market_declines):
    ax.axvspan(
        md.start, 
        md.end, 
        color='red', 
        alpha=0.5,
        label = None if i else "Market Decline",
    )
_ = ax.legend(fontsize=14, loc='upper left')
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
# Long Term := 10 years
n_long = 10 * 252

# Short Term := 3 months
n_short = 3 * 21

windows_long = get_rolling_windows(idxs, n_long)
windows_short = get_rolling_windows(idxs, n_short)
```

### Rolling 10-yr price range

```python
rolling_prices = prices[windows_long]
rolling_min_price = rolling_prices.min(-1)
rolling_max_price = rolling_prices.max(-1)
rolling_price_range = (rolling_prices[:,-1] - rolling_min_price) / (rolling_max_price - rolling_min_price)
```

```python
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(dates[n_long - 1:], rolling_price_range)
```

### Rolling 3mo mean & std

```python
rolling_3mo_returns = log_returns[windows_short]
rolling_3mo_mean = rolling_3mo_returns.mean(-1)
rolling_3mo_std = rolling_3mo_returns.std(-1)
```

```python
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(dates[n_short - 1:], rolling_3mo_std)
```

### Rolling 10yr mean & std

```python
rolling_10yr_returns = log_returns[windows_long]
rolling_10yr_mean = rolling_10yr_returns.mean(-1)
rolling_10yr_std = rolling_10yr_returns.std(-1)
```

```python
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(dates[n_long - 1:], rolling_10yr_std)
```

### Rolling 10-yr 3-month volatility range

```python
rolling_10yrs_3mo_std = rolling_3mo_std[windows_long[:-(n_short - 1)]]
rolling_min_3mo_std = rolling_10yrs_3mo_std.min(-1)
rolling_max_3mo_std = rolling_10yrs_3mo_std.max(-1)
rolling_3mo_std_range = (rolling_10yrs_3mo_std[:,-1] - rolling_min_3mo_std) / (rolling_max_3mo_std - rolling_min_3mo_std)
```

```python
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(dates[n_short + n_long - 2:], rolling_3mo_std_range)
```

```python
n_diff = len(rolling_price_range) - len(rolling_3mo_std_range)
```

```python
n_bins = 5
bins = np.quantile(rolling_price_range[n_diff:], q=np.arange(0, 1, 1/n_bins))
bin_labels = np.digitize(rolling_price_range[n_diff:], bins) - 1
print(np.unique(bin_labels, return_counts=True))

fig, axs = plt.subplots(ncols=n_bins, figsize=(15, 7), sharey=True)
for i in range(n_bins):
    axs[i].boxplot(
        rolling_3mo_std_range[bin_labels == i],
        sym='',                                 # No fliers
        whis=(5, 95),                           # 5% and 95% whiskers
    )
    axs[i].set_xlabel(f"{rolling_price_range[n_diff:][bin_labels == i].mean():.0%}")
# ax.scatter(rolling_price_range, rolling_vol_ratio, alpha=0.25)
```

## Aligned data

```python
n_forecast = 21
cumulative_ret = np.cumsum(log_returns[n_long + n_short - 2:])
next_month_ret = cumulative_ret[n_forecast:] - cumulative_ret[:-n_forecast]

# Root explanitory factors
price_range=rolling_price_range[n_short - 1:-n_forecast]
volatility_range=rolling_3mo_std_range[:-n_forecast]
price_vol_range_interaction = (
    (2 * price_range * volatility_range) 
    / (price_range + volatility_range + 1e-6)
)

d_aligned = dict(
    date=dates[n_long + n_short - 2:-n_forecast],
    mean_long_term=rolling_10yr_mean[n_short - 1:-n_forecast] * n_forecast,
    mean_short_term=rolling_3mo_mean[n_long - 1:-n_forecast] * n_forecast,
    std_long_term=rolling_10yr_std[n_short - 1:-n_forecast] * n_forecast**0.5,
    std_short_term=rolling_3mo_std[n_long - 1:-n_forecast] * n_forecast**0.5,
    price_range=price_range,
    volatility_range=volatility_range,
    price_vol_range_interaction=price_vol_range_interaction,
    next_month_ret=next_month_ret,
)

d_aligned = pd.DataFrame(d_aligned).set_index('date')
```

```python
d_aligned.head()
```

```python
d_aligned.tail()
```

## Downsample and split out columns

```python
d_sample = d_aligned.sample(frac=0.2)
d_sample.shape
```

```python
regressors = d_sample[['price_range', 'volatility_range', 'price_vol_range_interaction']].values
regressors = (regressors - 0.5) / 0.5  # Scale to [-1, 1]
print(regressors.shape)

current_means = d_sample[['mean_long_term', 'mean_short_term']].values
current_stds = d_sample[['std_long_term', 'std_short_term']].values
print(current_means.shape, current_stds.shape)

target = d_sample.next_month_ret.values
print(target.shape)
```

## Dristibution of tagret | Market evenironment

```python
decline_flag = in_market_decline(d_sample.index.values, market_declines)
```

```python
df_a, mu_a, sigma_a = stats.t.fit(data=target)
df_r, mu_r, sigma_r = stats.t.fit(target[~decline_flag])
df_d, mu_d, sigma_d = stats.t.fit(target[decline_flag])
print(f"           {'Mean':>8}  {'Std':>6}  {'df':>6}")
print(f"All        {mu_a:>8.4%}  {sigma_a:>6.2%}  {df_a:>6.2}")
print(f"Rising     {mu_r:>8.4%}  {sigma_r:>6.2%}  {df_r:>6.2}")
print(f"Declining  {mu_d:>8.4%}  {sigma_d:>6.2%}  {df_d:>6.2}")
```

```python
fig, axs = plt.subplots(nrows=3, figsize=(10,10), sharex=True)

_, bins, _ = axs[0].hist(
    target, 
    bins=101,
    density=True,
    color="black",
    edgecolor="black", 
    alpha=0.5, 
    label="All returns"
)
_ = axs[0].plot(
    bins,
    stats.norm(loc=mu_a, scale=sigma_a).pdf(bins),
    color="orange",
    linestyle="--",
    linewidth=2,
    label="N(mean_all, std_all)"
)
_ = axs[0].plot(
    bins,
    stats.t(df_a, loc=mu_a, scale=sigma_a).pdf(bins),
    color="black",
    linestyle="--",
    linewidth=2,
    label="t(mean_all, std_all, df_all)"
)
_ = axs[0].legend(loc='upper left')


_ = axs[1].hist(
    target[~decline_flag], 
    bins=bins, 
    density=True,
    color="green",
    edgecolor="black", 
    alpha=0.5, 
    label="Rising market returns"
)
_ = axs[1].plot(
    bins,
    stats.t(df_a, loc=mu_a, scale=sigma_a).pdf(bins),
    color="black",
    linestyle="--",
    linewidth=2,
    label="t(mean_all, std_all, df_all)",
)
_ = axs[1].plot(
    bins,
    stats.t(df_r, loc=mu_r, scale=sigma_r).pdf(bins),
    color="green",
    linestyle="--",
    linewidth=2,
    label="t(mean_rising, std_rising, df_rising)"
)
_ = axs[1].legend(loc='upper left')


axs[2].set_xlabel("log return")
_ = axs[2].hist(
    target[decline_flag], 
    bins=bins, 
    density=True,
    color="red",
    edgecolor="black", 
    alpha=0.5, 
    label="Declining market returns"
)
_ = axs[2].plot(
    bins,
    stats.norm(loc=mu_a, scale=sigma_a).pdf(bins),
    color="black",
    linestyle="--",
    linewidth=2,
    label="t(mean_all, std_all, df_all)",
)
_ = axs[2].plot(
    bins,
    stats.t(df_d, loc=mu_d, scale=sigma_d).pdf(bins),
    color="red",
    linestyle="--",
    linewidth=2,
    label="t(mean_declining, std_declining, df_declining)"
)
_ = axs[2].legend(loc='upper left')
```

## Hierarchical Regression Model


### Model specification

```python
with pm.Model() as model:
    # Skeptical priors for unknown coefficients
    coeffs = pm.Laplace("coeffs", mu=0, b=(1 / 4**0.5), shape=(3,5))
    
    # Apply coeffs to regressors
    # (N, 3, 1) * (1, 3, 5) -> (N, 3, 5)
    wt_regressors = regressors[..., None] * coeffs[None, ...]
    
    # Derive distribution params from weighted regressors
    mu_wts = pm.Deterministic("mu_wts", 0.5 + wt_regressors[..., 1:3].sum(1))
    mu = pm.Deterministic("mu", (current_means * mu_wts).sum(-1))
    
    sigma_wts = pm.Deterministic("sigma_wts", 0.5 + wt_regressors[..., 3:].sum(1))
    sigma = pm.Deterministic("sigma", np.exp((np.log(current_stds) * sigma_wts).sum(-1)))
    
    df = pm.Deterministic("df", np.exp(np.log(5) + wt_regressors[..., 0].sum(-1)))
    
    # Likelihood of latent mixture class
    like = pm.StudentT('like', mu=mu, sigma=sigma, nu=df, observed=target)
    
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
```

```python
trace
```

```python
az.summary(trace.posterior.coeffs)
```

```python
az.plot_trace(trace_mm_model, combined=True)
```

## Posterior probability of latent group membership
P(y_i from grp_k | data) = likelihood_grp_k(y_i) * wt_k / SUM(likelihood_grp_k(y_i) * wt_k, k=0, k=k)

```python
# poserior sample means for both dists & posterior std
mu_a = np.array(trace_mm_model.posterior.mu_a).reshape(-1)
mu_b = np.array(trace_mm_model.posterior.mu_b).reshape(-1)
std = np.array(trace_mm_model.posterior.variance).reshape(-1)**0.5

# Posterior dists for each group
grp_a_dist = stats.norm(loc=mu_a, scale=std)
grp_b_dist = stats.norm(loc=mu_b, scale=std)

# poserior wt of each latent group
wts = np.array(trace_mm_model.posterior.wts).reshape(-1, 2)

mu_a.shape, mu_b.shape, std.shape, wts.shape
```

```python
from typing import List

def get_posterior_membership_probs(
    y: float,
    grp_dists: List[stats.rv_continuous],
    grp_wts: ndarray
) -> ndarray:
    """Returns distribution of posterior membership probability of 
    an observation to each latent group in the mixture
    
    Parameters
    ----------
    y: float
        Observation
    grp_dists: List[stats.rv_continuous]
        Posterior distributions for each group in mixture.
    grp_wts: ndarray, shape=(n_posterior_samples, n_groups)
        Posterior mixture probabilities for groups.
        
    Returns
    -------
    distributions: ndarray, shape=(n_posterior_samples, n_groups)
    """
    
    # Liklihood of membership to each group based on posterior group distributions
    likelihoods = np.stack([dist.pdf(y) for dist in grp_dists], axis=-1)
    
    posterior_membership_probs = (likelihoods * wts) / (likelihoods * wts).sum(axis=-1, keepdims=True)
    
    return posterior_membership_probs
```

```python
y = 0.1

pmp = get_posterior_membership_probs(y, [grp_a_dist, grp_b_dist], wts)
print(pmp.mean(0))

fig, axs = plt.subplots(nrows=2, figsize=(10,10))
_ = axs[0].set_title("observed data")
_ = axs[0].hist(data, bins=30)
_ = axs[0].axvline(x=y, color="red")

bins = np.linspace(0, 1, 100)
_ = axs[1].set_title("Posterior probability distributions of group membership")
_ = axs[1].hist(
    pmp[:,0],
    bins=bins,
    alpha=0.5,
    label="Group A",
)
_ = axs[1].hist(
    np.array(pmp[:,1]),
    bins=bins,
    alpha=0.5,
    label="Group B",
)
_ = axs[1].legend()
```

```python

```

```python

```
