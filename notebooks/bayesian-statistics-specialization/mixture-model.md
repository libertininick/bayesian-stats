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
data = pd.read_csv(DATA_DIR.joinpath("mixture.csv"), header=None, names=["y"])
missing = data.isnull().values.any(-1)
data = data.loc[~missing, :]
data.shape
```

```python
fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(data, bins=30)
```

## Hierarchical Mixture Model


### Model specification

```python
with pm.Model() as mm_model:
    # Priors for unknown means of mixture distributions
    mu_a = pm.Normal("mu_a", mu=-1, sigma=10)
    mu_b = pm.Normal("mu_b", mu=1, sigma=10)
    
    # Prior for unknown common std of mixture distributions
    sigma = pm.InverseGamma("variance", **get_invgamma_params(variance_prior=1**2, effective_sample_size=1))**0.5
    
    # Component distributions
    components = [
        pm.Normal.dist(mu=mu_a, sigma=sigma),
        pm.Normal.dist(mu=mu_b, sigma=sigma),
    ]
    
    # Prior for mixture wts
    wts = pm.Dirichlet("wts", a=np.array([1.0, 1.0]))
    
    # Likelihood of latent mixture class
    like = pm.Mixture('like', w=wts, comp_dists=components, observed=data.y)
    
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
chains = 5

# Number of samples per chain
draws = 5000

with mm_model:
    # draw posterior samples
    trace_mm_model = pm.sample(draws=draws, chains=chains)
```

```python
az.summary(trace_mm_model)
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
