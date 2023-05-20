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

<!-- #region heading_collapsed=true -->
# Bernoulli likelihood
<!-- #endregion -->

<!-- #region hidden=true -->
## Load data
<!-- #endregion -->

```python hidden=true
DATA_DIR = Path(os.environ.get("DATA_DIR"))
data = pd.read_csv(DATA_DIR.joinpath("urine.csv"))
missing = data.isnull().values.any(-1)
data = data.loc[~missing, :]
print(data.shape)
data.head()
```

<!-- #region hidden=true -->
## Normalize predictors
<!-- #endregion -->

```python hidden=true
x = data.loc[:,['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']]
means, stds = x.mean(0), x.std(0)
x_norm = (x - means) / stds
```

```python hidden=true
means
```

<!-- #region hidden=true -->
## Default Reference Logit Model
<!-- #endregion -->

```python hidden=true
logit_model = sm.Logit(data.r.values, x_norm)
results = logit_model.fit()
results.summary()
```

<!-- #region hidden=true -->
## Bayesian Logit Model
<!-- #endregion -->

```python hidden=true
# Laplace prior
x = np.linspace(-5,5,1000)
fig, ax = plt.subplots(figsize=(10,5))
for s in [0.25, 0.5, 1, 2, 4]:
    ax.plot(x, stats.laplace(loc=0, scale=1/s**0.5).pdf(x), label=s)
_ = ax.legend()
```

<!-- #region hidden=true -->
### Priors
<!-- #endregion -->

```python hidden=true
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

<!-- #region hidden=true -->
### Model specification
<!-- #endregion -->

```python hidden=true
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

```python hidden=true
pm.model_to_graphviz(logit_model)
```

<!-- #region hidden=true -->
### Model fitting
<!-- #endregion -->

```python hidden=true
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

```python hidden=true
az.summary(trace_posterior)
```

```python hidden=true
az.plot_trace(trace_posterior, combined=True)
```

```python hidden=true
axs = az.plot_forest(trace_posterior, var_names=["betas"], combined=True, hdi_prob=0.95, r_hat=True)
axs[0].axvline(x=0, color="red", linestyle="--")
```

<!-- #region hidden=true -->
### Model specification 2
<!-- #endregion -->

```python hidden=true

betas_prior_params_2 = dict(
    mu=0,
    sigma=1000,
)

with pm.Model() as logit_model_2:
    # Priors for unknown model parameters
    intercept = pm.Normal("intercept", **intercept_prior)
    betas = pm.Normal("betas", **betas_prior_params_2, shape=3)

    # Expected value of outcome
    logit_p = intercept + x_norm.values[:, [0,3,5]] @ betas

    # Likelihood (sampling distribution) of observations
    y = pm.Bernoulli("y", logit_p=logit_p, observed=data.r)
    
pm.model_to_graphviz(logit_model_2)
```

<!-- #region hidden=true -->
### Model fitting
<!-- #endregion -->

```python hidden=true
RANDOM_SEED = 58 
rng = np.random.default_rng(RANDOM_SEED)

# Number of chains
chains = 5

# Number of samples per chain
draws = 5000

with logit_model_2:
    # draw posterior samples
    trace_posterior_2 = pm.sample(draws=draws, chains=chains)
    trace_posterior_2 = pm.sample_posterior_predictive(trace_posterior_2, extend_inferencedata=True, random_seed=rng)
```

```python hidden=true
az.summary(trace_posterior_2)
```

# Binomial likelihood


## Load data

```python
DATA_DIR = Path(os.environ.get("DATA_DIR"))
data = pd.read_csv(DATA_DIR.joinpath("OME.csv"))
missing = data.isnull().values.any(-1)
data = data.loc[~missing, :].reset_index(drop=True)
print(data.shape)
data.head()
```

```python
data["success_rate"] = data.Correct / data.Trials
```

### one-hot-encode predictors

```python
x = pd.concat(
    (
        data.loc[:, ["Age", "Loud"]],
        one_hot_encode(data.OME),
        one_hot_encode(data.Noise),
    ),
    axis=1,
)

# Drop OME_high & Noise_coherent
x = x[['Age', 'Loud', 'OME_low','Noise_incoherent']]
```

```python
x.columns
```

```python
fig, axs = plt.subplots(nrows=4, figsize=(12,20))
axs[0].scatter(data.Age, data.success_rate)
axs[1].scatter(data.OME, data.success_rate)
axs[2].scatter(data.Loud, data.success_rate)
axs[3].scatter(data.Noise, data.success_rate)
```

<!-- #region heading_collapsed=true -->
## Binomial GLM
<!-- #endregion -->

```python hidden=true
mask = x.Age == 60
mask = mask & (x.Loud == 50) & (x.OME_low == 0) & (x.Noise_incoherent == 0)
mask.mean()
```

```python hidden=true
data.success_rate[mask].mean()
```

```python hidden=true
help(sm.GLM)
```

```python hidden=true
binomial_glm = sm.GLM(
    endog=data.success_rate,
    exog=sm.add_constant(x),
    family=sm.families.Binomial(),
    freq_weights=data.Trials
)
results = binomial_glm.fit()
results.summary()
```

```python hidden=true
1 / (1 + np.exp(-(-7.2944 + 60*0.0189 + 50*0.1717)))
```

```python hidden=true
p_correct = 1 / (1 + np.exp(-(-7.2944 + 0.0189*x.Age + 0.1717*x.Loud + -0.2372*x.OME_low + 1.5763*x.Noise_incoherent)))
```

## Bayesian Binomial Logit


### Priors

```python
intercept_prior = dict(
    mu=0,
    sigma=3,
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
with pm.Model() as bin_logit_model:
    # Priors for unknown model parameters
    intercept = pm.Normal("intercept", **intercept_prior)
    betas = pm.Laplace("betas", **betas_prior_params, shape=4)

    # Expected value of outcome
    logit_p = intercept + x.values @ betas

    # Likelihood (sampling distribution) of observations
    y = pm.Binomial("y", n=data.Trials, logit_p=logit_p, observed=data.Correct)
    
pm.model_to_graphviz(bin_logit_model)
```

<!-- #region heading_collapsed=true -->
### Model fitting
<!-- #endregion -->

```python hidden=true
RANDOM_SEED = 58 
rng = np.random.default_rng(RANDOM_SEED)

# Number of chains
chains = 5

# Number of samples per chain
draws = 15000

with bin_logit_model:
    # draw posterior samples
    trace_posterior = pm.sample(draws=draws, chains=chains)
    trace_posterior = pm.sample_posterior_predictive(trace_posterior, extend_inferencedata=True, random_seed=rng)
```

```python hidden=true
az.summary(trace_posterior)
```

```python hidden=true
az.plot_trace(trace_posterior, combined=True)
```

```python hidden=true
axs = az.plot_forest(trace_posterior, var_names=["betas"], combined=True, hdi_prob=0.95, r_hat=True)
```

## Hierarchical Binomial Logit

```python
x["id"] = data.ID
x.columns
```

### Model specification

```python
with pm.Model() as h_bin_logit_model:
    # Priors for unknown model parameters
    a_mu = pm.Normal("a_mu", mu=0, sigma=10)
    a_std = pm.InverseGamma("a_var", alpha=1/2, beta=1/2)**0.5
    intercepts = pm.Normal("intercepts", mu=a_mu, sigma=a_std, shape=len(x["id"].unique()))
    betas = pm.Normal("betas", mu=0, sigma=4, shape=4)

    for i, child_id in enumerate(x["id"].unique()):
        x_i = x.query("id == @child_id").loc[:, ['Age', 'OME_low', 'Loud', 'Noise_incoherent']]
        logit_p_i = intercepts[i] + x_i.values @ betas
        y_i = pm.Binomial(
            f"y_{child_id}",
            n=data.query("ID == @child_id").Trials,
            logit_p=logit_p_i,
            observed=data.query("ID == @child_id").Correct
        )
    
    
pm.model_to_graphviz(h_bin_logit_model)
```

### Model fitting

```python
RANDOM_SEED = 58 
rng = np.random.default_rng(RANDOM_SEED)

# Number of chains
chains = 5

# Number of samples per chain
draws = 15000

with h_bin_logit_model:
    # draw posterior samples
    trace_hbl = pm.sample(draws=draws, chains=chains)
```

```python
az.summary(trace_hbl)
```

```python

```
