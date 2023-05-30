---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python [conda env:bayesian_stats]
    language: python
    name: conda-env-bayesian_stats-py
---

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
```

# What is the probability a coin is biased?
What is the probability a coin is materially biases towards heads ($P(heads) > 52.5\%$), given we flipped it 250 times and observed 140 heads?

![Coin Flipping](artifacts/coin-flipping-p-heads.png)
```
P(p_heads > 52.5% | data)
Non-informative prior    : 86.16%
Strong prior coin is fair: 63.11%
```

```python
# Discrete set of hypotheses for the latent probability of a heads for the coin
n_hypos = 1001
p_heads = np.linspace(0,1,n_hypos)

# Prior distribution for probability of heads
# Non-informative prior
noninform_priors = np.ones(n_hypos)
noninform_priors /= noninform_priors.sum()

# Using a fairly strong prior that the coin is unbiased
strong_priors = stats.beta(a=100, b=100).pdf(p_heads)
strong_priors /= strong_priors.sum()

# Binomial likelihoods
likelihoods = stats.binom.pmf(k=140, n=250, p=p_heads)

# Posteriors
noninform_posteriors = noninform_priors * likelihoods
noninform_posteriors /= noninform_posteriors.sum()

strong_posteriors = strong_priors * likelihoods
strong_posteriors /= strong_posteriors.sum()

df = pd.DataFrame(
    dict(
        noninform_prior=noninform_priors,
        strong_prior=strong_priors,
        likelihood=likelihoods,
        noninform_posterior=noninform_posteriors,
        strong_posterior=strong_posteriors,
    ),
    index=pd.Series(p_heads, name="P(heads)")
)
```

```python
%matplotlib ipympl
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df.index, df.noninform_prior, label="non-inform. prior", color="blue")
ax.plot(df.index, df.strong_prior, label="strong prior", color="orange")
ax.plot(df.index, df.noninform_posterior, label="non-inform. posterior", color="blue", linestyle="--")
ax.plot(df.index, df.strong_posterior, label="strong posterior", color="orange", linestyle="--")
ax.axvline(x=0.5, color="black", alpha=0.5)
ax.axvline(x=0.525, color="black", linestyle=":", label="material threshold")
ax.spines[['right', 'top']].set_visible(False)
ax.set_title(
    "Probability a coin is biased towards heads after observing 140 heads in 250 flips",
    loc='left', 
    fontdict={'fontsize': 12}
)
ax.set_xlabel("P(heads)")
ax.set_ylabel("P(hypothesis)")
ax.legend()
```

```python
# P(p_heads > 52.5% | data)
print(f"Non-informative prior    : {df.noninform_posterior[df.index > 0.525].sum():.2%}")
print(f"Strong prior coin is fair: {df.strong_posterior[df.index > 0.525].sum():.2%}")
```

# Dealing with social desirability bias
Social desirability bias is the tendency for people to change their answer to a question so it's more congruent with social norms. For example, asking people directly if they water their lawns on days the city has requested people refrain from watering to conserve resources will likely illicit untruthful answers from some. People will adjust their answer to show themselves in a more favorable light. This makes it challenging to estimate the true proportion of people who are watering their lawn on days the city has asked them not to. However, if you ask people indirectly whether they are cheating on their lawn watering, you can more accurately estimate this proportion.

For example, ask each surveyed person to flip a (fair) coin so only they can see whether it comes up heads or tails.
- If the coin comes up heads, they write down `YES` as their response to whether they water on non-watering days (irrespective of whether they do or not).
- If they get a tails, they answer truthfully

The introduction of the coin's random outcome and the fact that only the respondent knows if it came up heads or tails, will probably make people answer more truthfully; because, a `YES` response may be from either the coin landing on heads, or because they do cheat on their lawn watering.

Now, suppose you survey 30 people and 21 responded `YES` and 9 `NO` to the question of whether they water on day's the city has asked them no to. What is the posterior distribution for the proportion of people who cheat on their lawn watering?


## Priors for the proportion of lawn watering cheaters

```python

# Discrete set of hypotheses for the latent proportion of people who cheat on lawn watering
n_hypos = 1001
p_cheat = np.linspace(0,1,n_hypos)

# Prior distributions
# Non-informative prior
noninform_prior = stats.beta(a=1, b=1).logpdf(p_cheat)

# Fairly strong prior that people are mostly honest (~94% certain than p_cheat <= 10%)
only_a_few_cheat_prior = stats.beta(a=6, b=94).logpdf(p_cheat)

# Fairly strong prior that a majority of people chat (~95% certain that p_cheat > 60%)
majority_cheat_prior = stats.beta(a=68, b=32).logpdf(p_cheat)

priors = np.stack((noninform_prior, only_a_few_cheat_prior, majority_cheat_prior), axis=-1)
```

```python
%matplotlib ipympl

fig,ax = plt.subplots(figsize=(7,5))
ax.plot(p_cheat, np.exp(noninform_prior), label="non-inform.")
ax.plot(p_cheat, np.exp(only_a_few_cheat_prior), label="only a few cheat")
ax.plot(p_cheat, np.exp(majority_cheat_prior), label="majority cheat")
ax.axvline(x=0.1, color="black", alpha=0.5, linestyle="--")
ax.axvline(x=0.6, color="black", alpha=0.5, linestyle="--")
ax.set_title(
    "Prior beliefs of % of lawn watering cheaters",
    loc='left', 
    fontdict={'fontsize': 12}
)
ax.set_xlabel("Proportion of lawn watering cheaters")
ax.legend()
```

## Likelihood of the data


### Modeling the randomizing coin
- Given a fair coin, we'd expect 15 people to flip a heads and 15 to flip a tails, on average. 
- Therefore our initial guess (using just the data) is 15 people flipped a tails and 6 responded `YES`.
- This gives us an estimate that ~40% of people cheat on their lawn watering. 
- However, we really need the distribution of possible outcomes of 30 people flipping the coin,
- We need to take that uncertainty into account when estimating the proportion of people who flipped a tails and then answered `YES`. 
- Specifically, we need the likelihood of flipping 0 through 21 heads over 30 trials given a fair coin.

```python
p_heads = 0.5
n_trials = 30

n_heads = np.arange(0, 21 + 1)
head_log_likelihoods = stats.binom.logpmf(k=n_heads, n=n_trials, p=p_heads)
```

```python
fig,ax = plt.subplots(figsize=(7,5))
ax.plot(n_heads, np.exp(head_log_likelihoods))
ax.spines[['right', 'top']].set_visible(False)
ax.set_title(
    "Likelihood of flipping `X` heads in 30 trials, given a fair coin",
    loc='left', 
    fontdict={'fontsize': 12}
)
ax.set_xlabel("# Heads")
ax.set_ylabel("Likelihood")
```

### Likelihoods across coin outcomes
Rather than modeling the likelihood of a single observed number of `YES` responses given each hypothesis of the proportion of lawn watering cheaters, we need to account for the 22 different mixes of heads and tails that might have occurred due to the impact of the randomizing coin, because we don't get to see the outcome of each coin toss.

```python
n_yes = 21 - n_heads
n_tail_trails = n_trials - n_heads

log_likelihoods = stats.binom.logpmf(
    k=n_yes[..., None],
    n=n_tail_trails[..., None],
    p=p_cheat[None,...]
)
```

### Combined likelihood
Now we can combine the likelihood of each of the 22 mixes of heads and tails we got from the randomizing coin with the likelihoods of each each hypothesis. The likelihood of each head/tail mix weights the likelihoods of the proportion hypothesis. Then we can marginalize out the actual head/tail outcomes by summing up the weighted hypotheses.

```python
# Combine likehoods of head/tail mixes with hypotheses likelihoods
combined_likelhoods = head_log_likelihoods[..., None] + log_likelihoods

# Sum over head/tail mixes
combined_likelhoods = combined_likelhoods.sum(axis=0)
```

## Posteriors

```python
# Combine likelihood with prior
posteriors = priors + combined_likelhoods[..., None]

# Normalize
posteriors -= posteriors.max(axis=0, keepdims=True)
posteriors = np.exp(posteriors)
posteriors /= posteriors.sum(axis=0, keepdims=True)
```

```python
fig,ax = plt.subplots(figsize=(7,5))
ax.plot(p_cheat, posteriors[:,0], label="non-inform.")
ax.plot(p_cheat, posteriors[:,1], label="only a few cheat")
ax.plot(p_cheat, posteriors[:,2], label="majority cheat")
ax.axvline(x=0.1, color="black", alpha=0.5, linestyle="--")
ax.axvline(x=0.6, color="black", alpha=0.5, linestyle="--")
ax.legend()
```

```python

```
