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
import numpy as np
import pandas as pd
```

# Bayes's Theorem
`P(A|B) = P(A) * P(B|A) / P(B)`

- If you have the complete populate dataset, you can just filter to the subset's you're interested in and calculate the proportions directly
    - i.e. calculating `P(A|B)` is just as easy as calculating `P(B|A)`
- However, we often never have the complete data or we are asking questions about future unseen events and thus Bayes's theorem is very useful.
- Bayes's theorem gives us a way to update a belief (hypothesis `H`) after observing some data `D_obs`.
- It also gives us a way to update our beliefs about observing some other types of `D_new` given we've seen `D_obs`

```
P(H|D_obs) = P(H) * P(D_obs|H) / P(D_obs)
P(D_new|D_obs) = P(D_obs|H) * P(H|D_obs) / P(H)
```
- `P(H)`: Prior - probability of a hypothesis before seeing the data
- `P(H | D_obs)` : Posterior - probability of a hypothesis after seeing the data
- `P(D_obs | H)` : Likelihood - probability of observing the data if a given hypothesis were true.
- `P(D_obs)`: Unconditional (marginal) probability of observing the data - probability of observing the data irrespective of any hypothesis.
    - This is often the hard part to calculate.
    - It involves summing up all the likelihoods across all possible hypotheses: `P(H1) * P(D|H1) + P(H2) * P(D|H2) + ...`


# Solving the Monty Hall Problem w/ Bayes
- The host shows you three doors; 1 with a car behind in and the other two with goats
- You choose a door at random (assume it's door 1)
- After choosing a door, the host opens one of the other two doors to reveal a goat (assume it's door 2)
    - If you have chosen the door with the car, the host chooses one of the other doors at random.
    - If you have not chose the door with the car, the host opens the other door with the goat behind it (never revealing the car).
- Then, the host offers you the chance to switch your choice of doors.

Should you stick with your original choice or switch doors (to door 3)?

- For this problem, the likelihood is the `P(door 2 opened | car behind door i)`
    - If the car is behind door 1, likelihood = 50% of seeing door 2 opened because the host will open door 2 or 3 at random.
    - If the car is behind door 2, likelihood = 0% of seeing that door opened; that would reveal the car.
    - If the car is behind door 3, likelihood = 100% of seeing door 2 opened; the host cant open door 3 or your chosen door 1.
    
```
      prior  likelihood  posterior
1  0.333333         0.5   0.333333
2  0.333333         0.0   0.000000
3  0.333333         1.0   0.666667
```

```python
priors = np.ones(3) / 3
likelihoods = np.array([0.5, 0, 1])
posteriors = priors * likelihoods
posteriors /= posteriors.sum()

print(pd.DataFrame(
    dict(
        prior=priors,
        likelihood=likelihoods,
        posterior=posteriors
    ),
    index=np.arange(1,4)
))
```

Now suppose the host ALWAYS chooses door 2 to open if possible, and only opens door 3 if door 2 has the car behind it.
- Our priors stay the same, still 1/3, 1/3, 1/3
- Our likelihoods change:
    - `P(open door 2 | car behind 1) = 100%` 
    - `P(open door 2 | car behind 2) = 0%` 
    - `P(open door 2 | car behind 3) = 100%` 
    
```
      prior  likelihood  posterior
1  0.333333           1        0.5
2  0.333333           0        0.0
3  0.333333           1        0.5
```

```python
priors = np.ones(3) / 3
likelihoods = np.array([1, 0, 1])
posteriors = priors * likelihoods
posteriors /= posteriors.sum()

print(pd.DataFrame(
    dict(
        prior=priors,
        likelihood=likelihoods,
        posterior=posteriors
    ),
    index=np.arange(1,4)
))
```

# Which bag of candy?
- Assume there are two bags of colored candy:
    - Bag 1 has 70% green, 20% blue, 7% yellow, 3% red
    - Bag 2 has 50% green, 5% blue, 20% yellow, 25% red
- You choose a bag at random, and a piece at random; it's yellow
- Then from the other bag, you choose a piece at random, it's red

What is the probability the yellow piece came from Bag 1?

- Priors for each bag are 50/50
- Likelihoods are not the raw proportions of yellow, but the likelihood of seeing a yellow vs a red one:
    - Bag 1: likelihood of seeing yellow vs red = `7% / 3%` = `2.33`
    - Bag 2: likelihood of seeing yellow vs red = `20% / 25%` = `0.8`
    
```
   prior  likelihood  posterior
1    0.5    2.333333   0.744681
2    0.5    0.800000   0.255319
```

```python
priors = np.ones(2)
priors /= priors.sum()

likelihoods = np.array([0.07/ 0.03, 0.2/ 0.25])

posteriors = priors * likelihoods
posteriors /= posteriors.sum()

print(pd.DataFrame(
    dict(
        prior=priors,
        likelihood=likelihoods,
        posterior=posteriors
    ),
    index=np.arange(1,3)
))
```

## Taken to the limit
Assume that Bag 1 has 1 yellow candy and 0 red candies, while bag 2 has 1000 yellow candies and 1 red candy.

If we see 1 yellow and 1 red, the yellow has to come from bag 1 and the red from bag 2:

as the # of red candies in bag 1 approaches 0:
```
   prior   likelihood  posterior
1    0.5  100000000.0    0.99999
2    0.5       1000.0    0.00001
```

```python
priors = np.ones(2)
priors /= priors.sum()

likelihoods = np.array([1/1e-8, 1000/1])

posteriors = priors * likelihoods
posteriors /= posteriors.sum()

print(pd.DataFrame(
    dict(
        prior=priors,
        likelihood=likelihoods,
        posterior=posteriors
    ),
    index=np.arange(1,3)
))
```

```python

```
