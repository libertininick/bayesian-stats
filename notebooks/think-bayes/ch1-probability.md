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
import os
from enum import IntEnum
from pathlib import Path

import pandas as pd
```

# Load Data

```python
cwd = Path(os.getcwd())
data_dir = cwd.parents[1].joinpath("data/think-bayes")
if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)
data_dir
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## Download and save
<!-- #endregion -->

```python
(
    pd
    .read_csv("https://github.com/AllenDowney/ThinkBayes2/raw/master/data/gss_bayes.csv")
    .to_csv(data_dir.joinpath("gss_bayes.csv"))
)
```

## Load from local dir

```python
sample_population = pd.read_csv(data_dir.joinpath("gss_bayes.csv"), index_col=0)
sample_population.info()          
```

## Column Encodings

```python
class Sex(IntEnum):
    Male = 1
    Female = 2

class PolViews(IntEnum):
    ExtremelyLiberal = 1
    Liberal = 2
    SlightlyLiberal = 3
    Moderate = 4
    SlightlyConservative = 5
    Conservative = 6
    ExtremelyConservative = 7

class Party(IntEnum):
    StrongDemocrat = 0
    NotStrongDemocrat = 1
    IndependentNearDemocrat = 2
    Independent = 3
    IndependentNearRepublican = 4
    NotStrongRepublican = 5
    StrongRepublican = 6
    OtherParty = 7
```

# Unconditional (marginal) Probability
A marginal probability / marginal probability distribution is a probability that is not dependent (conditioned) on any other event or variable.
- i.e. it is the unconditional probability of an event:
- `P(female)` , `P(banker)` , `P(Democrat)` , `...`
- Typically, we just say "probability" and not "marginal probability" because the marginal part only comes into play when we have to factor in another event or variable(s).
- In those cases, marginal variables are the subset of variables being retained for the probability calculation.
	- The are called "marginal" because they're calculated by summing values in a table along rows or columns, and writing the sum in the margins of the table.
- The other variables are said to have been marginalized out; because we've summed along the rows or columns containing counts for these variables
    - thus, we no longer have any information about the distribution of those variables

```
                            Male  Female  | PartyTotal
==========================================|===========
StrongDemocrat              3252    4632  |       7884
NotStrongDemocrat           4184    5985  |      10169
IndependentNearDemocrat     3064    3113  |       6177
Independent                 3264    3665  |       6929
IndependentNearRepublican   2491    2071  |       4562
NotStrongRepublican         3678    4258  |       7936
StrongRepublican            2438    2491  |       4929
OtherParty                   408     296  |        704
------------------------------------------------------
SexTotal                   22779   26511       
```
- Marginal probability of `Female` = `26511 / 49290`
- Marginal distribution of `Sex` variable: `[22779, 26511]`
- Marginal probability of `StrongDemocrat` = `7884 / 49290`
- Marginal distribution of `PoliticalParty`:
```
StrongDemocrat                7884
NotStrongDemocrat            10169
IndependentNearDemocrat       6177
Independent                   6929
IndependentNearRepublican     4562
NotStrongRepublican           7936
StrongRepublican              4929
OtherParty                     704
```

```python
p = (sample_population.sex == Sex.Female).mean()

print(f"P(female) = {p:.2%}")
```

```python
sex_p_dist = sample_population.sex.value_counts(normalize=True)
sex_p_dist
```

```python
p = (sample_population.partyid == Party.StrongDemocrat).mean()

print(f"P(StrongDemocrat) = {p:.2%}")
```

# Joint Probability
- Joint probability is the probability of two (or more) events happening together or being observed together
- e.g. Probability of a person being a Female AND being a Democrat
- The AND operation means we are combining two probabilities via conjunction
	- The calculation of the joint probability is sometimes called the "product rule" of probability or the "chain rule" of probability.
- For independent events: `P(A,B) = P(A) * P(B)`
- For dependent events/variables with observation overlap in a dataset: `P(A,B) = P(A) * P(B | A)`
- Conjunction is commutative; `P(A,B) = P(B,A)` 
- Joint probability of `Female & StrongDemocrat` = `4632 / 49290`

```python
p = (
    (sample_population.sex == Sex.Female) &
    (sample_population.partyid == Party.StrongDemocrat)
).mean()

print(f"P(StrongDemocrat & female) = {p:.2%}")
```

# Conditional Probability
- A conditional probability is the probability of an event X occurring when a secondary event Y is true
- We are looking at a sub-segment of the total population/ data and asking a probability question within that sub-segment.


What is the probability that a respondent is a `StrongDemocrat`, given that they are a `Female`?

```python
females = sample_population[sample_population.sex == Sex.Female]
p = (liberals.partyid == Party.StrongDemocrat).mean()

print(f"P(StrongDemocrat|female) = {p:.2%}")
```

## Conditional Probability Is Not Commutative
- P(A|B) != P(B|A)


What is the probability that a respondent is female, given that they are a banker?

```python
# P(female | banker)
bankers = sample_population[sample_population.indus10 == 6870]
p = (bankers.sex == Sex.Female).mean()

print(f"{len(bankers):,} banker samples")
print(f"P(female|banker) = {p:.2%}")
```

What is the probability that a respondent is banker, given they are a female?

```python
# P(banker | female)
females = sample_population[sample_population.sex == Sex.Female]
p = (females.indus10 == 6870).mean()

print(f"{len(females):,} female samples")
print(f"P(banker|female) = {p:.2%}")
```

## Condition and Conjunction


What probability a respondent is female, given that they are a `StrongDemocrat` with `liberal` political views?

```python
liberal_strong_dems = sample_population[
    (sample_population.polviews <= PolViews.SlightlyLiberal) & 
    (sample_population.partyid == Party.StrongDemocrat)
]
p = (liberal_strong_dems.sex == Sex.Female).mean()

print(f"{len(liberal_strong_dems):,} liberal + StrongDemocrat samples")
print(f"P(female) = {(sample_population.sex == Sex.Female).mean():.2%}")
print(f"P(female|liberal, StrongDemocrat) = {p:.2%}")
```

## Conditional probability from joint
- The conditional probability is the joint probability normalized by the (marginal) probability of one (or more) variables in the joint distribution
- `P(A|B) = P(A,B) / P(B)`

```python
# P(banker)
p_banker = (sample_population.indus10 == 6870).mean()
print(f"P(banker) = {p_banker:.2%}")

# P(female, banker)
p_female_banker = (
    (sample_population.sex == Sex.Female) &
    (sample_population.indus10 == 6870)
).mean()
print(f"P(female, banker) = {p_female_banker:.2%}")

# P(female | banker)
p_female_given_banker = p_female_banker / p_banker
print(f"P(female | banker) = {p_female_given_banker:.2%}")
```

## Joint probability from conditional
- The probability of two events occurring together can be reframed as probability of the first event, multiplied by the probability of the second event GIVEN the first event has occurred.
- `P(A,B) = P(B) * P(A|B)`

```python
# P(banker)
p_banker = (sample_population.indus10 == 6870).mean()
print(f"P(banker) = {p_banker:.2%}")

# P(female | banker)
bankers = sample_population[sample_population.indus10 == 6870]
p_female_given_banker = (bankers.sex == Sex.Female).mean()
print(f"P(female | banker) = {p_female_given_banker:.2%}")

# P(female, banker)
p_female_banker = p_banker * p_female_given_banker
print(f"P(female, banker) = {p_female_banker:.2%}")
```

# Bayes's Theorem
- Joint probabilities are commutative: `P(A,B) = P(B,A)`
- Joint probabilities are related to conditional probabilities (which are NOT commutative):
    - `P(A,B) = P(A) * P(B|A)`
    - `P(B,A) = P(B) * P(A|B)`
- Thus, `P(A) * P(B|A) = P(B) * P(A|B)`
- Bayes's Theorem: `P(A|B) = P(A) * P(B|A) / P(B)`

```python
# P(female)
p_female = (sample_population.sex == Sex.Female).mean()
print(f"P(female) = {p_female:.2%}")

# P(banker | female)
females = sample_population[sample_population.sex == Sex.Female]
p_banker_given_female = (females.indus10 == 6870).mean()
print(f"P(banker|female) = {p_banker_given_female:.2%}")

# P(banker)
p_banker = (sample_population.indus10 == 6870).mean()
print(f"P(banker) = {p_banker:.2%}")

# P(female | banker)
p_female_given_banker = p_female * p_banker_given_female / p_banker
print(f"P(female | banker) = {p_female_given_banker:.2%}")
```

```python

```
