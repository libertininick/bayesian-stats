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

# Conditional Probability


What is the probability that a respondent is a Democrat, given that they are liberal?
```
P(Democrat | liberal)?
```

```python
liberals = sample_population[sample_population.polviews <= PolViews.SlightlyLiberal]
print(f"{len(liberals):,} liberal samples")
print(f"P(Democrat | liberal) = {(liberals.partyid <= Party.NotStrongDemocrat).mean():.2%}")
```

What is the probability that a respondent is female, given that they are a banker?

```python
bankers = sample_population[sample_population.indus10 == 6870]
print(f"{len(bankers):,} banker samples")
print(f"P(female | banker) = {(bankers.sex == Sex.Female).mean():.2%}")
```

## Conditional Probability Is Not Commutative

What is the probability that a respondent is banker, given that they are a female?

```python
females = sample_population[sample_population.sex == Sex.Female]
print(f"{len(females):,} female samples")
print(f"P(banker | female) = {(females.indus10 == 6870).mean():.2%}")
```

## Condition and Conjunction


What probability a respondent is female, given that they are a liberal Democrat?
```
P(female | Democrat & liberal)?
```

```python
liberal_dems = sample_population[
    (sample_population.polviews <= PolViews.SlightlyLiberal) & 
    (sample_population.partyid <= Party.NotStrongDemocrat)
]
print(f"{len(liberal_dems):,} liberal + Democrat samples")
print(f"P(female | liberal, democrat) = {(liberal_dems.sex == Sex.Female).mean():.2%}")
```

```python

```
