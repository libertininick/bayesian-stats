"""Test for MCMC module."""
from functools import partial

import pyro.distributions as dist
import pytest
import scipy.stats as stats
import torch
from hypothesis import given, settings, strategies as st, target, Verbosity
from torch import Tensor

from bayesian_stats.mcmc import (
    Bounds,
    get_proposal_distribution,
    initialize_samples,
    run_mcmc,
)
from bayesian_stats.utils import get_max_quantile_diff


# Test constants
NUM_SAMPLES = 1_024
NUM_PARAMETERS = 3


# Test fixtures
@pytest.fixture(scope="module")
def parameter_bounds() -> dict[str, Bounds]:
    """Get a dictionary of parameter bounds test fixture."""
    return dict(pi=Bounds(0.0, 1.0), mu=Bounds(-5.0, 2.0), sigma=Bounds(1e-6, 3.0))


@pytest.fixture(scope="module")
def samples(parameter_bounds: dict[str, Bounds]) -> Tensor:
    """Get parameter samples test fixture."""
    return initialize_samples(
        parameter_bounds.values(), num_samples=NUM_SAMPLES, seed=456
    )


@pytest.fixture(scope="module")
def proposal_dist(samples: Tensor) -> dist.Normal:
    """Get a proposal distribution text fixture."""
    return get_proposal_distribution(samples)


# Tests
@pytest.mark.parametrize("num_samples", [3, 5, 6, 7])
def test_initialize_samples_raise_on_bad_num_samples(num_samples: int) -> None:
    """Test that a `num_samples` that isn't a power of 2 raises an exception."""
    with pytest.raises(ValueError):
        initialize_samples([(0, 1)], num_samples)


def test_initialize_samples_seeded() -> None:
    """Test that seed samples reproduce."""
    expected = torch.tensor(
        [
            [0.9936113953590393, -3.014742136001587],
            [0.26954954862594604, -0.8952667117118835],
            [0.04656250774860382, -3.7364935874938965],
            [0.6903061270713806, 1.5815576314926147],
        ]
    )
    samples = initialize_samples(
        bounds=[(0.0, 1.0), (-5.0, 2.0)], num_samples=4, seed=1234
    )
    assert torch.allclose(expected, samples)


def test_proposal_distribution_shape(proposal_dist: dist.Normal) -> None:
    """Test that proposal distribution has correct shape."""
    assert proposal_dist.shape() == (NUM_SAMPLES, NUM_PARAMETERS)


def test_proposal_distribution_sample_shape(proposal_dist: dist.Normal) -> None:
    """Test a sample from the proposal distribution is correct shape."""
    proposals = proposal_dist.sample()
    assert proposals.shape == (NUM_SAMPLES, NUM_PARAMETERS)


@pytest.mark.slow
@given(
    a=st.integers(min_value=1, max_value=200),
    b=st.integers(min_value=1, max_value=200),
    pos=st.integers(min_value=1, max_value=200),
    neg=st.integers(min_value=1, max_value=200),
)
@settings(max_examples=500, deadline=None, verbosity=Verbosity.verbose)
def test_mcmc_beta_binomial(a: int, b: int, pos: int, neg: int) -> None:
    """Test that MCMC samples converge to analytic solution for beta-binomial."""
    n = pos + neg

    def prior_fun(a: float, b: float, p: Tensor) -> Tensor:
        return dist.Beta(a, b).log_prob(p)

    def likelihood_fun(p: Tensor, n: int, k: int) -> Tensor:
        return dist.Binomial(total_count=n, probs=p).log_prob(torch.tensor(k))

    num_samples = 2**13

    result = run_mcmc(
        parameter_bounds=dict(p=(0.0, 1.0)),
        prior_fun=partial(prior_fun, a=a, b=b),
        likelihood_fun=partial(likelihood_fun, n=n, k=pos),
        num_samples=num_samples,
        max_iter=200,
        seed=1234,
        verbose=False,
    )

    posterior = stats.beta(a=a + pos, b=b + neg)

    analytic_samples = torch.from_numpy(
        posterior.rvs(size=num_samples, random_state=1234)
    ).to(torch.float32)

    qdiff_dist = float(
        get_max_quantile_diff(
            result.get_samples("p")[:, None],
            analytic_samples[:, None],
            num_quantiles=100,
        ).item()
    )

    target(qdiff_dist)
    assert qdiff_dist < 0.03
