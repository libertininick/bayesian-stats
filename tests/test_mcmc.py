"""Test for MCMC module."""
import math
from functools import partial

import pyro.distributions as dist
import pytest
import scipy.stats as stats
import torch
from bayesian_stats.mcmc import ParameterSamples, initialize_samples, run_mcmc
from bayesian_stats.utils import Bounds, get_quantile_diffs
from hypothesis import Verbosity, given, seed, settings, target
from hypothesis import strategies as st
from pytest_check import check
from torch import Tensor

# Test constants
NUM_SAMPLES = 1_024
NUM_PARAMETERS = 3


# Test fixtures
@pytest.fixture(scope="module")
def parameter_bounds() -> dict[str, Bounds]:
    """Get a dictionary of parameter bounds test fixture."""
    return {
        "pi": Bounds(0.0, 1.0),
        "mu": Bounds(-5.0, 2.0),
        "sigma": Bounds(1e-6, 3.0),
    }


@pytest.fixture(scope="module")
def parameter_samples(parameter_bounds: dict[str, Bounds]) -> ParameterSamples:
    """Get parameter samples test fixture."""
    return ParameterSamples.random_initialization(
        bounds=parameter_bounds,
        shapes={k: (1,) for k in parameter_bounds},
        num_samples=NUM_SAMPLES,
        seed=456,
    )


@pytest.fixture(scope="module")
def beta_binomial_parameter_samples() -> ParameterSamples:
    """Get beta-binomial parameter samples test fixture."""
    return ParameterSamples.random_initialization(
        bounds={"p": (0.0, 1.0)},
        shapes={"p": (1,)},
        num_samples=2**13,
        seed=123456,
    )


@pytest.fixture(scope="module")
def proposal_distribution(parameter_samples: ParameterSamples) -> dist.Normal:
    """Get a proposal distribution text fixture."""
    return parameter_samples.proposal_distribution()


# Tests
@pytest.mark.parametrize("num_samples", [3, 5, 6, 7])
def test_initialize_samples_raise_on_bad_num_samples(num_samples: int) -> None:
    """Test `num_samples` that isn't a power of 2 raises an exception."""
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


@pytest.mark.parametrize(
    "a_shape",
    [(1,), (10,), (2, 2), (1, 2), (2, 1), (10, 2, 1, 2), (3, 2, 1), (5, 5)],
)
@pytest.mark.parametrize(
    "b_shape",
    [(1,), (10,), (2, 2), (1, 2), (2, 1), (10, 2, 1, 2), (3, 2, 1), (5, 5)],
)
def test_parameter_samples_shapes(
    a_shape: tuple[int, ...], b_shape: tuple[int, ...]
) -> None:
    """Test that multi-dimension parameters produce correct shapes."""
    num_samples = 8
    psamples = ParameterSamples.random_initialization(
        bounds={"a": (0.0, 1.0), "b": (0.0, 1.0)},
        shapes={"a": a_shape, "b": b_shape},
        num_samples=num_samples,
    )

    # Check that the flattened parameter matrix has correct # columns
    expected_num_cols = math.prod(a_shape) + math.prod(b_shape)
    check.equal(psamples.sample_matrix.shape[-1], expected_num_cols)

    # Check that the samples for each parameter have correct shape
    check.equal(psamples["a"].shape, (num_samples, *a_shape))
    check.equal(psamples["b"].shape, (num_samples, *b_shape))


def test_proposal_distribution_shape(
    proposal_distribution: dist.Normal,
) -> None:
    """Test that proposal distribution has correct shape."""
    assert proposal_distribution.shape() == (NUM_SAMPLES, NUM_PARAMETERS)


def test_proposal_distribution_sample_shape(
    proposal_distribution: dist.Normal,
) -> None:
    """Test a sample from the proposal distribution is correct shape."""
    proposals = proposal_distribution.sample()
    assert proposals.shape == (NUM_SAMPLES, NUM_PARAMETERS)


@pytest.mark.slow
@seed(1234)
@given(
    a=st.integers(min_value=1, max_value=200),
    b=st.integers(min_value=1, max_value=200),
    pos=st.integers(min_value=1, max_value=200),
    neg=st.integers(min_value=1, max_value=200),
)
@settings(max_examples=1_000, deadline=None, verbosity=Verbosity.verbose)
def test_mcmc_beta_binomial(
    a: int,
    b: int,
    pos: int,
    neg: int,
    beta_binomial_parameter_samples: ParameterSamples,
) -> None:
    """Test MCMC samples converge to analytic solution for beta-binomial."""
    n = pos + neg

    def prior_func(a: float, b: float, p: Tensor) -> Tensor:
        return dist.Beta(a, b).log_prob(p)

    def likelihood_func(p: Tensor, n: int, k: int) -> Tensor:
        return dist.Binomial(total_count=n, probs=p).log_prob(torch.tensor(k))

    # Run MCMC sampling
    result = run_mcmc(
        beta_binomial_parameter_samples,
        prior_func=partial(prior_func, a=a, b=b),
        likelihood_func=partial(likelihood_func, n=n, k=pos),
        max_iter=200,
        seed=1234,
        verbose=False,
    )

    # Sample from analytic posterior
    analytic_samples = torch.from_numpy(
        stats.beta(a=a + pos, b=b + neg).rvs(
            size=beta_binomial_parameter_samples.num_samples, random_state=1234
        )
    ).to(torch.float32)

    # Check average quantile difference between MCMC samples and analytic
    # samples is <= 2.5%
    avg_quantile_diff = (
        get_quantile_diffs(
            result.parameter_samples["p"],
            analytic_samples[:, None],
            num_quantiles=100,
        )
        .mean()
        .item()
    )

    target(avg_quantile_diff)
    assert avg_quantile_diff <= 0.025
